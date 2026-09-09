import io
import datetime
import os
import uuid
import re
import asyncio
import json
import time
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import qrcode
import pypdf
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

from dotenv import load_dotenv
from fastapi import FastAPI
from nicegui import app, ui, run

from google import genai
from google.genai import types

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    Image as ReportLabImage,
)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

GEMINI_MODEL = "gemini-3.5-flash-lite"

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 32
USABLE_WIDTH = PAGE_WIDTH - (2 * MARGIN)

# =====================================================================================
# JOB SCRAPING FUNCTIONS (ULTRA-ROBUST)
# =====================================================================================
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "").strip()
JSEARCH_HOST = "jsearch.p.rapidapi.com"
REQUEST_TIMEOUT = 20

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://wuzzuf.net/",
    "DNT": "1",
}

def log(msg):
    print(f"[JOB-SCRAPER] {msg}")

# ---------- PRIMARY: JSearch ----------
def scrape_jsearch(query, max_results=20):
    jobs = []
    if not RAPIDAPI_KEY:
        log("JSearch: No API key – skipping.")
        return jobs

    params_list = [
        {"query": query, "page": "1", "num_pages": "1", "engine": "google_jobs"},
        {"query": f"{query} Egypt", "page": "1", "num_pages": "1", "engine": "google_jobs"},
        {"query": query, "page": "1", "num_pages": "1", "country": "eg", "engine": "google_jobs"},
    ]
    for params in params_list:
        try:
            resp = requests.get(
                f"https://{JSEARCH_HOST}/search",
                headers={"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": JSEARCH_HOST},
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            log(f"JSearch {params} -> status={resp.status_code}, len={len(resp.text)}")
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    for item in data[:max_results]:
                        title = item.get("job_title")
                        if not title:
                            continue
                        company = item.get("employer_name") or "N/A"
                        city = item.get("job_city") or ""
                        country = item.get("job_country") or "Egypt"
                        location = ", ".join(p for p in [city, country] if p) or "Egypt"
                        desc = (item.get("job_description") or "No description")[:400]
                        url = item.get("job_apply_link") or item.get("job_google_link") or ""
                        source = item.get("job_publisher") or "JSearch"
                        if url:
                            jobs.append({
                                "title": title,
                                "company": company,
                                "location": location,
                                "description": desc,
                                "url": url,
                                "source": source,
                            })
                    if jobs:
                        break  # success
        except Exception as e:
            log(f"JSearch exception: {e}")

    log(f"JSearch parsed {len(jobs)} jobs")
    return jobs

# ---------- FALLBACK: Wuzzuf direct ----------
def scrape_wuzzuf_direct(query, max_results=20):
    jobs = []
    url = f"https://wuzzuf.net/search/jobs/?q={quote_plus(query)}&a=hpb"
    session = requests.Session()
    session.headers.update(HEADERS)
    session.cookies.set("wuzzuf_session", "1")
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        log(f"Wuzzuf GET {url} -> status={resp.status_code}, len={len(resp.text)}")
        if resp.status_code != 200:
            log(f"Wuzzuf non-200 body preview: {resp.text[:200]}")
            return jobs
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        log(f"Wuzzuf request failed: {e}")
        return jobs

    # Strategy 1: find all <a> with href containing '/jobs/p/'
    for a in soup.select('a[href*="/jobs/p/"]'):
        href = a.get("href")
        title = a.get_text(strip=True)
        if not href or not title:
            continue
        full_url = href if href.startswith("http") else f"https://wuzzuf.net{href}"
        # Find parent card – look for a container with multiple links
        card = a
        for _ in range(8):
            card = card.parent
            if card is None:
                break
            if len(card.find_all("a")) >= 2:
                break
        company, location, desc = "", "", ""
        if card is not None:
            # company
            company_link = card.find("a", href=re.compile(r"/employers/"))
            company = company_link.get_text(strip=True) if company_link else ""
            # location
            loc_elem = card.find("span", class_=re.compile(r"location", re.I)) or card.find("div", class_=re.compile(r"location", re.I))
            if loc_elem:
                location = loc_elem.get_text(strip=True)
            # description: collect all text from spans/divs not already captured
            chunks = [t.get_text(strip=True) for t in card.find_all(["span", "div"]) if t.get_text(strip=True)]
            chunks = [t for t in chunks if t not in (title, company, location)]
            desc = " | ".join(dict.fromkeys(chunks))[:400]
        jobs.append({
            "title": title,
            "company": company or "N/A",
            "location": location or "Egypt",
            "description": desc or "No description preview.",
            "url": full_url,
            "source": "Wuzzuf",
        })
        if len(jobs) >= max_results:
            break

    log(f"Wuzzuf parsed {len(jobs)} jobs")
    return jobs

# ---------- FALLBACK 2: Bayt ----------
def scrape_bayt_direct(query, max_results=20):
    jobs = []
    url = f"https://www.bayt.com/en/egypt/jobs/?search={quote_plus(query)}"
    session = requests.Session()
    session.headers.update(HEADERS)
    session.cookies.set("bayt_session", "1")
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        log(f"Bayt GET {url} -> status={resp.status_code}, len={len(resp.text)}")
        if resp.status_code != 200:
            log(f"Bayt non-200 body preview: {resp.text[:200]}")
            return jobs
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        log(f"Bayt request failed: {e}")
        return jobs

    # Bayt cards: <li class="has-pointer"> or <div class="job-card">
    cards = soup.select('li.has-pointer') or soup.select('div.job-card')
    for card in cards[:max_results]:
        try:
            a = card.find("h2") and card.find("h2").find("a")
            if not a:
                a = card.find("a", href=re.compile(r"/job/"))
            if not a:
                continue
            title = a.get_text(strip=True)
            href = a.get("href")
            full_url = href if href.startswith("http") else f"https://www.bayt.com{href}"
            company_el = card.select_one(".company-name, .jb-company")
            company = company_el.get_text(strip=True) if company_el else "N/A"
            loc_el = card.select_one(".location, .t-mute.t-small")
            location = loc_el.get_text(strip=True) if loc_el else "Egypt"
            desc_el = card.select_one("p")
            desc = desc_el.get_text(strip=True) if desc_el else "No description."
            jobs.append({
                "title": title,
                "company": company,
                "location": location,
                "description": desc,
                "url": full_url,
                "source": "Bayt",
            })
        except Exception as e:
            log(f"Bayt card parse error: {e}")
    log(f"Bayt parsed {len(jobs)} jobs")
    return jobs

# ---------- MAIN ENTRY ----------
def scrape_jobs(query, location=""):
    full_query = f"{query} {location}".strip() if location else query
    all_jobs = []

    # 1. JSearch (primary)
    try:
        all_jobs.extend(scrape_jsearch(full_query))
    except Exception as e:
        log(f"JSearch top-level error: {e}")

    # 2. If JSearch returned < 3, try Wuzzuf
    if len(all_jobs) < 3:
        log("JSearch returned few results – trying Wuzzuf.")
        try:
            all_jobs.extend(scrape_wuzzuf_direct(full_query))
        except Exception as e:
            log(f"Wuzzuf top-level error: {e}")

    # 3. If still less than 3, try Bayt
    if len(all_jobs) < 3:
        log("Wuzzuf also returned few results – trying Bayt.")
        try:
            all_jobs.extend(scrape_bayt_direct(full_query))
        except Exception as e:
            log(f"Bayt top-level error: {e}")

    # Deduplicate and sort (Wuzzuf first)
    seen = set()
    deduped = []
    for job in all_jobs:
        if job["url"] in seen:
            continue
        seen.add(job["url"])
        deduped.append(job)
    deduped.sort(key=lambda j: 0 if j["source"] == "Wuzzuf" else 1)

    log(f"TOTAL jobs after dedup: {len(deduped)}")
    return deduped


# =====================================================================================
# STYLING, PDF HELPERS, BOQ ENGINE, ETC. – ALL UNCHANGED (omitted for brevity)
# =====================================================================================
# ... (keep everything else exactly as in your previous code, 
#      including all styling, sanitization, PDF generation, BOQ functions, etc.)
# =====================================================================================

# =====================================================================================
# MAIN APP LAYOUT (with Job Board tab)
# =====================================================================================
@ui.page('/')
def main_page():
    ui.query('body').style('width: 100vw; height: 100vh; overflow-x: hidden;')

    # ---------------- SIDEBAR ----------------
    sidebar = ui.left_drawer().classes('sidebar-container').style('width: 380px;')
    with sidebar:
        with ui.row().classes('w-full items-center justify-between mb-4 p-2'):
            ui.label('📋 PROJECT METADATA').classes('text-white font-bold text-base tracking-wide')
            ui.button('✕', on_click=sidebar.toggle).classes(
                'bg-transparent text-white text-xl hover:text-[#FF8C00] p-1 min-w-[36px] !shadow-none !rounded-full !bg-transparent'
            ).style('font-size: 20px; line-height: 1;')

        project_name_input = ui.input(label='Project Name', value='Highway Expansion Project').classes('w-full mb-3')
        pour_location_input = ui.input(label='Structural Element / Chainage', value='Highway Section Ch. 12+500').classes('w-full mb-4')

        ui.label('Governing Design Code Basis').classes('text-white font-bold text-sm mb-1')
        ui.markdown('By default every AI output in this app is generated strictly per **ECP 203 / ECP 202 / ECP 104**. Change this to switch the primary basis.').classes('text-xs text-[#A9B6D0] mb-2')
        code_basis_select = ui.select(
            label='Code Type (applies app-wide)',
            options=CODE_BASIS_OPTIONS,
            value=CODE_BASIS_OPTIONS[0],
        ).classes('w-full mb-4')

        fcu_input = ui.number(label='Specified 28-Day Grade f_cu (N/mm2)', value=30.0, step=5.0).classes('w-full mb-4')

        ui.label('Batch Plant & Site Logs').classes('text-white font-bold text-sm mb-2')
        truck_input = ui.input(label='Mixer Truck No.', value='TRK-104').classes('w-full mb-2')
        ticket_input = ui.input(label='Batch Ticket ID', value='BT-99482').classes('w-full mb-4')

        ui.label('Mix Design Parameters').classes('text-white font-bold text-sm mb-2')
        cement_input = ui.input(label='Cement Content (kg/m3)', value='350.0').classes('w-full mb-2')
        water_input = ui.input(label='Free Water Content (kg/m3)', value='150.0').classes('w-full mb-4')

        engineer_input = ui.input(label='Engineer Name', value='Eng. Mohamed Abd Al Aty').classes('w-full mb-2')

        logo_status = ui.label('Logo: Not uploaded').classes('text-xs text-amber-400 mb-1')
        logo_bytes_holder = {'bytes': None}

        async def handle_logo_upload(e):
            try:
                logo_bytes_holder['bytes'] = await e.file.read()
                logo_status.set_text(f'Logo Loaded: {e.file.name}')
                logo_status.classes(replace='text-xs text-emerald-400 mb-1')
                ui.notify('Company logo loaded successfully!', type='positive')
            except Exception as ex:
                ui.notify(f'Error reading logo: {str(ex)}', type='negative')

        ui.upload(label='Upload Company Logo', auto_upload=True, on_upload=handle_logo_upload).props('flat dark').classes('w-full mb-2')

    ui.button('☰', on_click=sidebar.toggle).classes(
        'fixed top-4 left-4 z-50 bg-[#10203f] text-white border border-[#FF8C00] p-3 rounded-full shadow-lg hover:bg-[#1a2a4a]'
    ).style('font-size: 20px; min-width: 48px; min-height: 48px;')

    def current_meta(uid_prefix):
        return {
            'uid': f"{uid_prefix}-{uuid.uuid4().hex[:8].upper()}",
            'project': project_name_input.value,
            'location': pour_location_input.value,
            'engineer': engineer_input.value,
            'date': datetime.date.today().strftime('%Y-%m-%d'),
            'ticket': ticket_input.value,
        }

    # ---------------- MAIN COLUMN ----------------
    with ui.column().classes('w-full min-h-screen p-4 bg-[#031338]'):
        # Title block
        with ui.column().classes('w-full bg-[#0d1a35] px-6 py-4 rounded-xl border border-[#FF8C00] shadow-lg mb-4'):
            ui.label('SMART EGY-CIVIL AI AUDITOR').classes('main-title text-white')
            ui.label('Intelligent General Civil, Geotechnical & Structural Compliance Engine').classes('sub-title text-lg font-medium mt-1')
            ui.label('Lead Technical Auditor: Eng. Mohamed Abd Al Aty').classes('text-base text-[#A9B6D0] font-semibold mt-1')
            ui.label('Next-generation automated civil engineering and quality intelligence, precision-calibrated for the Egyptian Code of Practice.').classes('text-sm text-[#A9B6D0] mt-1 italic')

        ui.add_head_html('''
        <style>@keyframes marquee { 0% { transform: translate(0, 0); } 100% { transform: translate(-100%, 0); } }</style>
        ''')
        ui.html('''
        <div style="width: 100%; overflow: hidden; white-space: nowrap; background-color: rgba(13,26,53,0.6); backdrop-filter: blur(8px); color: #FFFFFF; padding: 10px 0; font-weight: 600; font-size: 13px; margin-bottom: 15px; border-radius: 8px; border: 1px solid rgba(255,140,0,0.3);">
          <div style="display: inline-block; padding-left: 100%; animation: marquee 28s linear infinite;">
            <span style="color: #FF8C00;">[CORE ACTIVE]</span> ECP 203 &middot; ECP 202 &middot; ECP 104 &middot; ASTM &middot; AASHTO &middot; BS EN &middot; ISO
            &nbsp;&nbsp;|&nbsp;&nbsp; Advanced Geotechnical & Concrete Calculation Sheet &nbsp;&nbsp;|&nbsp;&nbsp; Active Site Inspection Portal
          </div>
        </div>
        ''')

        # Tabs
        with ui.tabs().classes('w-full text-white bg-[#0d1a35] rounded-lg') as tabs:
            t_dash = ui.tab('Concrete Cube Verifier').classes('text-white font-bold')
            t_audit = ui.tab('AI Multi-Standard Auditor').classes('text-white font-bold')
            t_defect = ui.tab('Defect Diagnostic').classes('text-white font-bold')
            t_chat = ui.tab('AI Chatbot').classes('text-white font-bold')
            t_handwriting = ui.tab('Handwriting OCR').classes('text-white font-bold')
            t_jobs = ui.tab('Job Board').classes('text-white font-bold')

        with ui.tab_panels(tabs, value=t_dash).classes('w-full bg-transparent mt-4'):

            # ... (all other tabs remain exactly as they were – copy from your previous working code)
            # For brevity I'm skipping them here, but you MUST keep them.

            # TAB 6: JOB BOARD (updated)
            with ui.tab_panel(t_jobs):
                ui.label('Engineering Job Board - Egypt').classes('text-2xl font-bold text-white mb-4')
                ui.markdown('Search for the latest engineering jobs in Egypt. Uses **JSearch** (RapidAPI) if the key is set, otherwise falls back to direct Wuzzuf and Bayt scraping.').classes('markdown-body mb-2')

                key_status = ui.label(
                    '🔑 RapidAPI key: ' + ('✅ Set' if RAPIDAPI_KEY else '❌ Not set – using Wuzzuf/Bayt fallback.')
                ).classes('text-sm text-[#A9B6D0] mb-2')

                with ui.row().classes('w-full gap-4 mb-4'):
                    search_input = ui.input(label='Search for jobs', placeholder='e.g., Civil Engineer', value='Civil Engineer').classes('flex-1')
                    location_input = ui.input(label='Location (optional)', placeholder='e.g., Cairo').classes('flex-1')
                    search_button = ui.button('Search Jobs', on_click=lambda: search_jobs()).classes('primary-btn')

                filter_input = ui.input(label='Filter results', placeholder='Type to filter title, company, description...', on_change=lambda: filter_jobs()).classes('w-full mb-2')

                results_container = ui.column().classes('w-full')
                jobs_data = []

                def display_jobs(jobs, filter_text=''):
                    results_container.clear()
                    with results_container:
                        if not jobs:
                            ui.label('No jobs found. Try a different search.').classes('text-white')
                            return
                        filtered = jobs
                        if filter_text:
                            f_lower = filter_text.lower()
                            filtered = [j for j in jobs if f_lower in j['title'].lower() or f_lower in j['company'].lower() or f_lower in j['description'].lower()]
                        if not filtered:
                            ui.label('No jobs match the filter.').classes('text-white')
                            return
                        for job in filtered:
                            with ui.card().classes('w-full bg-[#0d1a35] border border-[#2c3f6b] rounded-lg p-3 mb-2'):
                                with ui.row().classes('w-full justify-between'):
                                    ui.label(job['title']).classes('text-lg font-bold text-white')
                                    ui.label(job['company']).classes('text-sm text-[#A9B6D0]')
                                ui.label(job['location']).classes('text-sm text-[#A9B6D0]')
                                desc = job['description'][:200] + ('...' if len(job['description']) > 200 else '')
                                ui.label(desc).classes('text-sm text-white mt-1')
                                ui.link('View Job', job['url'], new_tab=True).classes('text-[#4FC3F7] hover:text-[#FF8C00]')

                def filter_jobs():
                    display_jobs(jobs_data, filter_input.value.strip())

                async def search_jobs():
                    query = search_input.value.strip()
                    if not query:
                        ui.notify('Please enter a search term.', type='warning')
                        return
                    location = location_input.value.strip()
                    if location:
                        query += f' {location}'
                    ui.notify(f'Searching for "{query}"...', type='info')
                    results_container.clear()
                    with results_container:
                        ui.spinner('ios', size='lg').classes('self-center text-[#4FC3F7]')
                        ui.label('Fetching job listings...').classes('self-center text-sm')

                    jobs = await run.io_bound(scrape_jobs, query)
                    jobs_data.clear()
                    jobs_data.extend(jobs)
                    display_jobs(jobs_data)

        # ---------------- FOOTER ----------------
        ui.html('''
        <div class="app-footer">
            <b>Multi-Standard Engineering Quality Assurance Portal</b> &nbsp;|&nbsp; Automated compliance verification across ECP 203, ECP 202, ECP 104, ASTM, AASHTO, BS, EN, and ISO standards.<br>
            <b>Official Direct Contacts:</b>
            LinkedIn: <a href="https://www.linkedin.com/in/mohamed-abd-al-aty-a326a1214/" target="_blank">Mohamed Abd Al Aty</a> &nbsp;|&nbsp;
            Email: <a href="mailto:mohamedabdalaty63@gmail.com">mohamedabdalaty63@gmail.com</a><br>
            <i>Specialized in QA/QC, Civil Engineering Standards &amp; Automated Compliance.</i> &copy; 2026 Eng. Mohamed Abd Al Aty. All rights reserved.<br>
            <span style="color: #FFFFFF; font-weight: 600;">Disclaimer:</span> These AI modules have high accuracy and are specified for the Egyptian codes, but results should be rechecked by a qualified engineer before any decision-making.
        </div>
        ''')


ui.run(
    host='0.0.0.0',
    port=int(os.environ.get('PORT', 8080)),
    title='Multi-Standard Engineering Auditor',
    favicon='🏗️',
    reload=False,
    reconnect_timeout=30.0,
)
