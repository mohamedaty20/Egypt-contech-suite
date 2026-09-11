import os
import re
import requests
import cloudscraper
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "").strip()
JSEARCH_HOST = "jsearch.p.rapidapi.com"
REQUEST_TIMEOUT = 25

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

scraper = cloudscraper.create_scraper()
scraper.headers.update(HEADERS)


def log(msg):
    print(f"[JOB-SCRAPER] {msg}")


def detect_mime_type(filename: str, data: bytes) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.png']:
        return 'image/png'
    elif ext in ['.jpg', '.jpeg']:
        return 'image/jpeg'
    elif ext in ['.pdf']:
        return 'application/pdf'
    if data.startswith(b'\x89PNG'):
        return 'image/png'
    if data.startswith(b'\xff\xd8'):
        return 'image/jpeg'
    if data.startswith(b'%PDF'):
        return 'application/pdf'
    return 'image/jpeg'


# =====================================================================
# JSearch (RapidAPI) — only runs if RAPIDAPI_KEY is set
# =====================================================================
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
                        break
        except Exception as e:
            log(f"JSearch exception: {e}")
    log(f"JSearch parsed {len(jobs)} jobs")
    return jobs


# =====================================================================
# Wuzzuf direct scrape (broadened selectors)
# =====================================================================
def scrape_wuzzuf_direct(query, max_results=20):
    jobs = []
    urls_to_try = [
        f"https://wuzzuf.net/search/jobs/?q={quote_plus(query)}&a=hpb",
        f"https://wuzzuf.net/search/jobs/?q={quote_plus(query)}",
    ]
    soup = None
    for url in urls_to_try:
        try:
            resp = scraper.get(url, timeout=REQUEST_TIMEOUT)
            log(f"Wuzzuf GET {url} -> status={resp.status_code}, len={len(resp.text)}")
            if resp.status_code == 200 and len(resp.text) > 2000:
                soup = BeautifulSoup(resp.text, "html.parser")
                break
        except Exception as e:
            log(f"Wuzzuf request failed: {e}")
    if soup is None:
        return jobs

    # Broadened selectors — multiple URL patterns used by Wuzzuf over time
    anchors = soup.select('a[href*="/jobs/p/"]') or \
              soup.select('a[href*="/jobs/"]') or \
              soup.select('h2 a')

    seen_hrefs = set()
    for a in anchors:
        href = a.get("href")
        title = a.get_text(strip=True)
        if not href or not title or len(title) < 3:
            continue
        if href in seen_hrefs:
            continue
        seen_hrefs.add(href)
        full_url = href if href.startswith("http") else f"https://wuzzuf.net{href}"

        # Walk up to find the containing card
        card = a
        for _ in range(10):
            if card.parent is None:
                break
            card = card.parent
            if len(card.find_all("a")) >= 2:
                break

        company, location, desc = "", "", ""
        if card is not None:
            company_link = card.find("a", href=re.compile(r"/employers/"))
            company = company_link.get_text(strip=True) if company_link else ""
            loc_elem = card.find("span", class_=re.compile(r"location", re.I)) or \
                       card.find("div", class_=re.compile(r"location", re.I))
            if loc_elem:
                location = loc_elem.get_text(strip=True)
            chunks = [t.get_text(strip=True) for t in card.find_all(["span", "div"])
                      if t.get_text(strip=True)]
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


# =====================================================================
# Bayt direct scrape (broadened selectors)
# =====================================================================
def scrape_bayt_direct(query, max_results=20):
    jobs = []
    url = f"https://www.bayt.com/en/egypt/jobs/?search={quote_plus(query)}"
    try:
        resp = scraper.get(url, timeout=REQUEST_TIMEOUT)
        log(f"Bayt GET {url} -> status={resp.status_code}, len={len(resp.text)}")
        if resp.status_code != 200:
            return jobs
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        log(f"Bayt request failed: {e}")
        return jobs

    cards = (soup.select('li.has-pointer') or
             soup.select('div.job-card') or
             soup.select('[data-job-id]') or
             soup.select('article'))

    for card in cards[:max_results]:
        try:
            a = None
            h2 = card.find("h2")
            if h2:
                a = h2.find("a")
            if not a:
                a = card.find("a", href=re.compile(r"/job/"))
            if not a:
                continue
            title = a.get_text(strip=True)
            href = a.get("href")
            if not title or not href:
                continue
            full_url = href if href.startswith("http") else f"https://www.bayt.com{href}"
            company_el = card.select_one(".company-name, .jb-company, .t-default")
            company = company_el.get_text(strip=True) if company_el else "N/A"
            loc_el = card.select_one(".location, .t-mute.t-small, .jb-loc")
            location = loc_el.get_text(strip=True) if loc_el else "Egypt"
            desc_el = card.find("p")
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


# =====================================================================
# GUARANTEED FALLBACK — direct search links to major job sites.
# Runs only if the scrapers above return < 3 jobs.
# =====================================================================
def _search_links_fallback(query):
    q = quote_plus(query)
    return [
        {
            "title": f"🔗 Open full search on Wuzzuf — '{query}'",
            "company": "Wuzzuf (external)",
            "location": "Egypt",
            "description": "Click to open the complete live job results for this query on Wuzzuf.",
            "url": f"https://wuzzuf.net/search/jobs/?q={q}&a=hpb",
            "source": "Search Link",
        },
        {
            "title": f"🔗 Open full search on Bayt — '{query}'",
            "company": "Bayt (external)",
            "location": "Egypt",
            "description": "Click to open the complete live job results for this query on Bayt.",
            "url": f"https://www.bayt.com/en/egypt/jobs/?search={q}",
            "source": "Search Link",
        },
        {
            "title": f"🔗 Open full search on LinkedIn Jobs — '{query}'",
            "company": "LinkedIn (external)",
            "location": "Egypt",
            "description": "Click to open the complete live job results for this query on LinkedIn.",
            "url": f"https://www.linkedin.com/jobs/search/?keywords={q}&location=Egypt",
            "source": "Search Link",
        },
        {
            "title": f"🔗 Open full search on Indeed Egypt — '{query}'",
            "company": "Indeed (external)",
            "location": "Egypt",
            "description": "Click to open the complete live job results for this query on Indeed.",
            "url": f"https://eg.indeed.com/jobs?q={q}",
            "source": "Search Link",
        },
        {
            "title": f"🔗 Open full search on Glassdoor — '{query}'",
            "company": "Glassdoor (external)",
            "location": "Egypt",
            "description": "Click to open the complete live job results for this query on Glassdoor.",
            "url": f"https://www.glassdoor.com/Job/egypt-jobs-SRCH_IL.0,5_IN69_KO6,30.htm?sc.keyword={q}",
            "source": "Search Link",
        },
    ]


# =====================================================================
# Main entry point — unchanged signature
# =====================================================================
def scrape_jobs(query, location=""):
    full_query = f"{query} {location}".strip() if location else query
    log(f"scrape_jobs called: query={full_query!r} (RAPIDAPI_KEY set: {bool(RAPIDAPI_KEY)})")

    all_jobs = []

    # 1. Try JSearch if key is configured
    try:
        all_jobs.extend(scrape_jsearch(full_query))
    except Exception as e:
        log(f"JSearch top-level error: {e}")

    # 2. Try Wuzzuf if we don't have enough results
    if len(all_jobs) < 3:
        log("Trying Wuzzuf direct scrape…")
        try:
            all_jobs.extend(scrape_wuzzuf_direct(full_query))
        except Exception as e:
            log(f"Wuzzuf top-level error: {e}")

    # 3. Try Bayt if still not enough
    if len(all_jobs) < 3:
        log("Trying Bayt direct scrape…")
        try:
            all_jobs.extend(scrape_bayt_direct(full_query))
        except Exception as e:
            log(f"Bayt top-level error: {e}")

    # 4. GUARANTEED fallback — direct search links
    if len(all_jobs) < 3:
        log("Live scrapers returned <3 results — adding external search links.")
        all_jobs.extend(_search_links_fallback(full_query))

    # Deduplicate by URL
    seen = set()
    deduped = []
    for job in all_jobs:
        url = job.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(job)

    # Real Wuzzuf/Bayt/JSearch jobs first; search links last
    order = {"Wuzzuf": 0, "Bayt": 1, "LinkedIn": 2, "JSearch": 3, "Search Link": 9}
    deduped.sort(key=lambda j: order.get(j.get("source", ""), 5))

    log(f"TOTAL jobs after dedup: {len(deduped)}")
    return deduped
