import os
import re
import requests
import cloudscraper
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "").strip()
JSEARCH_HOST = "jsearch.p.rapidapi.com"
REQUEST_TIMEOUT = 30

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

def scrape_wuzzuf_direct(query, max_results=20):
    jobs = []
    url = f"https://wuzzuf.net/search/jobs/?q={quote_plus(query)}&a=hpb"
    try:
        resp = scraper.get(url, timeout=REQUEST_TIMEOUT)
        log(f"Wuzzuf GET {url} -> status={resp.status_code}, len={len(resp.text)}")
        if resp.status_code != 200:
            log(f"Wuzzuf non-200 body preview: {resp.text[:200]}")
            return jobs
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        log(f"Wuzzuf request failed: {e}")
        return jobs

    for a in soup.select('a[href*="/jobs/p/"]'):
        href = a.get("href")
        title = a.get_text(strip=True)
        if not href or not title:
            continue
        full_url = href if href.startswith("http") else f"https://wuzzuf.net{href}"
        card = a
        for _ in range(8):
            card = card.parent
            if card is None:
                break
            if len(card.find_all("a")) >= 2:
                break
        company, location, desc = "", "", ""
        if card is not None:
            company_link = card.find("a", href=re.compile(r"/employers/"))
            company = company_link.get_text(strip=True) if company_link else ""
            loc_elem = card.find("span", class_=re.compile(r"location", re.I)) or card.find("div", class_=re.compile(r"location", re.I))
            if loc_elem:
                location = loc_elem.get_text(strip=True)
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

    if not jobs:
        log(f"Wuzzuf: No job links found. HTML snippet: {re.sub(r'\s+', ' ', str(soup))[:500]}")
    else:
        log(f"Wuzzuf parsed {len(jobs)} jobs")
    return jobs

def scrape_bayt_direct(query, max_results=20):
    jobs = []
    url = f"https://www.bayt.com/en/egypt/jobs/?search={quote_plus(query)}"
    try:
        resp = scraper.get(url, timeout=REQUEST_TIMEOUT)
        log(f"Bayt GET {url} -> status={resp.status_code}, len={len(resp.text)}")
        if resp.status_code != 200:
            log(f"Bayt non-200 body preview: {resp.text[:200]}")
            return jobs
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        log(f"Bayt request failed: {e}")
        return jobs

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
    if not jobs:
        log(f"Bayt: No job cards found. HTML snippet: {re.sub(r'\s+', ' ', str(soup))[:500]}")
    else:
        log(f"Bayt parsed {len(jobs)} jobs")
    return jobs

def scrape_jobs(query, location=""):
    full_query = f"{query} {location}".strip() if location else query
    all_jobs = []
    try:
        all_jobs.extend(scrape_jsearch(full_query))
    except Exception as e:
        log(f"JSearch top-level error: {e}")
    if len(all_jobs) < 3:
        log("JSearch returned few results – trying Wuzzuf.")
        try:
            all_jobs.extend(scrape_wuzzuf_direct(full_query))
        except Exception as e:
            log(f"Wuzzuf top-level error: {e}")
    if len(all_jobs) < 3:
        log("Wuzzuf also returned few results – trying Bayt.")
        try:
            all_jobs.extend(scrape_bayt_direct(full_query))
        except Exception as e:
            log(f"Bayt top-level error: {e}")
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
