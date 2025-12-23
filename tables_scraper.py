"""
Simplified scraper - TABLES ONLY
"""

import asyncio
from asyncio import Semaphore
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError, Browser
import random
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup, Tag

# ============================================================================
# Browser Pool
# ============================================================================

class BrowserPool:
    """Manages a pool of shared browsers with tab-based concurrency"""
    
    def __init__(self, pool_size: int = 2, max_tabs_per_browser: int = 10):
        self.pool_size = pool_size
        self.max_tabs_per_browser = max_tabs_per_browser
        self.max_concurrent = pool_size * max_tabs_per_browser
        
        self.browsers: List[Browser] = []
        self.browser_semaphores: List[Semaphore] = []
        self.tab_counts = []
        self.lock = asyncio.Lock()
        
        self.playwright = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the browser pool"""
        if self.initialized:
            return
        
        print(f"🚀 Initializing browser pool: {self.pool_size} browsers × {self.max_tabs_per_browser} tabs")
        
        self.playwright = await async_playwright().start()
        
        for i in range(self.pool_size):
            browser = await self.playwright.chromium.launch(
                headless=True,
                args=['--disable-dev-shm-usage', '--no-sandbox']
            )
            self.browsers.append(browser)
            self.browser_semaphores.append(Semaphore(self.max_tabs_per_browser))
            self.tab_counts.append(0)
        
        self.initialized = True
        print(f"✅ Browser pool ready (capacity: {self.max_concurrent} tabs)")
    
    async def get_browser_and_semaphore(self) -> tuple[int, Browser, Semaphore]:
        """Get least-loaded browser"""
        if not self.initialized:
            await self.initialize()
        
        async with self.lock:
            browser_idx = min(range(len(self.browsers)), key=lambda i: self.tab_counts[i])
            self.tab_counts[browser_idx] += 1
        
        return browser_idx, self.browsers[browser_idx], self.browser_semaphores[browser_idx]
    
    async def release_browser(self, browser_idx: int):
        """Release a tab slot"""
        async with self.lock:
            self.tab_counts[browser_idx] -= 1
    
    async def close(self):
        """Close all browsers"""
        print("🧹 Closing browser pool...")
        for browser in self.browsers:
            try:
                await browser.close()
            except Exception as e:
                print(f"⚠️ Error closing browser: {e}")
        
        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception as e:
                print(f"⚠️ Error stopping playwright: {e}")
        
        self.browsers = []
        self.browser_semaphores = []
        self.tab_counts = []
        self.initialized = False
        print("✅ Browser pool closed")

# ============================================================================
# Table Extraction
# ============================================================================

def _normalize_cell(s: Optional[str]) -> str:
    """Normalize cell text"""
    if s is None:
        return ""
    return " ".join(s.split()).strip()

def _cell_text(cell: Tag, soup: BeautifulSoup) -> str:
    """Extract best text from table cell"""
    # Try image alt text
    img = cell.find("img")
    if img and img.get("alt"):
        return _normalize_cell(img.get("alt"))
    
    # Try aria-label
    if cell.has_attr("aria-label"):
        return _normalize_cell(cell["aria-label"])
    
    # Try data attributes
    for attr in ["data-value", "data-text", "title"]:
        if cell.has_attr(attr) and cell[attr]:
            return _normalize_cell(cell[attr])
    
    # Default to text content
    return _normalize_cell(cell.get_text(separator=" ", strip=True))

def html_table_to_md(html: str, title: str = "") -> Optional[str]:
    """Convert HTML table to markdown"""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return None
    
    # Extract caption
    caption_tag = table.find("caption")
    caption = _normalize_cell(caption_tag.get_text()) if caption_tag else ""
    
    # Extract rows
    rows: List[List[str]] = []
    for tr in table.find_all("tr"):
        cells = [_cell_text(cell, soup) for cell in tr.find_all(["th", "td"])]
        if any(c for c in cells):
            rows.append(cells)
    
    if not rows:
        return None
    
    # Detect header row
    header_cells = []
    first_tr = table.find("tr")
    if first_tr and first_tr.find("th"):
        header_cells = [_cell_text(th, soup) for th in first_tr.find_all("th")]
        rows = rows[1:]  # Remove header from rows
    elif rows:
        header_cells = rows[0]
        rows = rows[1:]
    
    if not header_cells:
        # Auto-generate headers
        max_cols = max(len(r) for r in rows) if rows else 0
        header_cells = [f"Col{i+1}" for i in range(max_cols)]
    
    # Normalize row lengths
    header_len = len(header_cells)
    normalized_rows = []
    for r in rows:
        row = list(r)
        if len(row) < header_len:
            row += [""] * (header_len - len(row))
        elif len(row) > header_len:
            row = row[:header_len]
        normalized_rows.append(row)
    
    # Skip empty tables
    if not any(any(cell.strip() for cell in r) for r in normalized_rows):
        return None
    
    # Build markdown
    def esc(s: str) -> str:
        return s.replace("|", "\\|")
    
    header_line = "| " + " | ".join(esc(h) for h in header_cells) + " |"
    sep_line = "| " + " | ".join("---" for _ in header_cells) + " |"
    row_lines = ["| " + " | ".join(esc(cell) for cell in r) + " |" for r in normalized_rows]
    
    # Add title if present
    md_parts = []
    ttl = title or caption
    if ttl:
        md_parts.append(f"**{_normalize_cell(ttl)}**\n")
    md_parts.append("\n".join([header_line, sep_line] + row_lines))
    
    return "\n".join(md_parts)

async def extract_tables_from_page(page) -> List[str]:
    """Extract all tables from page as markdown"""
    js = """
    () => {
        const tables = Array.from(document.querySelectorAll('table'));
        return tables.map(t => {
            const caption = t.querySelector('caption')?.innerText?.trim() || '';
            
            // Find preceding heading
            let prev = t.previousElementSibling;
            let heading = '';
            for (let i = 0; i < 5 && prev; i++) {
                const tag = prev.tagName?.toLowerCase();
                if (tag && /^h[1-6]$/.test(tag)) {
                    heading = prev.innerText?.trim() || '';
                    break;
                }
                prev = prev.previousElementSibling;
            }
            
            return {
                html: t.outerHTML,
                caption: caption,
                heading: heading
            };
        });
    }
    """
    
    try:
        tables = await page.evaluate(js)
    except Exception:
        return []
    
    md_tables = []
    seen = set()
    
    for t in tables:
        html = t.get("html", "")
        title = t.get("caption", "") or t.get("heading", "")
        
        md = html_table_to_md(html, title=title)
        if not md or md in seen:
            continue
        
        seen.add(md)
        md_tables.append(md)
    
    return md_tables

# ============================================================================
# Scraping
# ============================================================================

async def scrape_tables_from_url(page, url: str, timeout: int = 60000) -> Optional[List[str]]:
    """Scrape only tables from a URL"""
    print(f"🌐 Scraping tables from: {url}")
    
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
    except PlaywrightTimeoutError:
        print(f"⚠️ [{url}] Timeout - continuing anyway")
    except Exception as e:
        print(f"❌ [{url}] Navigation error: {e}")
        return None
    
    # Wait a bit for dynamic content
    await asyncio.sleep(2)
    
    # Extract tables
    try:
        tables = await extract_tables_from_page(page)
        if tables:
            print(f"✅ [{url}] Found {len(tables)} tables")
        else:
            print(f"ℹ️ [{url}] No tables found")
        return tables
    except Exception as e:
        print(f"❌ [{url}] Table extraction error: {e}")
        return None

async def worker_scrape_tables(
    url: str,
    browser_pool: BrowserPool,
    timeout: int = 60000
) -> Tuple[str, Optional[List[str]]]:
    """Worker that scrapes tables from one URL"""
    
    browser_idx, browser, semaphore = await browser_pool.get_browser_and_semaphore()
    
    async with semaphore:
        page = await browser.new_page()
        
        # Set user agent
        ua = (
            f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            f"AppleWebKit/537.36 Chrome/{random.randint(110,140)}.0.0.0 Safari/537.36"
        )
        
        try:
            await page.set_extra_http_headers({
                "Accept-Language": "en-US,en;q=0.9",
                "User-Agent": ua
            })
            await page.set_viewport_size({"width": 1366, "height": 768})
        except Exception:
            pass
        
        try:
            tables = await scrape_tables_from_url(page, url, timeout=timeout)
            return url, tables
        finally:
            try:
                await page.close()
            except Exception:
                pass
            await browser_pool.release_browser(browser_idx)

async def scrape_tables_parallel(
    urls: List[str],
    browser_pool: BrowserPool = None,
    timeout: int = 60000
) -> Dict[str, List[str]]:
    """Scrape tables from multiple URLs in parallel"""
    
    if not urls:
        return {}
    
    # Initialize pool if needed
    if not browser_pool or not browser_pool.initialized:
        num_urls = len(urls)
        pool_size = min(3, max(2, num_urls // 5))
        
        browser_pool = BrowserPool(pool_size=pool_size, max_tabs_per_browser=10)
        await browser_pool.initialize()
    
    # Create tasks
    tasks = [
        asyncio.create_task(
            worker_scrape_tables(url, browser_pool, timeout=timeout),
            name=f"scrape-{i}"
        )
        for i, url in enumerate(urls)
    ]
    
    # Gather results
    results = {}
    for coro in asyncio.as_completed(tasks):
        try:
            url, tables = await coro
            if tables:
                results[url] = tables
                print(f"✅ {url} -> {len(tables)} tables")
            else:
                results[url] = []
        except Exception as e:
            print(f"❌ Worker error: {e}")
    
    return results