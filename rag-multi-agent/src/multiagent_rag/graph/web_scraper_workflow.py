import asyncio
import hashlib
import re
from typing import List
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from multiagent_rag.utils.scrape_review_store import queue_page

from multiagent_rag.state.ingestion_state import IngestionState, ScrapedPage
from multiagent_rag.utils.chunker import Chunker
from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)

_MAX_PAGES = 100
_MIN_TEXT_LENGTH = 150
_REQUEST_DELAY = 0.5

_BLOCKED_PATH_PATTERNS = re.compile(
    r"/(login|logout|signin|signup|sign-in|sign-up|register|registration"
    r"|account|my-account|profile|cart|checkout|order|payment|password"
    r"|reset|activate|verify-email|unsubscribe|admin|dashboard)(\/|$|\?)",
    re.IGNORECASE,
)

_BLOCKED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".zip", ".exe", ".mp4", ".mp3", ".css", ".js",
}

_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a content relevance classifier for a telecom company's customer support AI. "
     "Decide if a webpage contains information useful for answering customer questions about "
     "telecom products, services, plans, troubleshooting, billing, devices, or network coverage.\n\n"
     "Respond ONLY in this exact format — two lines, nothing else:\n"
     "SCORE: <integer 1-5>\n"
     "REASON: <one sentence>\n\n"
     "5 — Directly useful: FAQs, plan details, troubleshooting, pricing, coverage info\n"
     "4 — Useful: product descriptions, service terms, device specs, contact info\n"
     "3 — Marginal: general company info, press releases, blog posts\n"
     "2 — Unlikely useful: partner pages, events, generic marketing\n"
     "1 — Not useful: error pages, empty pages, unrelated content"),
    ("human", "URL: {url}\nTitle: {title}\n\nContent preview:\n{preview}"),
])

_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
_classifier = _CLASSIFIER_PROMPT | _llm | StrOutputParser()
_chunker = Chunker()


def _is_allowed_url(url: str, domain: str, robots: RobotFileParser) -> bool:
    parsed = urlparse(url)
    if parsed.netloc != domain:
        return False
    path = parsed.path.lower()
    ext = "." + path.rsplit(".", 1)[-1] if "." in path.split("/")[-1] else ""
    if ext in _BLOCKED_EXTENSIONS:
        return False
    if _BLOCKED_PATH_PATTERNS.search(parsed.path):
        return False
    try:
        if not robots.can_fetch("*", url):
            return False
    except Exception:
        pass
    return True


def _load_robots(seed_url: str) -> RobotFileParser:
    parsed = urlparse(seed_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    rp = RobotFileParser()
    rp.set_url(f"{base_url}/robots.txt")
    try:
        rp.read()
    except Exception as e:
        logger.warning(f"Could not read robots.txt: {e}")
    return rp


def _parse_classifier_response(response: str) -> tuple:
    score = 3
    reason = "classification parsing failed"
    for line in response.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                score = max(1, min(5, int(line.split(":", 1)[1].strip())))
            except ValueError:
                pass
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return score, reason


async def _crawl_site(seed_url: str, domain: str, robots: RobotFileParser) -> List[dict]:
    browser_cfg = BrowserConfig(headless=True, verbose=False)
    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=10,
    )

    visited = set()
    to_visit = [seed_url]
    raw_pages = []

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        while to_visit and len(visited) < _MAX_PAGES:
            batch = [u for u in to_visit if u not in visited][:10]
            to_visit = [u for u in to_visit if u not in batch]

            if not batch:
                break

            results = await crawler.arun_many(batch, config=run_cfg)

            for result in results:
                url = result.url
                visited.add(url)

                if not result.success:
                    logger.warning(f"Crawl failed: {url}")
                    continue

                title = result.metadata.get("title", "") if result.metadata else ""
                text = result.markdown.fit_markdown if result.markdown else ""
                text = re.sub(r"\s{2,}", " ", text).strip()

                links = []
                if result.links:
                    for link_info in result.links.get("internal", []):
                        href = link_info.get("href", "")
                        if href and _is_allowed_url(href, domain, robots):
                            links.append(href)

                raw_pages.append({
                    "url": url,
                    "title": title,
                    "text": text,
                    "links": links,
                })

                new_links = [l for l in links if l not in visited and l not in to_visit]
                to_visit.extend(new_links)

                logger.info(f"Crawled: {url} ({len(text)} chars, {len(new_links)} new links)")

    return raw_pages


def crawler_node(state: IngestionState):
    seed_url = state["file_path"]
    domain = urlparse(seed_url).netloc
    robots = _load_robots(seed_url)

    logger.info(f"Crawler starting from: {seed_url}")

    try:
        raw_pages = asyncio.run(_crawl_site(seed_url, domain, robots))
    except Exception as e:
        logger.error(f"Crawl failed: {e}")
        return {"scraped_pages": [], "status": "failed"}

    scraped = [
        ScrapedPage(
            url=p["url"],
            title=p["title"],
            text=p["text"],
            filter_status="pending",
            ai_score=0,
            ai_reason="",
        )
        for p in raw_pages
    ]

    logger.info(f"Crawl complete: {len(scraped)} pages collected")
    return {"scraped_pages": scraped, "status": "crawled"}


def hard_filter_node(state: IngestionState):
    pages = state.get("scraped_pages", [])
    passed = 0
    blocked = 0

    for page in pages:
        if len(page["text"]) < _MIN_TEXT_LENGTH:
            page["filter_status"] = "blocked_short"
            blocked += 1
            continue
        if _BLOCKED_PATH_PATTERNS.search(page["url"]):
            page["filter_status"] = "blocked_path"
            blocked += 1
            continue
        page["filter_status"] = "passed"
        passed += 1

    logger.info(f"Hard filter: {passed} passed, {blocked} blocked")
    return {"scraped_pages": pages, "status": "filtered"}


def relevance_classifier_node(state: IngestionState):
    pages = state.get("scraped_pages", [])
    to_score = [p for p in pages if p["filter_status"] == "passed"]

    approved = 0
    quarantine = 0
    rejected = 0

    for page in to_score:
        try:
            response = _classifier.invoke({
                "url": page["url"],
                "title": page["title"],
                "preview": page["text"][:800],
            })
            score, reason = _parse_classifier_response(response)
            page["ai_score"] = score
            page["ai_reason"] = reason

            if score >= 4:
                page["filter_status"] = "approved"
                approved += 1
            elif score >= 2:
                page["filter_status"] = "quarantine"
                quarantine += 1
            else:
                page["filter_status"] = "rejected"
                rejected += 1

            logger.info(f"[{score}/5] {page['url']} — {reason}")

        except Exception as e:
            logger.error(f"Classifier failed for {page['url']}: {e}")
            page["filter_status"] = "quarantine"
            page["ai_score"] = 3
            page["ai_reason"] = f"Classifier error: {e}"
            quarantine += 1

    logger.info(f"Classifier: {approved} approved, {quarantine} quarantined, {rejected} rejected")
    return {"scraped_pages": pages, "status": "classified"}


def quarantine_node(state: IngestionState):
    seed_url = state["file_path"]
    pages = state.get("scraped_pages", [])
    queued = 0

    for page in pages:
        if page["filter_status"] != "quarantine":
            continue
        try:
            queue_page(
                url=page["url"],
                title=page["title"],
                preview_text=page["text"][:800],
                full_text=page["text"],
                ai_score=page["ai_score"],
                ai_reason=page["ai_reason"],
                seed_url=seed_url,
            )
            queued += 1
        except Exception as e:
            logger.error(f"Failed to queue page {page['url']}: {e}")

    logger.info(f"Quarantine: {queued} pages sent to review queue")
    return {}


def page_chunker_node(state: IngestionState):
    seed_url = state["file_path"]
    pages = state.get("scraped_pages", [])
    approved = [p for p in pages if p["filter_status"] == "approved"]
    all_chunks = []
    seed_hash = hashlib.sha256(seed_url.encode()).hexdigest()

    for page in approved:
        doc_hash = hashlib.sha256(page["text"].encode()).hexdigest()
        metadata = {
            "source": page["url"],
            "title": page["title"],
            "type": "web",
            "document_hash": doc_hash,
            "seed_hash": seed_hash,
        }
        try:
            chunks = _chunker.split_text(page["text"], metadata)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Chunking failed for {page['url']}: {e}")

    logger.info(f"Chunker: {len(all_chunks)} total chunks from {len(approved)} approved pages")

    if not all_chunks:
        return {"chunks": [], "document_hash": seed_hash, "status": "failed"}

    return {"chunks": all_chunks, "document_hash": seed_hash, "status": "extracted"}


def route_after_crawl(state: IngestionState) -> str:
    if state.get("status") == "failed":
        return "end"
    return "hard_filter"


_scraper_workflow = StateGraph(IngestionState)

_scraper_workflow.add_node("crawler", crawler_node)
_scraper_workflow.add_node("hard_filter", hard_filter_node)
_scraper_workflow.add_node("relevance_classifier", relevance_classifier_node)
_scraper_workflow.add_node("quarantine", quarantine_node)
_scraper_workflow.add_node("page_chunker", page_chunker_node)

_scraper_workflow.set_entry_point("crawler")

_scraper_workflow.add_conditional_edges(
    "crawler",
    route_after_crawl,
    {"hard_filter": "hard_filter", "end": END},
)

_scraper_workflow.add_edge("hard_filter", "relevance_classifier")
_scraper_workflow.add_edge("relevance_classifier", "quarantine")
_scraper_workflow.add_edge("quarantine", "page_chunker")
_scraper_workflow.add_edge("page_chunker", END)

web_scraper_app = _scraper_workflow.compile()
