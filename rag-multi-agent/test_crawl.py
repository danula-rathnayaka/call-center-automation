import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

async def main():
    browser_cfg = BrowserConfig(headless=True, verbose=False)
    run_cfg = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, word_count_threshold=10)
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun("https://ieee.iit.ac.lk/", config=run_cfg)
        print("Success:", result.success)
        print("Type of result.markdown:", type(result.markdown))
        if hasattr(result.markdown, "fit_markdown"):
            print("fit_markdown len:", len(result.markdown.fit_markdown or ""))
        if hasattr(result.markdown, "raw_markdown"):
            print("raw_markdown len:", len(result.markdown.raw_markdown or ""))
        if isinstance(result.markdown, str):
            print("markdown len:", len(result.markdown))
        
        print("has markdown_v2?", hasattr(result, "markdown_v2"))
        
        print("\nUsing what we have:")
        text = result.markdown.fit_markdown if hasattr(result.markdown, "fit_markdown") and result.markdown.fit_markdown else getattr(result.markdown, "raw_markdown", "") if hasattr(result.markdown, "raw_markdown") else str(result.markdown)
        print("Extracted text len:", len(text))
        
asyncio.run(main())
