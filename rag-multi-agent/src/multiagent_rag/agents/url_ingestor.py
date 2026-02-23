import os
import requests
from bs4 import BeautifulSoup
from multiagent_rag.agents.base_ingestor import BaseIngestor
from multiagent_rag.utils.chunker import Chunker
from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)

class URLIngestor(BaseIngestor):
    def __init__(self):
        self.chunker = Chunker()

    def process(self, url: str) -> list:
        try:
            logger.info(f"Fetching content from URL: {url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()

            text = soup.get_text(separator=' ')
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)

            if not clean_text.strip():
                logger.warning(f"No textual content extracted from URL: {url}")
                return []

            metadata = {
                "source": url,
                "type": "url"
            }

            return self.chunker.split_text(clean_text, metadata)

        except Exception as e:
            logger.error(f"Failed to process URL {url}: {str(e)}")
            return []
