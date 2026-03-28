import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from flashrank import Ranker, RerankRequest
from utils import get_logger

logger = get_logger("search_utils")

class SearchAndRank:
    """
    Retrieval-Augmented Researcher that fetches, scrapes, and reranks web results.
    """
    def __init__(self, model_name="ms-marco-TinyBERT-L-2-v2"):
        # FlashRank uses a 4MB TinyBERT by default for speed/efficiency
        self.ranker = Ranker(model_name=model_name)
        
    def search_web(self, query, max_results=5):
        """
        Fetches live search results from DuckDuckGo.
        """
        results = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
        return results

    def scrape_content(self, url):
        """
        Scrapes text content from a URL.
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.extract()
                
            text = soup.get_text(separator=' ', strip=True)
            # Limit to first 2000 chars for context window safety
            return text[:2000]
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return ""

    def get_best_results(self, query, top_k=3):
        """
        Executes Search -> Scrape -> Rerank pipeline.
        """
        search_results = self.search_web(query)
        if not search_results:
            return []
            
        passages = []
        for i, res in enumerate(search_results):
            # We use the search snippet as the initial text
            # Scraping the full page would be better but slower
            # Let's use snippet + title for the reranker
            passages.append({
                "id": str(i),
                "text": f"{res['title']}: {res['body']}",
                "meta": {"url": res['href'], "title": res['title']}
            })
            
        # Rerank based on relevancy to the query
        rerank_request = RerankRequest(query=query, passages=passages)
        ranked_results = self.ranker.rerank(rerank_request)
        
        # Take the top_k results
        return ranked_results[:top_k]

# Singleton
_search_rank_engine = None

def get_search_engine():
    global _search_rank_engine
    if _search_rank_engine is None:
        _search_rank_engine = SearchAndRank()
    return _search_rank_engine
