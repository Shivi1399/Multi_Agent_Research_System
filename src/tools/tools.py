from langchain.tools import tool
import requests
from dotenv import load_dotenv
import os
from tavily import TavilyClient
from bs4 import BeautifulSoup
from readability import Document
import trafilatura
import re


# Load environment variables (e.g. TAVILY_API_KEY) from a .env file into os.environ
load_dotenv()

# Tavily is a search API built for LLM agents — it returns clean, ready-to-use results
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """
    Perform a web search using the Tavily API.

    Args:
        query (str): The search query.

    Returns:
        str: The search results.
    """
    try:
        # Ask Tavily for up to 5 results. The response is a dict shaped like:
        # {"query": "...", "results": [{"title": ..., "url": ..., "content": ...}, ...]}
        response = tavily_client.search(query=query, max_results=5)
        output = []

        # Pull the actual list of results out of the dict (NOT iterate the dict itself,
        # which would just give you its string keys like "query", "results", etc.)
        for r in response.get("results", []):
            output.append(
                f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n"
                )

        # Join every formatted result into one big string the LLM can read
        return "\n----\n".join(output)
    except Exception as e:
        # Never let the tool crash the agent — return the error as text instead
        return f"An error occurred while performing the web search: {e}"

@tool
def scrape_url(url: str) -> str:
    """
    Scrape and extract clean readable content from a given URL.
    Uses multiple extraction strategies to ensure the best possible reliability and accuracy of the extracted content.

    Args:
        url (str): The URL to scrape.

    Returns:
        str: The scraped content.
    """
    try:
        # Pretend to be a real browser — many sites block or serve stripped-down
        # pages to requests with no/default User-Agent (e.g. python-requests/x.y)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/58.0.3029.110 Safari/537.3"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # raises HTTPError for 4xx/5xx status codes

        # Bail out early if the URL isn't actually a webpage (e.g. a PDF or image) —
        # running those through HTML parsers below would just produce garbage
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return f"URL does not point to an HTML page (Content-Type: {content_type or 'unknown'})."

        # Use raw bytes so each parser can detect the real encoding itself,
        # instead of trusting requests' guess (which mojibakes undeclared charsets).
        html = response.content

        # --- Strategy 1: trafilatura ---
        # Purpose-built for extracting the main article text from a webpage while
        # discarding menus, ads, related-article widgets, etc. Try this first since
        # it's usually the most accurate.
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            include_formatting=False,
            )
        # Only trust the result if it's non-trivial (guards against pages where
        # trafilatura extracts almost nothing, e.g. JS-heavy or unusual layouts)
        if extracted and len(extracted.strip()) > 200:
            return _clean_and_truncate(extracted)

        # --- Strategy 2: readability + BeautifulSoup ---
        # readability-lxml (same algorithm as Firefox's Reader View) finds the
        # "main content" block of the page and returns it as trimmed-down HTML
        doc = Document(html)
        clean_html = doc.summary()
        soup = BeautifulSoup(clean_html, "html.parser")
        # Strip out leftover non-content tags before extracting text
        for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
            tag.decompose()
        text_content = soup.get_text(separator="\n", strip=True)

        if text_content and len(text_content.strip()) > 200:
            return _clean_and_truncate(text_content)

        # --- Strategy 3: last-resort brute-force parse ---
        # Neither of the smarter extractors worked (or the page didn't have
        # enough content), so just strip the raw HTML down to visible text
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
            tag.decompose()
        text_content = soup.get_text(separator="\n", strip=True)

        if text_content.strip():
            return _clean_and_truncate(text_content)

        # All three strategies failed to find anything usable
        return "Could not extract meaningful content from the URL."

    # Exceptions are ordered from most specific to most general — Python checks
    # them top to bottom, so a more general except above a specific one would
    # "swallow" it and the specific branch would never run.
    except requests.exceptions.Timeout:
        return "The request timed out while trying to fetch the URL."
    except requests.exceptions.HTTPError as e:
        return f"HTTP error occurred while fetching the URL: {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"An error occurred while fetching the URL: {str(e)}"
    except Exception as e:
        return f"An error occurred while scraping the URL: {str(e)}"


def _clean_and_truncate(text: str, limit: int = 5000) -> str:
    """Collapse whitespace and truncate to `limit` chars without cutting a word in half."""
    # Collapse all runs of whitespace (newlines, tabs, multiple spaces) into single spaces
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    # Cut at the limit, then back up to the last full word so we don't return
    # a half-chopped word at the very end
    truncated = cleaned[:limit]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated + "..."
