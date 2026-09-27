import os
import cloudscraper
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document

# Load environment variables (e.g. GITHUB_TOKEN if needed later)
load_dotenv()


def fetch_clean_article(url: str) -> Document:
    """Fetches raw HTML using cloudscraper to bypass 403 anti-bot protections

    and parses the article text into a standard LangChain Document object.
    """
    # Create a scraper instance that bypasses Cloudflare anti-bot blocks
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "desktop": True}
    )

    # Fetch the webpage HTML
    response = scraper.get(url, timeout=15)
    response.raise_for_status()

    # Parse raw HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Remove clutter elements (scripts, styling, headers, footers)
    for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
        element.decompose()

    # Extract all text from paragraph tags (<p>)
    paragraphs = soup.find_all("p")
    article_text = "\n\n".join(
        [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
    )

    # Return structured LangChain Document object containing content & source metadata
    return Document(page_content=article_text, metadata={"source": url})


def run_data_ingestion():
    print("--- 🚚 Starting Supply Chain Data Ingestion ---")

    # Real-world target supply chain URLs
    supply_chain_urls = [
        "https://www.supplychaindive.com/news/shippers-are-exploring-port-to-inland-transit-to-de-risk-supply-chains/831261/",
        "https://www.freightwaves.com/news/china-lead-time-concerns-surge-as-shippers-widen-global-sourcing-networks",
    ]

    print(f"\n[1/2] Fetching content from {len(supply_chain_urls)} URLs...")
    documents = []

    for url in supply_chain_urls:
        try:
            doc = fetch_clean_article(url)
            documents.append(doc)
            print(f"  ✅ Successfully scraped: {url}")
        except Exception as e:
            print(f"  ❌ Failed to fetch {url}: {e}")

    # Learning Checkpoint: Output inspection
    print("\n--- 🔍 LEARNING CHECKPOINT: Scraped Output Inspection ---")
    for i, doc in enumerate(documents, start=1):
        print(f"\n📄 Document #{i}:")
        print(f"  • Source URL: {doc.metadata.get('source')}")
        print(f"  • Total Character Count: {len(doc.page_content)}")

        # Display first 300 characters of clean text
        preview = doc.page_content.replace("\n", " ")[:300]
        print(f"  • Content Preview:\n    \"{preview}...\"")
        print("  " + "-" * 60)


if __name__ == "__main__":
    run_data_ingestion()