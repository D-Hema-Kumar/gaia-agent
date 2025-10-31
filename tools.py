import re

from langchain.tools import tool
from langchain_community.document_loaders import ArxivLoader, WikipediaLoader


@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results."""

    docs_loaded = WikipediaLoader(query=query, load_max_docs=2).load()

    if not docs_loaded:
        return "No results found"

    return docs_loaded


@tool
def arXiv_search(query: str) -> str:
    """Search arXiv for a query and return maximum 2 results."""
    docs_loaded = ArxivLoader(query=query, load_max_docs=2).load()

    if not docs_loaded:
        return "No results found"

    return docs_loaded


def format_response(response: str) -> str:
    formatted_response = re.search(r"FINAL ANSWER:\s*(.*)", response, re.IGNORECASE)

    if formatted_response:
        return formatted_response.group(1).strip()

    else:
        return response


if __name__ == "__main__":
    res = wiki_search.invoke("What is the capital of France?")

    print(f"\n-----The wiki_search result is ---- \n\n {res}")
