from duckduckgo_search import DDGS
from pprint import pprint
import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp
import json
import wikipedia
from aiohttp import ClientSession

def get_search_results(query, max_results=5):
    """
    Get search results from DuckDuckGo search engine
    :param query:
    :param max_results:
    :return: title, href, body, __len__
    """
    results = DDGS().text(query, max_results=max_results)
    output_file="results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to {output_file}")
    return results

def save_content_to_file(content, filename):
    # Save the content to a file
    # Just test
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Content saved to {filename}")
    except Exception as e:
        print(f"Failed to save content: {e}")

async def fetch_webpage_content_through_url(session, url):
    """
    Fetch the content of a webpage
    :param session: aiohttp.ClientSession object
    :param url: URL of the webpage
    :return: content of the webpage
    """
    try:
        async with session.get(url) as response:
            page_content = await response.text()
            soup = BeautifulSoup(page_content, "html.parser")

            # Get the content of the webpage
            paragraphs = soup.find_all('p')
            content = "\n".join([p.get_text() for p in paragraphs])
            return content
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None

async def fetch_webpage_content_through_result(session, result):
    """
    Fetch the content of a webpage. If the URL is from Wikipedia, use the wikipedia library instead.
    :param session: aiohttp.ClientSession object
    :param result: search result
    :return: content of the webpage
    """
    url = result.get('href')

    # Check if the URL is a Wikipedia link
    if 'wikipedia.org' in url:
        try:
            # Extract the Wikipedia page title from the URL
            page_title = url.split('/')[-1].replace('_', ' ')
            # Use the wikipedia library to fetch the page content
            page = wikipedia.page(page_title, auto_suggest=False)
            return page.content
        except Exception as e:
            print(f"Failed to fetch Wikipedia page for {url}: {e}")
            return result.get('body')
    else:
        # If the URL is not a Wikipedia link, proceed with the regular fetching process
        try:
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
            }
            async with session.get(url, timeout=10, headers=headers) as response:
                page_content = await response.text()
                soup = BeautifulSoup(page_content, "html.parser")

                # Get the content of the webpage
                paragraphs = soup.find_all('p')
                content = "\n".join([p.get_text() for p in paragraphs])
                return content
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            return result.get('body')

    
async def search_and_process_async(query, max_results=5):
    results = get_search_results(query, max_results=max_results)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for result in results:
            # tasks.append(fetch_webpage_content(session, result.get('href')))
            tasks.append(fetch_webpage_content_through_result(session, result))

        contents = await asyncio.gather(*tasks)
        """
        Save the contents to a single file
        """
        # with open('contents.txt', 'w', encoding='utf-8') as f:
        #     f.write("\n\n".join(contents))

        """
        Save the contents to separate files
        """
        # for content in contents:
        #     if content:
        #         # process_content(content)
        #         save_content_to_file(content, f"{contents.index(content)+1}.txt")

        return contents

def process_content(content):
    print(f"Webpage Content:\n{content[:500]}...")

def save_results_as_json(results, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    search_query = "Teddy bear description"

    # search_results = get_search_results(search_query)
    # save_results_as_json( search_results, 'search_results.json')

    # search_results = get_search_results(search_query)
    # for result in search_results:
    #     print(f"Title: {result.get('title')}")
    #     print(f"URL: {result.get('href')}\n")

    asyncio.run(search_and_process_async(search_query))