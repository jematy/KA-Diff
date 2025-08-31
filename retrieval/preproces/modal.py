from prompt import prompt_summary_body, prompt_summary
import time
from datetime import datetime
import csv
import asyncio
from fake_useragent import UserAgent

class WebSearch:
    def __init__(self):
        from duckduckgo_search import DDGS
        import requests
        from bs4 import BeautifulSoup
        import asyncio
        import aiohttp
        import wikipedia
        from aiohttp import ClientSession
        self.DDGS = DDGS
        self.BeautifulSoup = BeautifulSoup
        self.aiohttp = aiohttp
        self.wikipedia = wikipedia
        self.asyncio = asyncio
        self.ClientSession = ClientSession

    def get_search_results(self, query, max_results=5):
        """
        Get search results from DuckDuckGo search engine
        :param query: search query string
        :param max_results: maximum number of results to fetch
        :return: search results
        """
        # headers = {
        # "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0"
        # }
        ua = UserAgent()
        headers = {
            "User-Agent": ua.random,
            # you can add other headers if needed
        }
        results = self.DDGS(headers=headers).text(query, max_results=max_results, backend="google")
        return results

    def save_content_to_file(self, content, filename):
        """Save the content to a file"""
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(content)
            print(f"Content saved to {filename}")
        except Exception as e:
            print(f"Failed to save content: {e}")

    async def fetch_webpage_body_result(self, session, result):
        """
        Fetch the content of a webpage. If the URL is from Wikipedia, use the wikipedia library instead.
        :param session: aiohttp.ClientSession object
        :param result: search result dictionary
        :return: body of the result or webpage content
        """
        body = result.get('body')
        if body:
            return body 
        else:
            url = result.get('href')
            if 'wikipedia.org' in url:
                try:
                    title = result.get('title')
                    summary = self.wikipedia.summary(title)
                    return summary
                except Exception as e:
                    print(f"Failed to fetch Wikipedia content for {url}: {e}")
                    return result.get('body')
            else:
                return await self.fetch_webpage_content_through_url(session, url)
            
    async def fetch_webpage_content_through_result(self, session, result):
        """
        Fetch the content of a webpage. If the URL is from Wikipedia, use the wikipedia library instead.
        :param session: aiohttp.ClientSession object
        :param result: search result dictionary
        :return: content of the webpage
        """
        url = result.get('href')
        if 'wikipedia.org' in url:
            try:
                title = result.get('title')
                summary = self.wikipedia.summary(title)
                return summary
            except Exception as e:
                print(f"Failed to fetch Wikipedia content for {url}: {e}")
                return result.get('body')
        else:
            # If the URL is not a Wikipedia link, proceed with the regular fetching process
            try:
                headers = {
                    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
                }
                async with session.get(url, timeout=10, headers=headers) as response:
                    page_content = await response.text()
                    soup = self.BeautifulSoup(page_content, "html.parser")

                    # Get the content of the webpage
                    paragraphs = soup.find_all('p')
                    content = "\n".join([p.get_text() for p in paragraphs])
                    return content
            except Exception as e:
                print(f"Failed to fetch {url}: {e}")
                return result.get('body')

    async def search_and_process_async_web_search(self, query, max_results=5):
        """
        Perform a search query and asynchronously fetch the content of the results.
        :param query: search query string
        :param max_results: maximum number of results to fetch
        :return: list of contents fetched from the results
        """
        results = self.get_search_results(query, max_results=max_results)
        async with self.ClientSession() as session:
            tasks = []
            for result in results:
                tasks.append(self.fetch_webpage_content_through_result(session, result))

            contents = await self.asyncio.gather(*tasks)
            if not tasks or all(content is None or content == '' for content in contents):
                raise ValueError(f"search api return no results for query: {query}")
            return contents
        
    async def search_and_process_async_body(self, query, max_results=5):
        """
        Perform a search query and asynchronously fetch the content of the results.
        :param query: search query string
        :param max_results: maximum number of results to fetch
        :return: list of contents fetched from the results
        """
        results = self.get_search_results(query, max_results=max_results)
        async with self.ClientSession() as session:
            tasks = []
            for result in results:
                tasks.append(self.fetch_webpage_body_result(session, result))

            contents = await self.asyncio.gather(*tasks)
            return contents
        
    def process_content(self, content):
        """Print the first 500 characters of the content"""
        print(f"Webpage Content:\n{content[:500]}...")

    def save_results_as_json(self, results, filename):
        """Save search results to a JSON file"""
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {filename}")


class TextAnalyzer_summary_body:
    def __init__(self, api_key=None, base_url=None, model_name_api="gpt-4o-mini", local_model_path=None, device="cpu"):
        self.api_key = ""
        self.base_url = ""
        self.model_name_api = model_name_api
        self.local_model_path = local_model_path
        self.device = device
        self.demo_prompt = prompt_summary_body.prompt.strip()

    def get_model_result_with_api(self, user_input="", object_name="", model_name="gpt-4o-mini", retries=3, delay=2):
        """Function to call OpenAI API and get a response for the text analysis."""
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        if isinstance(user_input, list):
            user_input = "\n".join(user_input)

        dynamic_prompt = f"Keyword: {object_name}\n\nContent:\n\n{user_input}\n\nExpected Output:"

        messages = [
            {"role": "system", "content": self.demo_prompt},
            {"role": "user", "content": dynamic_prompt},
        ]

        # response = client.chat.completions.create(
        #     model=model_name,
        #     messages=messages,
        # )

        attempt = 0
        while attempt < retries:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                )

                base_model_response = response.choices[0].message.content.strip()
                return base_model_response
            except Exception as e:
                attempt += 1
                print(f"API request failed (Attempt {attempt}/{retries}): {e}")
                if attempt < retries:
                    time.sleep(delay)  
                else:
                    raise RuntimeError(f"Failed to get a response from API after {retries} attempts.") from e

    def get_model_result_with_local(self, user_input="", object_name=""):
        """Function to use local LLaMA model and get a response for the text analysis."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(self.local_model_path).to(self.device)
        
        if isinstance(user_input, list):
            user_input = "\n".join(user_input)
        
        dynamic_prompt = f"Keyword: {object_name}\n\nContent:\n\n{user_input}\n\nExpected Output:"

        messages = [
            {"role": "system", "content": self.demo_prompt},
            {"role": "user", "content": dynamic_prompt},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = base_model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

class TextAnalyzer_summary_all:
    def __init__(self, api_key=None, base_url=None, model_name_api="gpt-4o-mini", local_model_path=None, device="cpu"):
        self.api_key = ""
        self.base_url = ""
        self.model_name_api = model_name_api
        self.local_model_path = local_model_path
        self.device = device
        self.demo_prompt = prompt_summary.prompt.strip()

    def get_model_result_with_api(self, user_input="", object_name="", model_name="gpt-4o-mini", retries=3, delay=2):
        """Function to call OpenAI API and get a response for the text analysis."""
        from openai import OpenAI
        
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        if isinstance(user_input, list):
            user_input = "\n".join(user_input)

        dynamic_prompt = f"""
        Keyword: {object_name}

        Website Content:

        {user_input}

        Expected Output:
        """
        dynamic_prompt = f"Keyword: {object_name}\n\nWebsite Content:\n\n{user_input}\n\nExpected Output:"
        
        messages = [
            {"role": "system", "content": self.demo_prompt},
            {"role": "user", "content": dynamic_prompt},
        ]

        attempt = 0
        while attempt < retries:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                )

                base_model_response = response.choices[0].message.content.strip()
                return base_model_response
            except Exception as e:
                attempt += 1
                print(f"API request failed (Attempt {attempt}/{retries}): {e}")
                if attempt < retries:
                    time.sleep(delay)  
                else:

                    raise RuntimeError(f"Failed to get a response from API after {retries} attempts.") from e

    def get_model_result_with_local(self, user_input="", object_name=""):
        """Function to use local LLaMA model and get a response for the text analysis."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(self.local_model_path).to(self.device)
        
        if isinstance(user_input, list):
            user_input = "\n".join(user_input)

        dynamic_prompt = f"Keyword: {object_name}\n\nWebsite Content:\n\n{user_input}\n\nExpected Output:"
        
        messages = [
            {"role": "system", "content": self.demo_prompt},
            {"role": "user", "content": dynamic_prompt},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = base_model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response