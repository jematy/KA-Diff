import os
import asyncio
import csv
from tqdm import tqdm
from datetime import datetime
from duckduckgo_search import DDGS
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests as request
import time
from fake_useragent import UserAgent
import pandas as pd

proxy_url = ''


def fetch_webpage_body_result(result):
    body = result.get('body')
    return body if body else None


def get_proxy(retries=3, delay=2):
    """get proxy from proxy_url"""
    for attempt in range(retries):
        try:
            proxy_resp = request.get(proxy_url, timeout=5)
            proxy_data = proxy_resp.json().get("data", []) if proxy_resp.status_code == 200 else []
            if proxy_data:
                return proxy_data
        except Exception as e:
            print(f"Failed to get proxy on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return None


def get_summary(word1, host, port):
    word2 = "description"
    query = word1 + " " + word2
    proxies = f'http://{host}:{port}'
    ua = UserAgent()
    headers = {
        "User-Agent": ua.random,
    }
    time.sleep(0.001)
    # print(proxies)
    try:
        results = DDGS(proxy=proxies, headers=headers, timeout=20).text(query, max_results=5)
        tasks = [fetch_webpage_body_result(result) for result in results]
        if not tasks or all(content is None or content == '' for content in tasks):
            raise ValueError(f"search api returned no results for query: {query}")
        return tasks
    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('error_log.csv', 'a', encoding='utf-8', newline='') as error_file:
            writer = csv.writer(error_file)
            if error_file.tell() == 0:
                writer.writerow(["query", "timestamp", "error_message"])
            writer.writerow([query, timestamp, str(e)])
        return None


def process_row(row, headers, proxy):
    row = [str(field).replace('\n', ' ').replace('\t', ' ') for field in row]
    caption_reference_description = row[headers.index('page_title')]
    host, port = proxy.get("ip"), proxy.get("port")
    summary_list = get_summary(caption_reference_description, host, port)
    if summary_list is None:
        summary_columns = [""] * 5
    else:
        summary_list = [" ".join(item.splitlines()) for item in summary_list]
        summary_columns = summary_list[:5] + [""] * (5 - len(summary_list))
    row.extend(summary_columns)
    return row


def process_file(input_file, output_file):
    data = pd.read_csv(input_file, sep='\t', quoting=csv.QUOTE_ALL, encoding='utf-8')
    headers = list(data.columns) + [f'summary{i + 1}' for i in range(5)]
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write('\t'.join(headers) + '\n')

        # load proxy
        proxy_data = get_proxy()
        if not proxy_data:
            print("No proxies available. Exiting program.")
            return

        total_lines = len(data)
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = []
            proxy_index = 0
            for _, row in tqdm(data.iterrows(), total=total_lines, desc=f"Processing {input_file}", unit="line"):
                row = row.tolist()

                # check proxy data and rotate
                while not proxy_data:
                    proxy_data = get_proxy()
                    print("reget proxy")
                if proxy_index >= len(proxy_data):
                    proxy_index = 0
                    proxy_data = get_proxy()
                    while not proxy_data:
                        print("Failed to retrieve proxies. Exiting program.")
                        proxy_data = get_proxy()

                proxy = proxy_data[proxy_index]
                proxy_index += 1

                # submit task to thread pool
                futures.append(executor.submit(process_row, row, headers, proxy))

            # collect and write processed results
            for future in as_completed(futures):
                processed_row = future.result()
                outfile.write('\t'.join(processed_row) + '\n')
                outfile.flush()


if __name__ == "__main__":
    input_directory = "split_files"
    output_directory = "output_files"
    os.makedirs(output_directory, exist_ok=True)

    # get all tsv files, we split the original tsv file into several parts and process them one by one, to prevent some unexpected errors
    input_files = [f for f in os.listdir(input_directory) if f.startswith("file_part_") and f.endswith(".tsv")]

    for input_filename in input_files:
        input_file = os.path.join(input_directory, input_filename)
        output_filename = input_filename.replace(".tsv", "_summary.tsv")
        output_file = os.path.join(output_directory, output_filename)
        process_file(input_file, output_file)

    print("All files processed.")