import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
from duckduckgo_search import DDGS
import requests as request
import time
from fake_useragent import UserAgent

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
    try:
        results = DDGS(proxy=proxies, headers=headers, timeout=20).text(query, max_results=5, backend="google")
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


def is_summary_missing(row):
    summary_columns = [f'summary{i + 1}' for i in range(5)]
    # print(row)
    for col in summary_columns:
        val = row[col]
        if pd.notna(val) and str(val).strip():
            return False
    return True


def reprocess_row(idx, row, headers, proxy):
    caption_reference_description = row['page_title']
    host, port = proxy.get("ip"), proxy.get("port")
    summary_list = get_summary(caption_reference_description, host, port)
    if summary_list is None:
        summary_columns_data = [""] * 5
    else:
        summary_list = [" ".join(item.splitlines()) for item in summary_list]
        summary_columns_data = summary_list[:5] + [""] * (5 - len(summary_list))
    for i in range(5):
        row[f'summary{i + 1}'] = summary_columns_data[i]
    return idx, row


def reprocess_missing_summaries(output_file):
    df_output = pd.read_csv(output_file, sep='\t', quoting=csv.QUOTE_ALL, encoding='utf-8', on_bad_lines='skip')
    headers = list(df_output.columns)
    summary_columns = [f'summary{i + 1}' for i in range(5)]

    # Identify rows with missing summaries
    df_missing_summaries = df_output[df_output.apply(is_summary_missing, axis=1)]
    if df_missing_summaries.empty:
        print("No missing summaries found.")
        return False
    print(f"Found {len(df_missing_summaries)} rows with missing summaries.")

    # Load proxies
    proxy_data = get_proxy()
    if not proxy_data:
        print("No proxies available. Exiting program.")
        return True

    # Reprocess missing summaries
    futures = []
    with ThreadPoolExecutor(max_workers=100) as executor:
        proxy_index = 0
        for idx, row in df_missing_summaries.iterrows():
            while not proxy_data:
                proxy_data = get_proxy()
                print("Re-get proxy")
            if proxy_index >= len(proxy_data):
                proxy_index = 0
                proxy_data = get_proxy()
                while not proxy_data:
                    print("Failed to retrieve proxies. Exiting program.")
                    proxy_data = get_proxy()
            proxy = proxy_data[proxy_index]
            proxy_index += 1
            # Submit the task
            futures.append(executor.submit(reprocess_row, idx, row.copy(), headers, proxy))

        # Collect the results
        for future in as_completed(futures):
            try:
                idx, updated_row = future.result()
                df_output.loc[idx, summary_columns] = updated_row[summary_columns]
            except Exception as e:
                print(f"Error processing row {idx}: {e}")

    # Write the updated DataFrame back to the output file
    df_output.to_csv(output_file, sep='\t', index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
    print("Reprocessing complete. Updated file saved.")
    return True


def process_files_in_directory(directory_path):

    for filename in os.listdir(directory_path):
        if filename.endswith('.tsv'):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing file: {file_path}")

            while reprocess_missing_summaries(file_path):
                print(f"continue {file_path}")
                pass
            
if __name__ == "__main__":
    # output_file = 'test_null_re.tsv'
    # while reprocess_missing_summaries(output_file):
    #     print(f"continue {output_file}")
    #     pass
    output_files_directory = 'output_files'  
    process_files_in_directory(output_files_directory)