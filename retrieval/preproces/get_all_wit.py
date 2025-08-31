import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from modal import TextAnalyzer_summary_body, TextAnalyzer_summary_all  # import two analysis classes
import time
from datetime import datetime
from ratelimit import limits, sleep_and_retry

# initialize the analysis class
text_analyzer_body = TextAnalyzer_summary_body()
text_analyzer_all = TextAnalyzer_summary_all()

MAX_REQUESTS_PER_MINUTE = 200    #need/5*6

@sleep_and_retry
@limits(calls=MAX_REQUESTS_PER_MINUTE, period=60)
def limited_rate_request(content, page_title):
    return text_analyzer_body.get_model_result_with_api(content, page_title)
    
def analyze_contents_in_parallel(summaries_list, page_title):
    """
    parallel analysis each summary, pass to the get_model_result_with_api method of TextAnalyzer_summary_body.
    :param summaries_list: summary list
    :param page_title: the page_title of the current row
    :return: the analysis result list of each summary
    """
    with ThreadPoolExecutor(max_workers = 5) as executor:
        futures = [
            executor.submit(limited_rate_request, content, page_title)
            for content in summaries_list if content
        ]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing content with page_title '{page_title}': {e}")
                results.append(None)  # you can use None to represent the wrong item
    return results


def analyze_summaries(row):
    """
    extract the summary column of the row, and call analyze_contents_in_parallel to process it.
    then call TextAnalyzer_summary_all to get the summary result.
    :param row: a row of DataFrame
    :return: the result of analyze_contents_in_parallel and the final summary string
    """
    # extract the summary column content and filter the empty item
    summaries = [row[f'summary{i + 1}'] for i in range(5)]
    filtered_summaries = [summary for summary in summaries if pd.notnull(summary)]

    # get the page_title and call the parallel analysis function
    page_title = row['page_title']
    analyzed_results = analyze_contents_in_parallel(filtered_summaries, page_title)
    # if all(result is None for result in analyzed_results):
    #     # if all the results are None, return the empty string to represent the error
    #     return analyzed_results, ""
    if any(result is None for result in analyzed_results):
    # if there is any result is None, return the empty string to represent the error
        return analyzed_results, ""
    # call
    summary_all_result = text_analyzer_all.get_model_result_with_api(page_title, analyzed_results)
    
    return analyzed_results, summary_all_result


def process_and_analyze_file(file_path):
    """
    read the file line by line, and use the parallel thread pool to analyze the summary content of each line, and add the Summary_all column.
    :param file_path: the file path
    :return: the DataFrame with the analysis result
    """
    data = pd.read_csv(file_path, sep='\t', encoding='utf-8')

    # create a list to store the Summary_all column result
    summary_all_results = []

    # parallel analysis each line
    # with ThreadPoolExecutor(max_workers = 100) as executor:
    with ThreadPoolExecutor() as executor:    
        futures = {executor.submit(analyze_summaries, row): index for index, row in data.iterrows()}
        for future in futures:
            index = futures[future]
            try:
                _, summary_all_result = future.result()
                summary_all_results.append(summary_all_result)
            except Exception as e:
                print(f"Error analyzing row {index} in file {file_path}: {e}")
                summary_all_results.append("")  # if there is an error, add the empty string as a placeholder

    # add the Summary_all column result to the DataFrame
    data['Summary_all'] = summary_all_results

    # save the new DataFrame to the file
    output_file_path = file_path.replace("_summary.tsv", "_final_summary.tsv")
    data.to_csv(output_file_path, sep='\t', index=False, encoding='utf-8')

    print(f"File saved with Summary_all column: {output_file_path}")


def process_files_sequentially(output_directory):
    """
    process each file sequentially, and generate a new file with the Summary_all column.
    :param output_directory: the directory to store the file
    """
    processed_files = [os.path.join(output_directory, f) for f in os.listdir(output_directory) if
                       f.endswith("_summary.tsv")]

    # process each file sequentially
    for file in processed_files:
        print(f"Processing file: {file}")
        start_time = time.time()
        process_and_analyze_file(file)
        end_time = time.time() 
        processing_time = end_time - start_time 
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # get the current timestamp
        print(f"[{current_time}] Finished processing {file} in {processing_time:.2f} seconds.\n")

def process_file(file_path):
    """
    process the specified file, and generate a new file with the Summary_all column.
    :param file_path: the
    """
    if not os.path.isfile(file_path):
        print(f"{file_path} is not a valid file path.")
        return

    print(f"Processing file: {file_path}")
    process_and_analyze_file(file_path)


# # use example
output_directory = "wit_dataset"
process_files_sequentially(output_directory)

# # use example
# start_time = time.time()
# # file_path = "food2k_with_summary_raw.tsv"
# file_path = "file_part_2_summary.tsv"
# process_file(file_path)
# end_time = time.time()
# print("program end.")
# print(f"total running time: {end_time - start_time:.2f} seconds")