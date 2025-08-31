from retrieval_modal import ImageEvaluator
from pathlib import Path
import pandas as pd
from retrieval_modal import TextEncoder
import faiss
import numpy as np
import pickle
from tqdm import tqdm  
import json
import math

def is_match(result, ground_truth):
    return (
        result.get('url') == ground_truth.get('image_url') and
        result.get('caption') == ground_truth.get('page_title')
    )

def find_matching_row_url(x, file_path = "merged_image_path.tsv"):
    try:
        df = pd.read_csv(file_path, sep='\t')

        matching_row = df[df['url'] == x].head(1)

        if not matching_row.empty:
            return matching_row[['absolute_image_path', 'width', 'height']].iloc[0].to_dict()
        else:
            return "No matching rows found."
    except FileNotFoundError:
        return "File not found. Please check the file path."
    except Exception as e:
        return f"An error occurred: {e}"

def find_matching_row_url_nan(x, file_path = "merged_image_path.tsv"):
    try:
        df = pd.read_csv(file_path, sep='\t')

        matching_row = df[df['caption'] == x].head(1)

        if not matching_row.empty:
            return matching_row[['absolute_image_path', 'width', 'height']].iloc[0].to_dict()
        else:
            return "No matching rows found."
    except FileNotFoundError:
        return "File not found. Please check the file path."
    except Exception as e:
        return f"An error occurred: {e}"

# def retrieve_neighbors(query_vector, k=15):
#     distances, indices = index.search(query_vector, k)
#     results = []
#     for idx, dist in zip(indices[0], distances[0]):
#         result = df.iloc[idx][non_feature_columns].to_dict()
#         result['distance'] = dist  
#         results.append(result)
#     return results

def find_matching_row_image_path(x, file_path = "merged_image_path.tsv"):
    try:
        df = pd.read_csv(file_path, sep='\t')

        matching_row = df[df['absolute_image_path'] == x].head(1)

        if not matching_row.empty:
            return matching_row[['url', 'caption']].iloc[0].to_dict()
        else:
            return "No matching rows found."
    except FileNotFoundError:
        return "File not found. Please check the file path."
    except Exception as e:
        return f"An error occurred: {e}"
    
if __name__ == "__main__":
    Encoder = TextEncoder()

    database_file = "data_test_change_summary_all_with_another_name_fixed_detailed_embeddings_query(encode_Summary_all)_openai_3_large.parquet"
    df = pd.read_parquet(database_file)

    feature_columns = [col for col in df.columns if col.startswith('dim_')]
    non_feature_columns = [col for col in df.columns if not col.startswith('dim_')]
    db_encodings = df[feature_columns].values.astype('float32') 

    print("database_loaded")

    dimension = db_encodings.shape[1]  
    cpu_index = faiss.IndexFlatL2(dimension) 
    gpu_res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index) 
    gpu_index.add(db_encodings)  
     
    print("Faiss index created, the total number of images in the database is: ", len(db_encodings))

    query_file = "wit_test_set.tsv"
    query_df = pd.read_csv(query_file, sep="\t")
    num_queries = len(query_df)
    correct_count = 0
    failure_count = 0 
    with tqdm(total=num_queries, desc="Processing Queries") as pbar:
        for _, row in query_df.iterrows():
            query = row['another_name_detailed']
            ground_truth = {
                'image_url': row['image_url'],
                'page_title': row['page_title']
            }

            # encode query
            query_vector = Encoder.encode_query(query)

            # make sure query_vector is numpy array
            if not isinstance(query_vector, np.ndarray):
                query_vector = np.array(query_vector, dtype='float32')
            if len(query_vector.shape) == 1:
                query_vector = query_vector[np.newaxis, :]

            # use Faiss to retrieve nearest neighbors
            distances, indices = gpu_index.search(query_vector, k=5)
            nearest_neighbors = [df.iloc[idx].to_dict() for idx in indices[0]]
            
            # Process image paths for all neighbors
            image_paths = []
            descriptions = []
            for neighbor in nearest_neighbors:
                image_url = neighbor.get('image_url')
                description = neighbor.get('summary_all')  # Get description from summary_all field
                
                
                if image_url is None or (isinstance(image_url, float) and np.isnan(image_url)):
                    image_caption = neighbor.get('caption_reference_description')
                    result = find_matching_row_url_nan(image_caption)
                else:
                    result = find_matching_row_url(image_url)
                    
                if isinstance(result, dict) and 'absolute_image_path' in result:
                    image_paths.append(result['absolute_image_path'])
                    descriptions.append(description)
                else:
                    print(f"Skipping one of the results {query} because no matching image was found.")
            
            if not image_paths:
                print(f"No valid image paths found for query: {query}")
                failure_count += 1
                continue
                
            # Initialize evaluator
            evaluator = ImageEvaluator()
            
            # First, check if top1 is a match using evaluate_top1_overall_match
            top1_result = evaluator.evaluate_top1_overall_match(query, descriptions[0], image_paths[0])
            top1_is_match = top1_result.get('is_match', False)
            
            if top1_is_match:
                # If top1 is a match, use it directly
                target_image_path = image_paths[0]
            else:
                # If top1 is not a match, evaluate all images using dual_eval
                evaluation_results = []
                for i, (image_path, description) in enumerate(zip(image_paths, descriptions)):
                    try:
                        # Use evaluate_single_image_dual_eval for each image
                        results = evaluator.evaluate_single_image_dual_eval(query, description, image_path)
                        
                        if not results:
                            continue
                            
                        for result in results:
                            # Skip None results
                            if result is None:
                                continue
                                
                            # Extract scores safely
                            textual_score = 0
                            visual_score = 0
                            
                            if 'textual_match_score' in result:
                                textual_score = result['textual_match_score'] or 0
                                
                            if 'visual_match_score' in result:
                                visual_score = result['visual_match_score'] or 0
                                
                            avg_score = (textual_score + visual_score) / 2
                            
                            evaluation_results.append({
                                'image_path': image_path,
                                'textual_match_score': textual_score,
                                'visual_match_score': visual_score,
                                'average_score': avg_score
                            })
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")
                
                # Find the image with the highest average score
                if evaluation_results:
                    highest_score = max(item['average_score'] for item in evaluation_results)
                    highest_score_group = next(item for item in evaluation_results if item['average_score'] == highest_score)
                    target_image_path = highest_score_group['image_path']
                else:
                    # Fallback to top1 if evaluation fails
                    target_image_path = image_paths[0]
            
            # Get the URL and caption for the selected image
            find_image = find_matching_row_image_path(target_image_path)

            # Check if the result matches the ground truth
            if is_match(find_image, ground_truth):
                correct_count += 1
            else:
                failure_count += 1
                with open("failed_examples_two_step.txt", "a") as file:
                    file.write(f"Failure #{failure_count}:\n")
                    file.write(f"Query: {query}\n")
                    file.write(f"Ground Truth: {ground_truth}\n")
                    file.write(f"Predicted: {find_image}\n")
                    file.write(f"Top1 Is Match: {top1_is_match}\n")
                    if not top1_is_match and 'evaluation_results' in locals():
                        file.write(f"Evaluation Results: {evaluation_results}\n")
                        json.dump(evaluation_results, file, ensure_ascii=False, indent=4)
                    file.write("\n" + "-" * 50 + "\n")  # separator, for reading
                print(f"saved failure #{failure_count}")
                
            # Calculate current accuracy and update progress bar description
            current_accuracy = (correct_count / (pbar.n + 1)) * 100  # pbar.n is the number of processed items
            pbar.set_description(f"Processing Queries (Accuracy: {current_accuracy:.2f}%)")
            pbar.update(1)  # update progress bar

    # Calculate final accuracy
    accuracy = correct_count / num_queries
    print(f"Final Accuracy: {accuracy * 100:.2f}%")