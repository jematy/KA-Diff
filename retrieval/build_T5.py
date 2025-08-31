import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

# load sentence-transformers model
model_name = "sentence-transformers/sentence-t5-base"  # replace with your model name
model = SentenceTransformer(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')

file_path = "wit_test_set.tsv"  # replace with your file path
print(f"read file: {file_path}")
data = pd.read_csv(file_path, sep='\t')

# make sure all required columns exist
required_columns = [
    "Summary_all", 
    "page_title", 
    "image_url", 
    "caption_reference_description",
    "another_name",
    "another_name_detailed"
]
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"missing required columns: {', '.join(missing_columns)}")

# handle missing values (if any)
data[required_columns] = data[required_columns].fillna("")

# extract columns data
output_image_urls = data["image_url"].tolist()
output_page_titles = data["page_title"].tolist()
output_caption_reference_descriptions = data["caption_reference_description"].tolist()
output_another_names = data["another_name"].tolist()
output_another_names_detailed = data["another_name_detailed"].tolist()
output_summaries = data["Summary_all"].tolist()

# encode Summary_all
print("encoding Summary_all...")
batch_size = 32  # adjust batch size according to memory
embeddings = []
for i in tqdm(range(0, len(output_summaries), batch_size), desc="encoding progress"):
    batch = output_summaries[i:i+batch_size]
    batch_embeddings = model.encode(batch, show_progress_bar=False)
    embeddings.extend(batch_embeddings)

# create embeddings DataFrame
print("creating embeddings DataFrame...")
embedding_dim = len(embeddings[0])
embedding_columns = [f"dim_{i}" for i in range(embedding_dim)]
embeddings_df = pd.DataFrame(embeddings, columns=embedding_columns)

# add other columns
print("adding other columns to DataFrame...")
embeddings_df["image_url"] = output_image_urls
embeddings_df["page_title"] = output_page_titles
embeddings_df["caption_reference_description"] = output_caption_reference_descriptions
embeddings_df["another_name"] = output_another_names
embeddings_df["another_name_detailed"] = output_another_names_detailed
embeddings_df["Summary_all"] = output_summaries

# save to Parquet file
output_file = "wit_test_embeddings(encode_another)_T5.parquet"  # replace with your file name
print(f"saving to {output_file}...")
embeddings_df.to_parquet(output_file, engine="pyarrow", index=False)

print("done!")