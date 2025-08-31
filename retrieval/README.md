### 📥 Dataset Download and Web Search

This section explains how to construct the retrieval database via large-scale image downloading, semantic web augmentation, initial retrieval and and multimodal re-ranking.

---

#### 🖼️ Image Downloading

We use [**img2dataset**](https://github.com/rom1504/img2dataset) for efficient, multithreaded large-scale image downloading.

- A reference script is provided: [`download.py`](./download.py)
- Please refer to the [official img2dataset documentation](https://github.com/rom1504/img2dataset) for installation and usage.

> ⚠️ **Note**: Since the download process is multithreaded, some images may fail to download due to broken links or network errors. The final dataset size may slightly vary from the original list.

After downloading, use the script [`preprocess/get_wit_image_path.py`](./preprocess/get_wit_image_path.py) to map each downloaded image to its original entry. This ensures accurate linkage between image paths and their corresponding metadata.

---

#### 🧠 Web Search for Semantic Augmentation

To enrich rare or ambiguous concepts with additional textual context, we use the [**DuckDuckGo Search** (`ddgs`)](https://github.com/deedy5/ddgs) Python package.

- Our semantic expansion pipeline is implemented in the [`preprocess/`](./preprocess) folder.
- It automatically extracts aliases, attributes, and contextual facts using web search and LLM summarization.

> ⚠️ **Note**: DuckDuckGo may rate-limit high-frequency requests. We recommend using a **rotating proxy** or queue system. Please refer to the [ddgs issues page](https://github.com/deedy5/ddgs/issues) for details.

---

#### 📄 Preprocessed Semantic Augmentation TSV

We also provide a preprocessed `.tsv` file containing results from web-enhanced semantic expansion.

- You can directly use this file to **skip the web search and LLM processing step**.
- Update [`download.py`](./download.py) to parse this file for parallel image downloading.

We have uploaded the relevant files to Google Drive:  
- [Database (retrieval corpus)](https://drive.google.com/file/d/1-WCeUixccpM9PrQ-XrLJ4wZ5efzfRVxN/view?usp=drive_link)  
- [Test / Query Pool (evaluation set for retrieval)](https://drive.google.com/file/d/1x_zIBO4Lf_aFt2mHAROD1smdQRm1zCLS/view?usp=drive_link)  
---

#### 🔍 Multimodal Retrieval and Re-ranking

To construct the retrieval database and perform inference:

- Use [`build_T5.py`](./build_T5.py) to encode the corpus and construct the FAISS index using a **T5-based text encoder**.
- Use [`retrieve_image.py`](./retrieve_image.py) to perform retrieval from text prompts.

For multimodal re-ranking, we adopt the [**llava-onevision-qwen2-7b-ov-chat**](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov) vision-language model.

> ⚠️ This reranker requires a full installation of the [LLaVA-NeXT environment](https://github.com/LLaVA-VL/LLaVA-NeXT?tab=readme-ov-file). Please follow the official setup instructions.

---
