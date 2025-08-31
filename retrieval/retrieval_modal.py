
import json
import time
import base64
from PIL import Image
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from prompt import prompt_evaluate
import copy
import warnings
warnings.filterwarnings("ignore")
custom_cache_dir = "/root/autodl-tmp/llava-ov-checkpoint"
pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov-chat"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map,cache_dir=custom_cache_dir)
model.eval()

from sentence_transformers import SentenceTransformer

class TextEncoder:
    def __init__(self, model_name="sentence-transformers/sentence-t5-base", device="cuda"):
        self.model = SentenceTransformer(model_name, device=device)

    def encode_query(self, query):
        return self.model.encode(query, convert_to_numpy=True)

def resize_image_if_large(image, max_width=768, max_height=768):
    """
    Resize the image if its width or height exceeds the specified max values.
    Keeps aspect ratio.
    """
    width, height = image.size
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        new_size = (int(width * scale), int(height * scale))

        # Use compatible resampling method
        if hasattr(Image, "Resampling"):  # Pillow >= 10
            return image.resize(new_size, resample=Image.Resampling.LANCZOS)
        else:
            return image.resize(new_size, resample=Image.ANTIALIAS)
    return image

class ImageEvaluator:
    def __init__(self):
        pass

    def _generate_response(self, prompt, images=None, image_sizes=None, max_new_tokens=2048, temperature=0.0):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        if DEFAULT_IMAGE_TOKEN in prompt:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            # use tokenizer_image_token to encode the prompt with image_token
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        text_outputs = tokenizer.batch_decode(output, skip_special_tokens=True)
        return text_outputs[0]

    def evaluate_single_image_one_step(self, query, description, image_paths, model_name="gpt-4o-mini-2024-07-18", retries=20, delay=3):
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        demo_prompt = prompt_evaluate.PromptTemplates_dual_eval.prompt_demo
        formatted_prompt_template = prompt_evaluate.PromptTemplates_dual_eval.prompt_dual_eval

        results = []

        for image_path in image_paths:
            attempt = 0
            while attempt < retries:
                try:
                    with open(image_path, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                    image = Image.open(image_path)
                    image_format = image.format.lower()
                    image = resize_image_if_large(image)

                    formatted_prompt = formatted_prompt_template.format(
                        query=query,
                        description=description
                    )

                    conv_template = "qwen_1_5"  
                    conv = copy.deepcopy(conv_templates[conv_template])

                    user_prompt = DEFAULT_IMAGE_TOKEN + "\n" + formatted_prompt

                    conv.append_message(conv.roles[0], demo_prompt)  # system
                    conv.append_message(conv.roles[1], user_prompt)  # user
                    final_prompt = conv.get_prompt()

                    # process image
                    processed_images = process_images([image], image_processor, model.config)
                    processed_images = [_image.to(dtype=torch.float16, device=device) for _image in processed_images]
                    image_sizes = [image.size]

                    base_model_response = self._generate_response(final_prompt, images=processed_images, image_sizes=image_sizes)
                    json_start = base_model_response.index("{")  
                    json_end = base_model_response.rindex("}")
                    valid_json = base_model_response[json_start:json_end + 1]
                    parsed_json = json.loads(valid_json)
                    textual_match_score = parsed_json.get("textual_match_score", None)
                    visual_match_score = parsed_json.get("visual_match_score", None)

                    results.append({
                        "image_path": image_path,
                        "response": parsed_json,
                        "textual_match_score": textual_match_score,
                        "visual_match_score": visual_match_score
                    })
                    break
                except Exception as e:
                    attempt += 1
                    print(f"Local model request failed for {image_path} (Attempt {attempt}/{retries}): {e}")
                    if attempt < retries:
                        time.sleep(delay)
                    else:
                        raise RuntimeError(f"Failed to get a response for {image_path} after {retries} attempts.") from e

        return results


    def extract_scores(self, json_data):
        data = json.loads(json_data)
        print(json_data)
        scores = [item["score"] for item in data["evaluation_results"]]
        return scores


    def evaluate_single_image_dual_eval(self, query, description, image_paths, model_name="gpt-4o-mini-2024-07-18", retries=20, delay=3):
        """
        Evaluate using the dual_eval prompt (textual and visual match) for each image.
        Args:
            query: The user query (text)
            description: The description to compare (text)
            image_paths: List of image paths or a single image path
        Returns:
            List of dicts with image_path, response (parsed JSON), textual_match_score, visual_match_score
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        demo_prompt = prompt_evaluate.PromptTemplates_dual_eval.prompt_demo
        formatted_prompt_template = prompt_evaluate.PromptTemplates_dual_eval.prompt_dual_eval

        results = []
        for image_path in image_paths:
            attempt = 0
            while attempt < retries:
                try:
                    with open(image_path, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                    image = Image.open(image_path)
                    image_format = image.format.lower()
                    image = resize_image_if_large(image)

                    formatted_prompt = formatted_prompt_template.format(
                        query=query,
                        description=description
                    )

                    conv_template = "qwen_1_5"
                    conv = copy.deepcopy(conv_templates[conv_template])

                    user_prompt = DEFAULT_IMAGE_TOKEN + "\n" + formatted_prompt

                    conv.append_message(conv.roles[0], demo_prompt)  # system
                    conv.append_message(conv.roles[1], user_prompt)  # user
                    final_prompt = conv.get_prompt()

                    processed_images = process_images([image], image_processor, model.config)
                    processed_images = [_image.to(dtype=torch.float16, device=device) for _image in processed_images]
                    image_sizes = [image.size]

                    base_model_response = self._generate_response(final_prompt, images=processed_images, image_sizes=image_sizes)
                    json_start = base_model_response.index("{")
                    json_end = base_model_response.rindex("}")
                    valid_json = base_model_response[json_start:json_end + 1]
                    parsed_json = json.loads(valid_json)
                    textual_match_score = parsed_json.get("textual_match_score", None)
                    visual_match_score = parsed_json.get("visual_match_score", None)

                    results.append({
                        "image_path": image_path,
                        "response": parsed_json,
                        "textual_match_score": textual_match_score,
                        "visual_match_score": visual_match_score
                    })
                    break
                except Exception as e:
                    attempt += 1
                    print(f"Dual eval model request failed for {image_path} (Attempt {attempt}/{retries}): {e}")
                    if attempt < retries:
                        time.sleep(delay)
                    else:
                        raise RuntimeError(f"Failed to get a response for {image_path} after {retries} attempts.") from e
        return results

    def evaluate_top1_overall_match(self, query, description, image_path, model_name="gpt-4o-mini-2024-07-18", retries=20, delay=3):
        """
        Evaluate the top-1 retrieval result using the overall match prompt.
        Args:
            query: The user query (text)
            description: The description to compare (text)
            image_path: The image path (single image)
        Returns:
            Dict with image_path, response (parsed JSON), is_match
        """
        demo_prompt = None  # No demo for overall match in the prompt file
        formatted_prompt_template = prompt_evaluate.PromptTemplates_dual_eval.prompt_overall_match
        attempt = 0
        while attempt < retries:
            try:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                image = Image.open(image_path)
                image_format = image.format.lower()
                image = resize_image_if_large(image)

                formatted_prompt = formatted_prompt_template.format(
                    query=query,
                    description=description
                )

                conv_template = "qwen_1_5"
                conv = copy.deepcopy(conv_templates[conv_template])

                user_prompt = DEFAULT_IMAGE_TOKEN + "\n" + formatted_prompt

                if demo_prompt:
                    conv.append_message(conv.roles[0], demo_prompt)  # system
                conv.append_message(conv.roles[1], user_prompt)  # user
                final_prompt = conv.get_prompt()

                processed_images = process_images([image], image_processor, model.config)
                processed_images = [_image.to(dtype=torch.float16, device=device) for _image in processed_images]
                image_sizes = [image.size]

                base_model_response = self._generate_response(final_prompt, images=processed_images, image_sizes=image_sizes)
                json_start = base_model_response.index("{")
                json_end = base_model_response.rindex("}")
                valid_json = base_model_response[json_start:json_end + 1]
                parsed_json = json.loads(valid_json)
                is_match = parsed_json.get("is_match", None)

                return {
                    "image_path": image_path,
                    "response": parsed_json,
                    "is_match": is_match
                }
            except Exception as e:
                attempt += 1
                print(f"Overall match model request failed for {image_path} (Attempt {attempt}/{retries}): {e}")
                if attempt < retries:
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Failed to get a response for {image_path} after {retries} attempts.") from e