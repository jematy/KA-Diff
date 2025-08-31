from openai import OpenAI

def get_model_result_with_api(user_input="", object_name="", model_name="gpt-4o-mini"):
    # openai.api_key = ""
    client = OpenAI(
        # change the api
        api_key = "",
        # change the base_url
        base_url = ""

    )

    prompt = f"The object is {object_name}. And here is the text: {user_input}."

    messages = [
        {"role": "system", "content": "You are a helpful assistant. I have a text passage below, and I need you to determine if it contains any detailed descriptions of the appearance and shape characteristics of an object. The object could be anything, such as a gun, a dog, a painting, a mountain, etc. If the text does include this information, please extract and share the specific details about the object's appearance and shape with me. If the text does not contain such information, please inform me that no relevant information is present. Here's the text: [Insert text here]."},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    base_model_response = response.choices[0].message.content.strip()
    # print("result: ", base_model_response)
    return base_model_response

if __name__ == '__main__':
    model_name = "gpt-4"
    prompt = "A teddy bear is a soft and cuddly toy bear made of plush or fabric, typically filled with stuffing and designed to resemble a real bear. Teddy bears are often used as a comfort object or a symbol of affection, and are commonly given as gifts to children or loved ones. They can vary in size, shape, and color, but are usually depicted with a cute and endearing appearance. The name teddy bear comes from the fact that they were originally inspired by a real bear cub that was caught by President Theodore Teddy Roosevelt during a hunting trip in 1902."
    get_model_result_with_api(prompt, "teddy bear")