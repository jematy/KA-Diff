import csv

classes = {
    "backpack": "backpack",
    "backpack_dog": "backpack",
    "bear_plushie": "stuffed animal",
    "berry_bowl": "bowl",
    "can": "can",
    "candle": "candle",
    "cat": "cat",
    "cat2": "cat",
    "clock": "clock",
    "colorful_sneaker": "sneaker",
    "dog": "dog",
    "dog2": "dog",
    "dog3": "dog",
    "dog5": "dog",
    "dog6": "dog",
    "dog7": "dog",
    "dog8": "dog",
    "duck_toy": "toy",
    "fancy_boot": "boot",
    "grey_sloth_plushie": "stuffed animal",
    "monster_toy": "toy",
    "pink_sunglasses": "glasses",
    "poop_emoji": "toy",
    "rc_car": "toy",
    "red_cartoon": "cartoon",
    "robot_toy": "toy",
    "shiny_sneaker": "sneaker",
    "teapot": "teapot",
    "vase": "vase",
    "wolf_plushie": "stuffed animal"
}

live_subjects = {"cat", "cat2", "dog", "dog2", "dog3", "dog5", "dog6", "dog7", "dog8"}

object_prompts = [
    'a {0} in the jungle',
    'a {0} in the snow',
    'a {0} on the beach',
    'a {0} on a cobblestone street',
    'a {0} on top of pink fabric',
    'a {0} on top of a wooden floor',
    'a {0} with a city in the background',
    'a {0} with a mountain in the background',
    'a {0} with a blue house in the background',
    'a {0} on top of a purple rug in a forest',
    'a {0} with a wheat field in the background',
    'a {0} with a tree and autumn leaves in the background',
    'a {0} with the Eiffel Tower in the background',
    'a {0} floating on top of water',
    'a {0} floating in an ocean of milk',
    'a {0} on top of green grass with sunflowers around it',
    'a {0} on top of a mirror',
    'a {0} on top of the sidewalk in a crowded street',
    'a {0} on top of a dirt road',
    'a {0} on top of a white rug',
    'a red {0}',
    'a purple {0}',
    'a shiny {0}',
    'a wet {0}',
    'a cube shaped {0}'
]

live_subject_prompts = [
    'a {0} in the jungle',
    'a {0} in the snow',
    'a {0} on the beach',
    'a {0} on a cobblestone street',
    'a {0} on top of pink fabric',
    'a {0} on top of a wooden floor',
    'a {0} with a city in the background',
    'a {0} with a mountain in the background',
    'a {0} with a blue house in the background',
    'a {0} on top of a purple rug in a forest',
    'a {0} wearing a red hat',
    'a {0} wearing a santa hat',
    'a {0} wearing a rainbow scarf',
    'a {0} wearing a black top hat and a monocle',
    'a {0} in a chef outfit',
    'a {0} in a firefighter outfit',
    'a {0} in a police outfit',
    'a {0} wearing pink glasses',
    'a {0} wearing a yellow shirt',
    'a {0} in a purple wizard outfit',
    'a red {0}',
    'a purple {0}',
    'a shiny {0}',
    'a wet {0}',
    'a cube shaped {0}'
]

with open("dreambooth_prompts.tsv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(["name", "subject", "prompt"])

    for name, subject in classes.items():
        is_live = name in live_subjects
        prompts = live_subject_prompts if is_live else object_prompts
        for prompt_template in prompts:
            prompt = prompt_template.format(subject)
            writer.writerow([name, subject, prompt])
