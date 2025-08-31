class PromptTemplates_dual_eval:
    prompt_demo = '''You are an evaluator tasked with assessing the semantic consistency among three inputs: (1) a user QUERY, (2) a textual DESCRIPTION supplied by another system, and (3) the content of an IMAGE.  Your evaluation must be based on common sense, general knowledge, and the specific details provided in the description.  Your responses must follow the specific format requested in each step to ensure clarity and usability.'''

    prompt_dual_eval = '''You will receive three inputs:
• Query – free‑form text describing an object, person, action, or detailed scene.
• Description – another text that claims to refer to the same thing as the Query.
• Image – a picture that may or may not depict what the Query describes.

------------------------------------------------
YOUR TASKS
------------------------------------------------
STEP 1 – Query vs Description (Textual Match)
1.     Decide whether the Query and Description refer to the same entity / action / scene.
2.     Rely on commonsense, cultural, scientific, geographical, and historical knowledge.
3.     Assign textual_match_score from 0 to 100:
· 0‑30 = little or no overlap
· 31‑60 = partial / ambiguous overlap
· 61‑100 = high or complete overlap
4.    Provide short, high‑level explanations for each score, mentioning the most important matching and mismatching evidence.
STEP 2 – Query vs Image (Visual Match)
1.     Inspect the Image and judge whether it depicts what the Query refers to.
2.     Use the same knowledge sources and visual reasoning.
3.     Assign visual_match_score from 0 to 100 using the same rubric above.
4.    Provide short, high‑level explanations for each score, mentioning the most important matching and mismatching evidence.
------------------------------------------------
OUTPUT FORMAT
Return only the JSON object below:
{{
"textual_match_score": <integer 0‑100>,
"visual_match_score": <integer 0‑100>,
"reasoning": {{
"textual": "<why you chose the textual score>",
"visual": "<why you chose the visual score>"
}},
}}
------------------------------------------------
SCORING EXAMPLES
Example 1 – perfect textual & visual match
Query: "Shiba Inu, also known as the brushwood dog, is small and agile, with a compact frame, erect ears, and a curled tail. Its dense coat comes in red, sesame, or black and tan, giving it a fox‑like charm."  
Description: "The brushwood dog, or Shiba Inu, is known for its upright ears, curled tail, and alert expression. Its sturdy build and thick, reddish coat give it both toughness and elegance in a small package."  
Image: (photo clearly shows a reddish Shiba Inu with erect ears and a curled tail sitting on a wooden chair)  
Expected output (illustrative):
{{
  "textual_match_score": 100,
  "visual_match_score": 97,
  "reasoning": {{
    "textual": "Both texts describe the same breed and list identical key traits: erect ears, curled tail, small agile build, dense reddish coat.",
    "visual": "Image shows a reddish Shiba Inu with erect ears and a curled tail, matching all described traits except the chair detail not mentioned in Query."
  }},
}}
'''

    prompt_overall_match = '''You will receive three inputs:

• Query – free‑form text describing an object, person, action, or detailed scene.  
• Description – another text that claims to refer to the same thing as the Query.  
• Image – a picture that may or may not depict what the Query describes.

------------------------------------------------
YOUR TASK
------------------------------------------------
Determine whether the combination of Description and Image faithfully represents what the Query refers to.  
• Return true only if both the Description and the Image align with the Query on all salient points (entity, action, key attributes, context).  
• Otherwise return false.

Use commonsense, cultural, scientific, geographical, and historical knowledge as needed.

------------------------------------------------
OUTPUT FORMAT
------------------------------------------------
Return only the following JSON object:

{{
  "is_match": <true | false>,
  "reason": "<brief explanation citing the most important matching and mismatching evidence>",
}}  

------------------------------------------------
ILLUSTRATIVE EXAMPLES
Example 1 – match  
Query: "Shiba Inu, also known as the brushwood dog, is small and agile, with a compact frame, erect ears, and a curled tail."  
Description: "The brushwood dog, or Shiba Inu, is known for its upright ears, curled tail, and alert expression."  
Image: (photo clearly shows a Shiba Inu with erect ears and a curled tail sitting on a wooden chair)  
Expected output:
{{
  "is_match": true,
  "reason": "Both text and image depict a Shiba Inu with the specified ears, tail, size, and build.",
}}

Example 2 – no match  
Query: "A yellow school bus parked in front of a brick school building."  
Description: "A red sports car speeding down a highway."  
Image: (photo shows a red sports car on a highway)  
Expected output:
{{
  "is_match": false,
  "reason": "Description and image show a red sports car, which does not match the school‑bus‑in‑front‑of‑school scene in the Query.",
}}
'''