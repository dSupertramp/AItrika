"""Prompts"""


pre_prompt = """
Act like an experienced doctor with over 30 years of experience. 
You have to answer this following topic:

{query} 

Before answering, take a deep breath and make and be sure to read the document carefully: there must be absolutely no errors in the response. 
Give me the answer in a conversational manner, with a serious tone.
Don't tell me that you're an experience doctor and that it's always best to consult with a medical professional: just provide me the answer.
"""


associations_prompt = """
According to the text provided, can you tell me if:
{pairs}
As result, provide me a CSV with:
- Boolean result (only 'Yes' or 'No')
- The entire part before the sentence "is associated with"
- The entire part after the sentence "is associated with"
For instance:
'Yes,X,Y'
Also, remove the numbers list (like 1)) from the CSV.
"""
