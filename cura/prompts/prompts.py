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
Answer all these questions:
{pairs}

The output should be a csv formatted as follow:
```
gene_id,gene_name,disease_id,disease_name,is_associated
123,x,321,y,True
999,m,020,p,False
```
Make sure that the result is a csv (comma-separated value) and also make sure that the header is in lower-case
"""
