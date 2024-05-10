"""Prompts"""

results_prompt = """
In the given text, extract the results and format it as a digest. 
Split each entry into main concepts using a dash (-). 
Put each entry on a new line.
"""


bibliography_prompt = """
In the given text, identify the references or bibliography section and extract the following information for each entry:

- Author(s)
- Title
- Publication date
- Journal/Book title
- Volume and issue numbers (if applicable)
- Page numbers (if applicable)
- DOI or URL (if applicable)
"""
