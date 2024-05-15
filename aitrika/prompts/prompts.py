"""Prompts"""

results_prompt = """
In the given text, extract the results and format it as a digest. 
Split each entry into main concepts using a dash (-) and each main concept in a new line.
Put each entry on a new line.

Structure the output like this:
** RESULTS ** \n
<Output>
</Output>
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

Structure the output like this:
** BIBLIOGRAPHY ** \n
<Output>
</Output>
"""


methods_prompt = """
In the given text, extract the methods and format it as a digest.
Split each entry into main concepts using a dash (-) and each main concept in a new line.
Put each entry on a new line.

Structure the output like this:
** METHODS ** \n
<Output>
</Output>
"""


introduction_prompt = """
In the given text, extract the introductin and format it as a digest.

Structure the output like this:
** INTRODUCTION ** \n
<Output>
</Output>
"""


acknowledgments_prompt = """
In the given text, extract the acknowledgements and format it as a digest.

Structure the output like this:
** ACKNOWLEDGEMENTS ** \n
<Output>
</Output>
"""
