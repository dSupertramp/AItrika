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


paper_results_prompt = """
Based on the text, you have to extract the results.

Before providing the final output, please think through the following steps:
1. Identify the relevant data that will be used to extract the results.
2. Ensure that the extracted results are formatted correctly as a string and well-detailed, with each result separated by three dashes (---).
3. Handle the case where no results are found, and return "No results" in the output.

Structure the output like this:
All results as string (--- as separator)

This is an example of output:
** PAPER RESULTS ** \n
"Result 1 --- Result 2 --- ... --- Result N"


Don't provide any kind of additional explaination.
If you don't find results, return "No results"

Once you have completed the thinking process, provide the results in the format specified above.
"""


effect_sizes_prompt = """
Based on the text, you have to extract the effect sizes and their description. 

Before providing the final output, please think through the following steps:
1. Identify the relevant data that will be used to extract the effect sizes.
2. For each element extracted, remember to add the explaination of the effect and its size.
3. Ensure that the extracted effect sizes are formatted correctly as a string and well-detailed, with each effect size separated by three dashes (---).
4. Handle the case where no effect sizes are found, and return "No effect sizes" in the output.

Structure the output like this:
All effect with sizes as string (--- as separator)

This is an example of output:
** EFFECT SIZES ** \n
"Effect 1: Size of effect 1 --- Effect 2: Size of effect 2 --- ... --- Effect N: Size of effect N"

Don't provide any kind of additional explaination.
If you don't find effect sizes, return "No effect sizes"

Once you have completed the thinking process, provide the effect with sizes in the format specified above.
"""


number_of_participants_prompt = """
Based on the text, you have to extract the number of participants.

Before providing the final result, please think through the following steps:
1. Carefully read the input text and identify any mentions of the number of participants.
2. Verify that the number of participants is a numerical value and not a textual description.
3. Ensure that the extracted number is the total number of participants, not a partial count or a range.

Structure the output like this:
Number of participants without separator

This is an example of output:
** NUMBER OF PARTICIPANTS ** \n
1500

Once you have completed the thinking process, provide the number of participants in the format specified above.
"""


characteristics_of_participants_prompt = """
Based on the text, you have to extract the characteristics of participants.

Before providing the final output, please think through the following steps:
1. Identify the relevant data that will be used to extract the characteristics.
2. Ensure that the extracted characteristics are formatted correctly as a string and well-detailed, with each characteristic separated by three dashes (---).
3. Handle the case where no characteristics are found, and return "No characteristics" in the output.

Structure the output like this:
All characteristics of the partecipants as string (--- as separator)

This is an example of output:
** CHARACTERISTICS OF PARTICIPANTS ** \n
"Characteristic 1 --- Characteristic 2 --- ... --- Characteristic N"


Don't provide any kind of additional explaination.
If you don't find characteristics, return "No characteristics"

Once you have completed the thinking process, provide the characteristics of participants in the format specified above.
"""


interventions_prompt = """
Based on the text, you have to extract the interventions. 

Before providing the final output, please think through the following steps:
1. Identify the relevant data that will be used to extract the interventions.
2. Ensure that the extracted interventions are formatted correctly as a string and well-detailed, with each intervention separated by three dashes (---).
3. Handle the case where no interventions are found, and return "No interventions" in the output.

Structure the output like this:
All interventions as string (--- as separator)

This is an example of output:
** INTERVENTIONS ** \n
"Intervention 1 --- Intervention 2 --- ... --- Intervention N"


Don't provide any kind of additional explaination.
If you don't find interventions, return "No interventions"

Once you have completed the thinking process, provide the interventions in the format specified above.
"""


outcomes_prompt = """
Based on the text, you have to extract the outcomes. 

Before providing the final output, please think through the following steps:
1. Identify the relevant data that will be used to extract the outcomes.
2. Ensure that the extracted outcomes are formatted correctly as a string and well-detailed, with each outcome separated by three dashes (---).
3. Handle the case where no outcomes are found, and return "No outcomes" in the output.

Structure the output like this:
All outcomes as string (--- as separator)

This is an example of output:
** OUTCOMES ** \n
"Outcome 1 --- Outcome 2 --- ... --- Outcome N"

Don't provide any kind of additional explaination.
If you don't find outcomes, return "No outcomes"

Once you have completed the thinking process, provide the outcomes in the format specified above.
"""
