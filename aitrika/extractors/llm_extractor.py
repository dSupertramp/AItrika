from aitrika.llm.base_llm import BaseLLM
from aitrika.prompts.prompts import (
    results_prompt,
    bibliography_prompt,
    methods_prompt,
    introduction_prompt,
    acknowledgments_prompt,
    paper_results_prompt,
    effect_sizes_prompt,
    number_of_participants_prompt,
    characteristics_of_participants_prompt,
    interventions_prompt,
    outcomes_prompt,
)


class LLMExtractor:
    def extract_results(self, llm: BaseLLM) -> str:
        return llm.query(query=results_prompt)

    def extract_bibliography(self, llm: BaseLLM) -> str:
        return llm.query(query=bibliography_prompt)

    def extract_methods(self, llm: BaseLLM) -> str:
        return llm.query(query=methods_prompt)

    def extract_introduction(self, llm: BaseLLM) -> str:
        return llm.query(query=introduction_prompt)

    def extract_acknowledgements(self, llm: BaseLLM) -> str:
        return llm.query(query=acknowledgments_prompt)

    def extract_paper_results(self, llm: BaseLLM) -> str:
        return llm.query(query=paper_results_prompt)

    def extract_effect_sizes(self, llm: BaseLLM) -> str:
        return llm.query(query=effect_sizes_prompt)

    def extract_number_of_participants(self, llm: BaseLLM) -> str:
        return llm.query(query=number_of_participants_prompt)

    def extract_characteristics_of_participants(self, llm: BaseLLM) -> str:
        return llm.query(query=characteristics_of_participants_prompt)

    def extract_interventions(self, llm: BaseLLM) -> str:
        return llm.query(query=interventions_prompt)

    def extract_outcomes(self, llm: BaseLLM) -> str:
        return llm.query(query=outcomes_prompt)
