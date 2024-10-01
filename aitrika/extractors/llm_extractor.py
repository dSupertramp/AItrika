from aitrika.llm.base_llm import BaseLLM
from aitrika.utils.prompt_loader import load_prompt


class LLMExtractor:
    def __init__(self):
        pass

    def _extract(self, llm: BaseLLM, template_path: str) -> str:
        prompt = load_prompt(template_path)
        return llm.query(query=prompt)

    def extract_results(self, llm: BaseLLM) -> str:
        return self._extract(llm, "aitrika/prompts/results.tmpl")

    def extract_bibliography(self, llm: BaseLLM) -> str:
        return self._extract(llm, "aitrika/prompts/bibliography.tmpl")

    def extract_methods(self, llm: BaseLLM) -> str:
        return self._extract(llm, "aitrika/prompts/methods.tmpl")

    def extract_introduction(self, llm: BaseLLM) -> str:
        return self._extract(llm, "aitrika/prompts/introduction.tmpl")

    def extract_acknowledgements(self, llm: BaseLLM) -> str:
        return self._extract(llm, "aitrika/prompts/acknowledgments.tmpl")

    def extract_paper_results(self, llm: BaseLLM) -> str:
        return self._extract(llm, "aitrika/prompts/paper_results.tmpl")

    def extract_effect_sizes(self, llm: BaseLLM) -> str:
        return self._extract(llm, "aitrika/prompts/effect_sizes.tmpl")

    def extract_number_of_participants(self, llm: BaseLLM) -> str:
        return self._extract(llm, "aitrika/prompts/number_of_participants.tmpl")

    def extract_characteristics_of_participants(self, llm: BaseLLM) -> str:
        return self._extract(
            llm, "aitrika/prompts/characteristics_of_participants.tmpl"
        )

    def extract_interventions(self, llm: BaseLLM) -> str:
        return self._extract(llm, "aitrika/prompts/interventions.tmpl")

    def extract_outcomes(self, llm: BaseLLM) -> str:
        return self._extract(llm, "aitrika/prompts/outcomes.tmpl")
