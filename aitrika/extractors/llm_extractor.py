from aitrika.llm.base_llm import BaseLLM
from aitrika.utils.prompt_loader import load_prompt


class LLMExtractor:
    results_prompt = load_prompt("aitrika/prompts/results.tmpl")
    bibliography_prompt = load_prompt("aitrika/prompts/bibliography.tmpl")
    methods_prompt = load_prompt("aitrika/prompts/methods.tmpl")
    introduction_prompt = load_prompt("aitrika/prompts/introduction.tmpl")
    acknowledgments_prompt = load_prompt("aitrika/prompts/acknowledgments.tmpl")
    paper_results_prompt = load_prompt("aitrika/prompts/paper_results.tmpl")
    effect_sizes_prompt = load_prompt("aitrika/prompts/effect_sizes.tmpl")
    number_of_participants_prompt = load_prompt(
        "aitrika/prompts/number_of_participants.tmpl"
    )
    characteristics_of_participants_prompt = load_prompt(
        "aitrika/prompts/characteristics_of_participants.tmpl"
    )
    interventions_prompt = load_prompt("aitrika/prompts/interventions.tmpl")
    outcomes_prompt = load_prompt("aitrika/prompts/outcomes.tmpl")

    def extract_results(self, llm: BaseLLM) -> str:
        return llm.query(query=self.results_prompt)

    def extract_bibliography(self, llm: BaseLLM) -> str:
        return llm.query(query=self.bibliography_prompt)

    def extract_methods(self, llm: BaseLLM) -> str:
        return llm.query(query=self.methods_prompt)

    def extract_introduction(self, llm: BaseLLM) -> str:
        return llm.query(query=self.introduction_prompt)

    def extract_acknowledgements(self, llm: BaseLLM) -> str:
        return llm.query(query=self.acknowledgments_prompt)

    def extract_paper_results(self, llm: BaseLLM) -> str:
        return llm.query(query=self.paper_results_prompt)

    def extract_effect_sizes(self, llm: BaseLLM) -> str:
        return llm.query(query=self.effect_sizes_prompt)

    def extract_number_of_participants(self, llm: BaseLLM) -> str:
        return llm.query(query=self.number_of_participants_prompt)

    def extract_characteristics_of_participants(self, llm: BaseLLM) -> str:
        return llm.query(query=self.characteristics_of_participants_prompt)

    def extract_interventions(self, llm: BaseLLM) -> str:
        return llm.query(query=self.interventions_prompt)

    def extract_outcomes(self, llm: BaseLLM) -> str:
        return llm.query(query=self.outcomes_prompt)
