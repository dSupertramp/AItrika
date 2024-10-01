def load_prompt(filename: str) -> str:
    """
    Load the prompt from .tmpl files.

    Args:
        filename (str): Prompt from .tmpl file

    Returns:
        str: Prompt
    """
    with open(filename, "r") as file:
        return file.read()
