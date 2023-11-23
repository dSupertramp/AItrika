import pathlib


def create_id_folder(pubmed_id: str) -> None:
    """
    Create a folder with Pubmed ID.

    Args:
        pubmed_id (str): Pubmed ID
    """
    pathlib.Path(f"output/{pubmed_id}").mkdir(parents=True, exist_ok=True)
