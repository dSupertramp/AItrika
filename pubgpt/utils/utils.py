import pathlib


def create_id_folder(pubmed_id: str) -> None:
    pathlib.Path(f"output/{pubmed_id}").mkdir(parents=True, exist_ok=True)
