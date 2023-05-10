import pathlib


def create_id_folder(document_id: str) -> None:
    pathlib.Path(f"output/{document_id}").mkdir(parents=True, exist_ok=True)
