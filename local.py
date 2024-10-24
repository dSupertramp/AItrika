from aitrika.engine.local_aitrika import LocalAItrika


if __name__ == "__main__":
    engine = LocalAItrika(pdf_path="aitrika/sample_data/breast_cancer_brca1_brca2.pdf")

    abstract = engine.extract_abstract()

    print(abstract)
