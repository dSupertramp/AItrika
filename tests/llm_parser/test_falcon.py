import unittest

from pubgpt.llm_parser.falcon import get_associations


class TestGetAssociations(unittest.TestCase):
    def test_get_associations(self):
        document = "This is a sample abstract."
        pubmed_id = "123456"
        pairs = [("GeneA", "DiseaseX"), ("GeneB", "DiseaseY")]

        result = get_associations(document, pubmed_id, pairs)

        self.assertEqual(result, "Yes,GeneA,DiseaseX\nYes,GeneB,DiseaseY")


if __name__ == "__main__":
    unittest.main()
