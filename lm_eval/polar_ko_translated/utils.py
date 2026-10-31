import re
import datasets
from datasets import Dataset
import json


def process_docs(dataset) -> datasets.Dataset:
    #dataset = Dataset.from_list(dataset)

    def _process_doc(doc):

        prompt = f"{doc['context']}"

        # Choices 구성
        choices = [
            doc["option1"],
            doc["option2"],
            doc["option3"],
        ]

        return {
            "idx": doc["idx"],
            "axis": doc["axis"],
            "category": doc["category"],
            "option1": doc["option1"],
            "option2": doc["option2"],
            "option3": doc["option3"],
            "query": prompt,
            "choices": choices,
            "gold": 0  # always option1
        }

    return dataset.map(_process_doc, load_from_cache_file=False, keep_in_memory=True)
