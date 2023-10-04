from lm_eval.base import MultipleChoiceTask

class medmcqa(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "augtoma/medmcqa"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return map(self._process_doc, self.dataset["train"])

    def validation_docs(self):
        return []

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        return {
            "query": doc["question"] + "\n" + \
                    "".join([f" ({k}) {v}" if i else f"({k}) {v}" \
                    for i, (k, v) in enumerate(doc["options"].items())]),
            "choices": list(doc["options"].values()),
            "gold": ord(doc["answer_idx"])-ord("A"),
        }

    def doc_to_text(self, doc):
        return f"Question: {doc['query']}\nAnswer:"