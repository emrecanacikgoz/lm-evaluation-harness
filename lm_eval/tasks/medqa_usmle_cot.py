"""
What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams
https://arxiv.org/abs/2009.13081

This contains the English portion of the full MedQA dataset, containing 12,723 multiple (4) choice questions from the US medical licensing exam.

Homepage: None
Credit to: https://github.com/TimD1 for the pending PR in the lm-evaluation harness repo

"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@misc{jin2020disease,
    title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams}, 
    author={Di Jin and Eileen Pan and Nassim Oufattole and Wei-Hung Weng and Hanyi Fang and Peter Szolovits},
    year={2020},
    eprint={2009.13081},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

"""
CoT Prompts are implemented from the "Gemini Goes to Med School: Exploring the Capabilities of Multimodal Large Language Models on Medical Challenge Problems & Hallucinations": https://arxiv.org/pdf/2402.07023.pdf
"""

class MedQA_USMLE_cot(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "augtoma/medqa_usmle"
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
            options = doc['choices']
            formatted_options = '\n'.join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])

            return f"The following is a multiple choice question about medical knowledge. Solve it in a step-by-step fashion, starting by summarizing the available information. Output a single option from the four options as the final answer.:\n{doc['query']}\nOptions:\n{formatted_options}\nAnswer:"