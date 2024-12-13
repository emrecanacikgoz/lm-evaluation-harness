
# Medical Language Model Evaluation Harness

This project is a fork of the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), specifically adapted for evaluating medical language models used in the paper. It provides a unified framework for testing generative language models on various medical and general-purpose evaluation tasks.

---

## Features

- Comprehensive support for evaluating medical language models.
- Benchmarks for six widely recognized medical tasks:
  - **MedMCQA**
  - **MedQA-USMLE**
  - **PubMedQA**
  - **USMLE Step 1**
  - **USMLE Step 2**
  - **USMLE Step 3**
- Easy-to-use CLI for evaluation and few-shot learning setups.

---

## Installation

To install the `lm-eval` framework from this repository, follow these steps:

```bash
git clone https://github.com/emrecanacikgoz/lm-evaluation-harness/
cd lm-evaluation-harness
pip install -e .
```

---

## Benchmarking Medical Models

### 1. Evaluate a Model

To evaluate a model on the supported medical tasks, run the following command:

```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=emrecanacikgoz/hippollama \
    --tasks medmcqa,medqa_usmle,pubmedqa,usmle_step1,usmle_step2,usmle_step3 \
    --device cuda:0
```

Replace `emrecanacikgoz/hippollama` with your model's Hugging Face path if different.

### 2. Few-Shot Evaluation

For few-shot evaluation, specify the number of examples (e.g., 5) as follows:

```bash
python write_out.py \
    --model hf-causal \
    --model_args pretrained=emrecanacikgoz/hippollama \
    --tasks medmcqa,medqa_usmle,pubmedqa,usmle_step1,usmle_step2,usmle_step3 \
    --num_fewshot 5 \
    --output_base_path /path/to/output/folder
```

This will generate one text file per task in the specified output folder.

---

## Acknowledgements

This repository is adapted from the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). We extend our gratitude to the original authors for their contributions to open-source AI research.

---

## Contributing

We welcome contributions to improve and expand the functionality of this evaluation harness. Please open an issue or submit a pull request if you have suggestions or enhancements.

---

## License

This project follows the same licensing terms as the original [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). Please refer to the original repository for details.
