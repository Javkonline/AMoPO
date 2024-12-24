# AMoPO: Adaptive Multi-objective Preference Optimization without Reward Models and Reference Models

We provide a simplified version of our **AMoPO**.

## Dataset

### Training dataset

[help_steer2_train_format_po2.json](./help_steer2_train_format_po2.json) is our training dataset, which is an extension of Helpsteer2 including **instruction following** score.

### Evaluation dataset

[AlpacaEval 2](https://github.com/tatsu-lab/alpaca_eval), [Arena-Hard](https://github.com/lmarena/arena-hard-auto/blob/main/data/arena-hard-v0.1/question.jsonl), [MT-bench](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/mt_bench/question.jsonl) are popular benchmarks.

## Key trainer

Our AMoPO trainer is in [trainer.py](./src/train/amopo/trainer.py).

