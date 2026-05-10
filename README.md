# Reproducing Reasoning Emergence in Small LLMs via GRPO Post-Training

> A reproduction of DeepSeek-R1's reinforcement learning pipeline at sub-3B scale using open-source tools, public datasets, and a single GPU.

**🎥 Presentation Video:** [ Insert Video Link Here ]  
**🤗 Model Weights (Google Drive):** https://drive.google.com/drive/folders/117hnUye-XuguR5YWbVxcYPJzk4VaL8Dz?usp=sharing

---

## Overview

This project investigates whether the GRPO (Group Relative Policy Optimization) post-training paradigm from DeepSeek-R1 can be reproduced at small scale — under 3B parameters — using only free-tier hardware and open-source datasets.

**Pipeline:** Baseline → SFT Warm-up → GRPO Training → Evaluation  
**Model:** Qwen2.5-1.5B  
**Datasets:** NuminaMath-CoT (SFT) · GSM8K (GRPO + Eval)  
**Hardware:** Google Colab T4 GPU (15 GB VRAM) · RunPod (Config C)  
**Benchmark:** GSM8K Test Set — Exact Answer Match

### Results Summary

| Stage | GSM8K Accuracy |
|---|---|
| Baseline (zero-shot) | 8% |
| After SFT (5 epochs) | 28% |
| After GRPO — Config A (G=2, 128tok, LR=5e-6, KL=0.01) | 44% → 48% |
| After GRPO — Config B ★ (G=2, 256tok, LR=1e-6, KL=0.01) | 54% → **60%** → 44% |
| After GRPO — Config C (G=4, 256tok, LR=1e-6, KL=0.1) | 52% → 55% → 54% |

---

## Repository Structure

```
RL_Project/
│
├── 1_qwen2_5_sft_training.ipynb          # Stage 1: Supervised Fine-Tuning on NuminaMath-CoT
├── 2_grpo_train.ipynb         # Stage 2: GRPO RL Training on GSM8K
├── 3_grpo_test.ipynb  # Evaluate saved checkpoints on GSM8K test set
├── 4_reasoning_quality.ipynb  # Qualitative analysis of model outputs per stage
│
└── README.md                     # This file
```

> **Note:** Model weights and LoRA adapter checkpoints are too large for GitHub. All checkpoints are stored in Google Drive — see the link at the top of this README.

---

## File Descriptions

### `1__qwen2_5_sft_training.ipynb` — SFT Warm-up
Supervised fine-tuning of Qwen2.5-1.5B on NuminaMath-CoT chain-of-thought examples. This stage teaches the model the `<think>...</think>` output format and stabilises generation before RL training begins. Without this step, the base model produces incoherent outputs and GRPO receives near-zero reward signal with no gradient to learn from.

**Key parameters:**
| Parameter | Value | Reason |
|---|---|---|
| Dataset | NuminaMath-CoT, 2,000 samples | Harder than GSM8K → better generalisation |
| Epochs | 5 | Loss plateaus; more epochs overfit to NuminaMath style |
| Learning rate | 2e-4 | Standard LoRA fine-tuning rate for Qwen family |
| LoRA rank / alpha | 16 / 32 | ~3% trainable params; fits T4 VRAM |
| Batch / GradAcc | 2 / 8 (eff. 16) | Memory constraint with gradient quality |
| Max seq length | 512 tokens | Covers 95% of NuminaMath solutions without truncation |
| Output format | `<think>…</think> Final Answer: X` | Primes model for GRPO format reward |

---

### `2_grpo_train.ipynb` — GRPO Training
Implements Group Relative Policy Optimisation using TRL's `GRPOTrainer`. For each prompt, G completions are sampled, scored with a combined reward function, and the policy is updated to favour higher-advantage completions subject to a KL divergence penalty from the SFT checkpoint.

**Reward function:**
- Correctness reward: `+1.0` if predicted number = ground truth (exact match)
- Format reward: `+0.3` for step-by-step reasoning (length > 20 words, mathematical keywords, numbers present)
- Max reward per completion: `1.3`

**Three configurations were run (ablation):**
| Config | G | Max Gen Tokens | KL β | LR | Hardware |
|---|---|---|---|---|---|
| A | 2 | 128 | 0.01 | 5e-6 | Colab T4 |
| B ★ | 2 | 256 | 0.01 | 1e-6 | Colab T4 |
| C | 4 | 256 | 0.1 | 1e-6 | RunPod |

**Fixed parameters across all configs:**
| Parameter | Value | Reason |
|---|---|---|
| GRPO samples | 500 (GSM8K train) | G×completions×256 tok approaches T4 memory ceiling |
| Batch / GradAcc | 4 / 8 (eff. 32) | Stable gradients within memory budget |
| Max prompt length | 256 tokens | Covers all GSM8K problem statements |
| LR scheduler | Cosine + 5% warmup | Fixed to isolate LR magnitude in ablation |
| Epochs | 3 | Limits over-optimisation on small dataset |
| Precision | fp16 | Saves ~3 GB VRAM vs fp32 |

---

### `3_grpo_test.ipynb` — Checkpoint Evaluation
Loads saved LoRA checkpoints from each GRPO epoch and evaluates accuracy on a 50-sample probe of the GSM8K test set using exact answer match. Run this notebook to reproduce the per-epoch accuracy numbers reported.

---

### `4_reasoning_quality.ipynb` — Reasoning Quality Analysis
Qualitative comparison of model outputs across the three training stages (baseline, post-SFT, post-GRPO). Loads each checkpoint and generates completions for the same set of GSM8K problems, allowing side-by-side inspection of reasoning quality — structure, arithmetic correctness, constraint application, and use of `<think>` tags.

---

## How to Run

### Prerequisites
```bash
pip install transformers==4.46.3
pip install trl==0.15.1
pip install peft==0.13.2
pip install bitsandbytes==0.43.3
pip install datasets==3.1.0
pip install torch==2.3.0
```

Or simply open the notebooks in **Google Colab** — all dependencies are installed in the first cell of each notebook.

### Step-by-Step Execution

**Step 1 — SFT Warm-up**
```
Open 1__qwen2_5_sft_training.ipynb in Colab (T4 GPU)
Run all cells top to bottom
Checkpoint saved to: /content/sft_output/ (or upload to Drive)
```

**Step 2 — GRPO Training**
```
Open 2_grpo_trai.ipynb in Colab (T4 GPU) or RunPod
Set config parameters at the top of the notebook (G, max_gen_tokens, kl_beta, lr)
Load the SFT checkpoint path
Run all cells
Checkpoints saved per epoch
```

**Step 3 — Checkpoint Evaluation**
```
Open 3_grpo_test.ipynb
Point checkpoint_paths to your saved GRPO epoch directories
Run all cells — outputs per-epoch GSM8K accuracy
```

**Step 4 — Reasoning Quality Check**
```
Open 4_reasoning_quality.ipynb
Load baseline, SFT, and GRPO checkpoint paths
Run all cells — generates side-by-side output comparison
```

### Using Pretrained Weights
To skip training entirely and reproduce evaluation results, download the checkpoints from the Google Drive link at the top of this README and point the evaluation notebooks to the downloaded directories.

---

## Key Findings

1. **GRPO unlocks reasoning beyond SFT.** Every GRPO configuration surpassed the SFT ceiling of 28% within epoch 1. The 8→28→60% trajectory reflects a phase transition — SFT teaches format, GRPO teaches reasoning.
   
2. **The exploration-stability tradeoff is empirically measurable.** KL=0.01 with G=2 peaks at 60% but collapses at epoch 3 (reward hacking). KL=0.1 with G=4 is stable across all epochs at 52–55%. The right choice depends on whether peak performance or training stability is the priority.

---

## Authors

**Sai Akshitha Boddupalli · Keyaba Gohil**  
Course Project — Reinforcement Learning from Human Feedback  

---

## References

- DeepSeek-R1: [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)
- Qwen2.5: [huggingface.co/Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- NuminaMath-CoT: [huggingface.co/datasets/AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
- GSM8K: [huggingface.co/datasets/openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)
- TRL GRPOTrainer: [huggingface.co/docs/trl](https://huggingface.co/docs/trl)
