# Empathetic Call Centre AI — LLaMA 2 Fine-Tuning Pipeline

An LLM fine-tuning project that builds an empathetic, fact-grounded conversational AI for call centre use. The model is fine-tuned from **LLaMA 2 7B Chat** using Supervised Fine-Tuning (SFT) with DoRA adapters, followed by Direct Preference Optimisation (DPO) to align responses toward empathy and factual faithfulness.

---

## Project Goal

The model acts as a call centre assistant that:
- Acknowledges customer emotions (anxious, dissatisfied, fearful, neutral, etc.)
- Answers only from the provided facts — never hallucinating
- Responds concisely without asking multiple clarifying questions (anti-slot-filling)

---

## Architecture Overview

```
Customer query + emotion + RAG-retrieved facts
           ↓
  LLaMA 2 7B Chat (4-bit quantised)
  + SFT DoRA Adapter (rank=64)
  + DPO Alignment Adapter
           ↓
  Empathetic, fact-grounded 2-sentence response
```

**HuggingFace Hub models:**
- Base: `meta-llama/Llama-2-7b-chat-hf`
- SFT adapter: `Winindux/emowoz-llama2-sft`
- DPO adapter: `Winindux/emowoz-llama2-dpo`
- Dataset: `Winindux/emowoz-llama2-data`

---

## Training Data

| Source | Purpose | Size |
|--------|---------|------|
| EmoWOZ (emowoz-multiwoz.json) | Task-oriented dialogues with emotion labels | ~12,000 turns after filtering |
| ESConv (thu-coai/esconv) | Human-written empathetic support dialogues | ~1,000 supporter turns |
| DPO preference pairs | Chosen (empathetic+faithful) vs rejected (cold/hallucinated/slot-filling) | 1,515 pairs |

### Why two datasets?
EmoWOZ provides knowledge-grounded task dialogues but defaults to a cold, transactional tone. ESConv provides warm, complete-sentence empathetic responses but has no factual grounding. Mixing them gives the model both capabilities.

---

## Pipeline — Notebooks in Order

### Notebook 1 — Data Preparation (`01_data_preparation.ipynb`)

Builds all training data from raw sources.

**What it does:**
1. Loads EmoWOZ dialogues and applies hard quality filters:
   - Removes fragmented responses (space-dot-space pattern: ` . `)
   - Removes `"none"` placeholder leakage (`is none`, `was none`, `are none`)
   - Removes cold responses (<80 chars, no empathy markers) to negative-emotion users
   - Removes slot-filling turns (responses with 2+ question marks)
2. Loads ESConv supporter turns and filters out seeker (user) turns
3. Mixes EmoWOZ + ESConv into SFT training data
4. Oversamples negative-emotion examples (dissatisfied, fearful, anxious, etc.) 3–5× to fix the 70% neutral skew in raw EmoWOZ
5. Splits into `sft_train.jsonl`, `sft_val.jsonl`, `sft_test.jsonl`
6. Generates DPO preference pairs using Groq LLaMA 3 as the rejected-response generator
7. Outputs 1,515 cleaned DPO pairs across 6 rejection types: `emotionally_cold`, `hallucinated`, `irrelevant`, `mixed_failure`, `slot_filling`, `verbose`

**Prompt format used throughout training:**
```
<s>[INST] <<SYS>>
{SYSTEM_PROMPT}
<</SYS>>

Customer emotional state: {emotion}
Facts:
{facts}

Customer inquiry: {customer_query} [/INST]
```

---

### Notebook 2 — Upload to HuggingFace (`02_upload_to_hugginface.ipynb`)

Uploads all processed JSONL files to the private HuggingFace dataset repo (`Winindux/emowoz-llama2-data`):
- `data/train/sft_train.jsonl`
- `data/val/sft_val.jsonl`
- `data/test/sft_test.jsonl`
- `data/dpo_pairs/dpo_pairs.jsonl`

---

### Notebook 3 — SFT Training (`03_sft_training.ipynb`)

Supervised fine-tuning of LLaMA 2 7B Chat with DoRA adapters.

**Platform:** Kaggle (T4 GPU) or Vast AI (A100)

**Key config:**
| Parameter | Value |
|-----------|-------|
| Adapter type | DoRA (Weight-Decomposed LoRA) |
| Rank | 64 |
| Target modules | All attention + FFN layers |
| Epochs | 2 |
| Learning rate | 2e-4 |
| Quantisation | 4-bit NF4 (bitsandbytes) |
| Batch size | 1 × 8 gradient accumulation |
| Optimiser | paged_adamw_8bit |

The adapter is pushed to `Winindux/emowoz-llama2-sft` on HuggingFace Hub after training.

---

### Notebook 4 — DPO Training (`04_dpo_training.ipynb`)

Direct Preference Optimisation on the SFT-trained model.

**Platform:** Vast AI (A100)

**What it does:**
1. Loads the SFT adapter from Hub and wraps it for DPO training
2. Loads 1,515 preference pairs: `{prompt, chosen, rejected}`
3. Trains with DPO loss (β=0.1, sigmoid) — pushes the model toward chosen responses and away from rejected ones
4. Plots the train/eval loss curve
5. Saves the final aligned adapter to `Winindux/emowoz-llama2-dpo`

**Key config:**
| Parameter | Value |
|-----------|-------|
| β (KL penalty) | 0.1 |
| Loss type | Sigmoid |
| Epochs | 2 |
| Learning rate | 5e-5 (cosine schedule) |
| Max sequence length | 512 |
| Gradient accumulation | 16 (effective batch = 16) |

**Rejection types taught:**
- `emotionally_cold` — cold or robotic response to a distressed customer (799 pairs)
- `slot_filling` — fires rapid booking/clarifying questions without answering (383 pairs)
- `verbose` — unnecessarily long rambling response (97 pairs)
- `hallucinated` — invents facts not in the Facts field (85 pairs)
- `irrelevant` — ignores the customer's actual question (78 pairs)
- `mixed_failure` — combines multiple failure modes (73 pairs)

---

### Notebook 5 — Evaluation (`05_evaluation.ipynb`)

Evaluates both SFT and DPO models using Groq `llama-3.3-70b-versatile` as judge.

**Metrics scored per response (0.0–1.0):**
- **Faithfulness** — does the response only use facts from the Facts field?
- **Relevance** — does the response actually answer what the customer asked?
- **Empathy** — does the response acknowledge the customer's emotional state appropriately?

**Evaluation results (Run 2, 100 test examples, DPO model):**

| Metric | SFT | DPO | Change |
|--------|-----|-----|--------|
| Faithfulness | 0.373 | 0.424 | +0.051 |
| Relevance | 0.563 | 0.232 | −0.331 |
| Empathy | 0.567 | 0.705 | +0.138 |
| Composite | 0.501 | 0.454 | −0.047 |

**Key findings:**
- DPO successfully improved empathy (+0.138) — the model now opens with empathetic language for distressed customers
- Relevance collapsed after DPO — the model over-learned emotional support framing at the expense of directly answering questions
- Root cause: ESConv examples (mid-conversation emotional counselling) contaminated the SFT baseline, making the model give open-ended support responses for factual task queries

---

### Notebook 6 — Inference Pipeline (`06_inference_pipeline.py`)

Standalone inference module. Integrates with any RAG pipeline.

**Usage:**
```python
from inference_pipeline import initialize, generate_response

initialize()  # downloads ~14GB model on first run

response = generate_response(
    customer_query="I need a hotel but I'm really stressed about the cost",
    facts="Hotel name: The Crown. Price: moderate. City centre. Phone: 01223111222.",
    emotion="anxious"
)
print(response)
# → "I completely understand your concern — The Crown is in the moderate price range, so it's quite affordable. Would you like their phone number?"
```

**Generation parameters** (tuned to best observed results):
```python
max_new_tokens=80, temperature=0.4, do_sample=True, repetition_penalty=1.4
```

Post-processing trims output to 3 sentences maximum.

---

## Installation

```bash
pip install transformers datasets peft trl accelerate bitsandbytes torch huggingface_hub
```

Set your HuggingFace token:
```bash
export HF_TOKEN=hf_your_token_here
```

For evaluation notebooks on Vast AI, also set:
```bash
export GROQ_API_KEY=gsk_your_key_here
```

---

## Repository Structure

```
6 notebooks/
├── 01_data_preparation.ipynb     # Data cleaning, mixing, DPO pair generation
├── 02_upload_to_hugginface.ipynb # Dataset upload to HuggingFace Hub
├── 03_sft_training.ipynb         # DoRA fine-tuning on cleaned data
├── 04_dpo_training.ipynb         # DPO alignment training
├── 05_evaluation.ipynb           # LLM-as-judge evaluation (faithfulness, relevance, empathy)
└── 06_inference_pipeline.py      # Standalone inference module for RAG integration
```

---

## Known Limitations

- **Relevance regression after DPO:** The DPO-trained model has lower relevance scores (0.232) than SFT (0.563). ESConv training data — which lacks factual grounding — causes the model to give open-ended emotional support responses for task-oriented queries.
- **Slot-filling residue:** LLaMA 2 7B Chat inherited MultiWOZ's slot-collection pattern from pretraining. DPO reduced but did not fully eliminate it.
- **Base model constraint:** LLaMA 2's RLHF conditioning causes assistant-style rambling. Switching to Mistral 7B Instruct v0.3 (better instruction following, no `<<SYS>>` conditioning overhead) is the recommended next step.
