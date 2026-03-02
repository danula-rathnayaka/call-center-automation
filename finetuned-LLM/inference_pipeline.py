"""
=============================================================
  Inference Pipeline — Winindux/emowoz-llama2-dpo
  LLaMA 2 7B fine-tuned for empathetic + grounded responses
=============================================================

HOW YOUR FRIEND USES THIS FILE:
─────────────────────────────────────────────────────────────
  Step 1 — Install requirements (one time only):
      pip install transformers peft accelerate bitsandbytes torch

  Step 2 — At the top of his RAG file:
      from inference_pipeline import initialize, generate_response

  Step 3 — Load the model once at the start:
      initialize()

  Step 4 — Call your function whenever he needs a response:
      reply = generate_response(
          customer_query = "Where is my taxi?",
          facts          = "Taxi car: tesla. Contact: 01223 456789.",
          emotion        = "anxious"
      )
      print(reply)
─────────────────────────────────────────────────────────────
"""

# ── 1. IMPORTS ───────────────────────────────────────────────────────────────
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os  


# ── 2. CONFIG ────────────────────────────────────────────────────────────────
HF_TOKEN   = os.getenv("HF_TOKEN") 
BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
DPO_REPO   = "Winindux/emowoz-llama2-dpo"


# ── 3. SYSTEM PROMPT (must match training exactly — do not change!) ───────────
SYSTEM_PROMPT = """You are a compassionate and helpful assistant. Follow these rules strictly:
1. Respond using ONLY the information provided in the Facts section.
2. Never add information that is not present in the Facts.
3. Adapt your tone to match the user's emotional state.
4. Be warm and empathetic while remaining factually accurate."""


# ── 4. These start empty — model loads only when initialize() is called ───────
model     = None
tokenizer = None


# ── 5. PROMPT FORMATTER (must match training format exactly — do not change!) ─
def format_prompt(customer_query: str, emotion: str, facts: str, history: list = None) -> str:
    history_str = "\n".join(history) if history else "No previous turns."
    user_content = (
        f"Emotional State: {emotion}\n"
        f"Facts: {facts}\n"
        f"Conversation History:\n{history_str}\n"
        f"User: {customer_query}"
    )
    prompt = (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_content} [/INST]"
    )
    return prompt


# ── 6. MODEL LOADER ───────────────────────────────────────────────────────────
def initialize():
    """
    Loads the model into memory.
    Your friend calls this ONCE at the start of his script — not on every query.
    Takes a few minutes the first time (downloads ~14GB base model).
    """
    global model, tokenizer

    print(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16
    )

    print(f"Applying DPO adapter from {DPO_REPO}...")
    model = PeftModel.from_pretrained(base, DPO_REPO, token=HF_TOKEN)
    model.eval()

    print("✅ Model loaded and ready!")


# ── 7. THE CORE INFERENCE FUNCTION (this is what your friend imports) ─────────
def generate_response(
    customer_query: str,
    facts: str,
    emotion: str = "neutral",
    max_new_tokens: int = 200
) -> str:
    """
    Takes a customer query + facts from RAG + emotion → returns empathetic response.

    Parameters:
        customer_query  — the customer's question (string)
        facts           — relevant facts retrieved by RAG (string)
        emotion         — detected emotion e.g. 'dissatisfied', 'anxious', 'neutral' (string)
        max_new_tokens  — max length of response (default 200)

    Returns:
        response        — empathetic, fact-grounded reply (string)

    Example:
        reply = generate_response(
            customer_query = "Where is my taxi?",
            facts          = "Taxi car: tesla. Taxi contact: 01223 456789.",
            emotion        = "anxious"
        )
    """
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded. Please call initialize() first.")

    prompt = format_prompt(customer_query, emotion, facts)  # same — no change needed here
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Strip the prompt tokens — keep only the new generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response
