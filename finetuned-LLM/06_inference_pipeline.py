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

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


HF_TOKEN   = os.getenv("HF_TOKEN")
BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
DPO_REPO   = "Winindux/emowoz-llama2-dpo"


SYSTEM_PROMPT = """You are a call centre assistant. You MUST follow every rule below:

1. FACTS ONLY: Use ONLY the facts provided. Never invent information not in the facts.

2. EMPATHY FIRST — MANDATORY: Your very first words must acknowledge the customer emotion:
   - emotion is "anxious": begin with "I completely understand your concern —" or "I can see this is worrying —"
   - emotion is "dissatisfied": begin with "I'm really sorry about that —" or "I sincerely apologise —"
   - emotion is "fearful": begin with "Please don't worry —" or "You'll be absolutely fine —"
   - emotion is "neutral" or "satisfied": no empathy opener needed, just answer directly.

3. CONCISE: Maximum 2 sentences total. No more.

4. ONE QUESTION MAX: Ask at most one follow-up. Never ask multiple questions.

5. NO SLOT-FILLING: Do NOT ask about arrival dates, parking, price range, or booking preferences \
unless the customer specifically mentioned them.

Examples of correct responses:

Customer emotional state: anxious
Facts: Hotel name: The Crown. Price: moderate. Located in city centre. Phone: 01223111222.
Customer inquiry: I'm worried about the cost, is it expensive?
Response: I completely understand your concern — The Crown is in the moderate price range, so it's quite affordable. Would you like their phone number?

Customer emotional state: dissatisfied
Facts: No pool available. Free WiFi. Check-out at 11am.
Customer inquiry: I was really hoping for a pool. Do you have one?
Response: I'm really sorry about that — unfortunately there's no pool available here. Is there anything else I can help with?

Customer emotional state: fearful
Facts: Flight is on time. Gate: B12. Boarding at 14:30.
Customer inquiry: My connecting flight is in 45 minutes, am I going to make it?
Response: Please don't worry — your flight is on time and boarding starts at 14:30 at gate B12, so you should be fine.

Customer emotional state: neutral
Facts: Restaurant name: The Golden Curry. Area: city centre. Price range: moderate. Phone: 01223366611. Open daily 12pm-10pm.
Customer inquiry: Do you have a restaurant recommendation?
Response: I recommend The Golden Curry in the city centre — it is moderately priced and open daily from 12pm to 10pm."""


model     = None
tokenizer = None


def format_prompt(customer_query: str, emotion: str, facts: str) -> str:
    prompt = (
        "<s>[INST] <<SYS>>\n"
        + SYSTEM_PROMPT + "\n"
        + "<</SYS>>\n\n"
        + f"Customer emotional state: {emotion}\n"
        + f"Facts: {facts}\n\n"
        + f"Customer inquiry: {customer_query} [/INST]"
    )
    return prompt


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
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
    )

    print(f"Applying DPO adapter from {DPO_REPO}...")
    model = PeftModel.from_pretrained(base, DPO_REPO, token=HF_TOKEN)
    model.eval()

    print("Model loaded and ready!")


def generate_response(
    customer_query: str,
    facts: str,
    emotion: str = "neutral",
    max_new_tokens: int = 80
) -> str:
    """
    Takes a customer query + facts from RAG + emotion → returns empathetic response.

    Parameters:
        customer_query  — the customer's question (string)
        facts           — relevant facts retrieved by RAG (string)
        emotion         — detected emotion e.g. 'dissatisfied', 'anxious', 'neutral' (string)
        max_new_tokens  — max length of response (default 80)

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

    prompt = format_prompt(customer_query, emotion, facts)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.4,
            do_sample=True,
            repetition_penalty=1.4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    sentences = [s.strip() for s in response.replace("?", "?.").replace("!", "!.").split(".") if s.strip()]
    response  = ". ".join(sentences[:3]).strip()
    if response and response[-1] not in ".?!":
        response += "."

    return response


if __name__ == "__main__":
    initialize()

    tests = [
        ("anxious",      "Hotel name: The Gonville. Price: expensive. City centre. Phone: 01223366611.",
                         "I need a hotel but I'm really stressed about the cost"),
        ("dissatisfied", "The hotel has free WiFi. No swimming pool available. Check-out at 11am.",
                         "I was really hoping for a pool. Do you have one?"),
        ("fearful",      "Flight is on time. Gate: B12. Boarding at 14:30.",
                         "My connecting flight is in 45 minutes, am I going to make it?"),
        ("neutral",      "Restaurant name: The Golden Curry. Area: city centre. Price range: moderate. Phone: 01223366611. Open daily 12pm-10pm.",
                         "Do you have a restaurant recommendation?"),
    ]

    for emotion, facts, query in tests:
        print(f"Emotion  : {emotion}")
        print(f"Customer : {query}")
        reply = generate_response(query, facts, emotion)
        print(f"System   : {reply}")
        print()
