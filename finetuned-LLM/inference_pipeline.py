import os
import re

import dotenv
from huggingface_hub import InferenceClient

dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
SFT_REPO = "Winindux/emowoz-llama2-dpo"

SYSTEM_PROMPT = ("You are a compassionate and helpful assistant. Follow these rules strictly:\n"
                 "1. Respond using ONLY the information provided in the Facts section.\n"
                 "2. Never add information that is not present in the Facts.\n"
                 "3. Adapt your tone to match the user's emotional state.\n"
                 "4. Be warm and empathetic while remaining factually accurate.")

_client = None


def initialize():
    global _client
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN environment variable is not set. "
                           "Add HF_TOKEN=hf_your_token to your .env file.")
    _client = InferenceClient(model=SFT_REPO, token=HF_TOKEN)
    print(f"HuggingFace Inference client ready — model: {SFT_REPO}")


def format_prompt(customer_query: str, emotion: str, facts: str) -> str:
    user_content = (f"Emotional State: {emotion}\n"
                    f"Facts: {facts}\n"
                    f"Conversation History:\nNo previous turns.\n"
                    f"User: {customer_query}")
    return (f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
            f"{user_content} [/INST]")


def generate_response(customer_query: str, facts: str, emotion: str = "neutral", max_new_tokens: int = 200, ) -> str:
    if _client is None:
        raise RuntimeError("Client not initialized. Call initialize() first.")

    prompt = format_prompt(customer_query, emotion, facts)

    response = _client.text_generation(prompt, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9,
                                       repetition_penalty=1.1, do_sample=True, )

    text = response.strip() if isinstance(response, str) else str(response).strip()
    text = re.sub(r"\[/?INST\]|<s>|</s>", "", text).strip()
    return text
