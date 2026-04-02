import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from multiagent_rag.agents.emotion_agent import EmotionAgent
from multiagent_rag.agents.finetuned_llm_agent import FinetunedLLMAgent
from multiagent_rag.agents.confidence_agent import ConfidenceAgent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()


def test_integration():
    print("Initializing EmotionAgent...")
    emotion_agent = EmotionAgent()

    print("\nInitializing FinetunedLLMAgent...")
    llm_agent = FinetunedLLMAgent()

    print("\nInitializing ConfidenceAgent...")
    confidence_agent = ConfidenceAgent()

    print("\n--- Testing Emotion Agent ---")
    text_query = "I am so frustrated, my internet is not working again!"
    emotion_result = emotion_agent.detect_from_text(text_query)
    print(f"Detected Emotion from text: {emotion_result}")

    emotion = emotion_result.get("emotion", "neutral")

    print("\n--- Testing Finetuned LLM Agent ---")
    context = "Internet outages in your area are currently being resolved and should be fixed in 2 hours."
    history = [HumanMessage(content=text_query)]

    response = llm_agent.generate(query=text_query, context=context, emotion=emotion, history=history, summary=None)
    print(f"LLM Response: {response}")

    print("\n--- Testing Confidence Agent ---")
    retrieved_chunks = [{"content": context}]
    confidence_result = confidence_agent.evaluate(query=text_query, response=response,
                                                  retrieved_chunks=retrieved_chunks, emotion=emotion)
    print(f"Confidence Evaluation: {confidence_result}")


if __name__ == "__main__":
    test_integration()
