# 📞 Call Center Automation System

This project is an AI-powered call center automation solution designed to improve customer satisfaction, reduce waiting times, and enhance operational efficiency.
It integrates **Retrieval-Augmented Generation (RAG)**, **a Multi-Agent System**, **Emotion Detection**, **Confidence Evaluation**, and a **Fine-tuned LLM** to deliver accurate, empathetic, and context-aware responses.

---

## 🚀 Key Features

### 🔹 1. Multi-Agent System + RAG

* Upload PDFs, documents, or URLs
* Automated text extraction, chunking & embedding
* Vector database storage
* Retrieval-augmented responses using a fine-tuned LLM
* Coordinated agent workflow for high accuracy

### 🔹 2. Emotion Detection Model

* Processes customer voice input
* Extracts acoustic features
* Classifies emotions in real-time
* Helps the system adapt tone and determine escalation needs

### 🔹 3. Confidence Evaluation Model

* Validates AI-generated responses against retrieved knowledge
* Measures emotional alignment
* Outputs a confidence score (0–1)
* Escalates the call automatically if confidence is low

### 🔹 4. Fine-Tuned LLM

* Custom training using domain-specific customer service datasets
* Knowledge-grounded responses
* Emotion-aware instructions for empathetic conversations
* Continuous improvement from feedback loops

## 📁 Project Structure

```
├── README.md
├── LICENSE
├── .gitignore
│
├── frontend/
│   └── (Frontend application logic)
│
├── rag-multi-agent/
│   └── (RAG pipeline, agents, vector DB)
│
├── emotion-model/
│   └── (Audio processing & emotion classification)
│
├── confidence-model/
│   └── (Response evaluation & escalation logic)
│
├── finetuned-LLM/
│   └── (Fine-tuning scripts & inference)
│
└── docs/
    └── (Proposal, SRS, diagrams, documentation)
```


## 🧠 System Workflow (High-Level)

1. **Customer query/voice input → Emotion Model**
2. **Query text → RAG Retriever**
3. **Relevant knowledge chunks → LLM Generator**
4. **Response + emotion → Confidence Model**
5. **High confidence → Continue AI conversation**
6. **Low confidence → Escalate to human agent**


## 👥 Contributors

* **Danula Rathnayaka – 20231649**
* **Arosha Withanage – 20231983**
* **Winindu Rajapaksha – 20232406**
* **Thevinu Dassanayaka – 20240857**


## 📚 Documentation

All documents are located in the `/docs` directory.


## 📄 License

This project is licensed under the **MIT License**.