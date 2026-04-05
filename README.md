![Project Cover](docs/cover_img.png)

# Adaptive Call Center Automation System

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Next.js](https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-000000?style=for-the-badge&logo=pinecone&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![Clerk](https://img.shields.io/badge/Clerk-6C47FF?style=for-the-badge&logo=clerk&logoColor=white)
![Pusher](https://img.shields.io/badge/Pusher-352A50?style=for-the-badge&logo=pusher&logoColor=white)

This platform is a context-aware and emotionally intelligent AI solution designed for the telecommunications industry. It automates initial support tiers while maintaining a human-centric approach, bridging the "Empathy Gap" in traditional automation through real-time acoustic analysis and stateful multi-agent orchestration.

Built using **LangGraph**, **Pinecone**, and **FastAPI**, the system ensures every customer interaction is grounded in organizational knowledge and handled with the appropriate emotional tone.

---

## Core High-Level Workflow

The system processes interactions through a sophisticated pipeline to ensure accuracy and empathy:

1.  **Input Processing**: Captures live audio and converts it to text via **Speech-to-Text (STT)**. Simultaneously, an **Emotion Detection Model** analyzes acoustic features like pitch, tone, and intensity.
2.  **Multi-Agent Orchestration**: Transcribed text enters a **LangGraph** state machine. Specialized agents decompose queries, validate safety guardrails, and perform semantic searches.
3.  **Grounded Synthesis**: A **Fine-tuned LLM** combines retrieved context with the customer query to draft a professional, empathetic response.
4.  **Deterministic Validation**: A **Confidence Model** scores the response. If confidence is low or severe distress is detected, the system triggers an autonomous human-escalation protocol.
5.  **Output**: Validated responses are converted back to audio via **Text-to-Speech (TTS)** and delivered to the caller.

---

## Technical Deep Dive

### 1. Multi-Agent RAG Pipeline
Unlike standard vector lookups, this system uses a multi-stage validation cycle to eliminate hallucinations.
* **Hybrid Search Logic**: Combines **Dense Vectors** (semantic similarity) with **Sparse BM25 Vectors** (exact keyword matching) for high-precision recall.
* **Neural Reranking**: Utilizes a **Cross-Encoder Model** to score the top 20 retrieved chunks, passing only the top 5 most relevant snippets to the generator.
* **Specialized Agents**: Includes Guardrails (PII scrubbing), Query Decomposers, and Tool Agents for real-time CRM and database lookups.

### 2. Emotion Detection (Acoustic CNN-BiLSTM)
A hybrid model designed to detect immediate emotional states from raw audio.
* **Architecture**: 1D-Convolutional layers for spatial feature extraction (Log-Mel Spectrograms) and BiLSTM for temporal memory.
* **Attention Mechanism**: Focuses on emotionally charged frames (e.g., sudden pitch rises) to ensure robustness against diverse accents.

### 3. Fine-Tuned Large Language Model
A **Llama-2-7B-Chat** model adapted for the telecom domain using **QLoRA** and **Direct Preference Optimization (DPO)**.
* **Dataset**: Trained on the **EmoWOZ** corpus, featuring 11,000+ task-oriented dialogues with emotion annotations.
* **Instruction Tuning**: Conditioned to generate responses that are both factually grounded in context and emotionally aligned with the caller's state.

### 4. Confidence Calculation Model
A deterministic safety net utilizing an **XGBoost classifier** to evaluate response reliability.
* **Hybrid Features**: Analyzes semantic embeddings alongside psycholinguistic markers (e.g., hesitation fillers vs. assertive language).
* **Action**: Proactively flags potential hallucinations, serving as the primary trigger for human-in-the-loop escalation.

---

## Project Structure

```text
├── frontend/             # Next.js web application (TypeScript/Tailwind CSS)
├── rag-multi-agent/      # LangGraph orchestration, Pinecone integration, & agents
├── emotion-model/        # CNN-BiLSTM training & real-time audio inference
├── confidence-model/     # XGBoost classification & linguistic feature extraction
├── finetuned-LLM/        # QLoRA adapters, DPO scripts, & Llama-2 inference
├── api/                  # FastAPI backend with RESTful RAG & ingestion endpoints
├── docs/                 # Proposal, Final Thesis, SRS, & System Diagrams
└── trails/               # Experimental notebooks (EDA, Benchmarking, Ablation)lala```
```
---

## Getting Started

### Prerequisites
* **Hardware**: NVIDIA GPU (RTX 3060 12GB+), 16GB RAM, 1TB NVMe SSD.
* **Software**: Python 3.12, Node.js (latest), and NVIDIA CUDA drivers.

### Installation & Setup

1.  **Initialize Environment**
    The system uses `uv` for lightning-fast dependency management.
    ```bash
    uv sync
    ```

2.  **Launch the Backend API**
    Start the FastAPI server from the root directory:
    ```bash
    uv run uvicorn api.main:app --reload
    ```
    *Interactive Swagger documentation is available at `http://localhost:8000/docs`.*

3.  **Launch the Frontend**
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

---

## Deployment Configuration

### Backend (.env)
```env
PINECONE_API_KEY=xxx
OPENAI_API_KEY=xxx
GROQ_API_KEY=xxx
LANGFUSE_PUBLIC_KEY=xxx
LANGFUSE_SECRET_KEY=xxx
```

### Frontend (.env)
```env
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=xxx
CLERK_SECRET_KEY=xxx

NEXT_PUBLIC_PUSHER_KEY=xxx
PUSHER_APP_ID=xxx
PUSHER_KEY=xxx
PUSHER_SECRET=xxx
```

---

## Contributors
* **Danula Rathnayaka** - Multi-Agent RAG & Backend
* **Arosha Withanage** - Confidence Model & Frontend
* **Winindu Rajapaksha** - Generative AI & LLM Fine-tuning
* **Thevinu Dassanayaka** - Machine Learning & Emotion Detection

---
**License**: This project is licensed under the **MIT License**.
