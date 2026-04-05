import torch
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingManager:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self._model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    def get_embedding(self, sentence: str):
        return self._model.embed_query(sentence)

    def get_embeddings(self, sentences: list[str]):
        return self._model.embed_documents(sentences)

    def get_model(self):
        return self._model


if __name__ == "__main__":
    embedding_manager = EmbeddingManager()

    single_embedding = embedding_manager.get_embedding("Hello World")
    print(single_embedding)

    documents = [
        "Document 1 Content",
        "Document 2 Content",
        "Document 3 Content"
    ]

    doc_embeddings = embedding_manager.get_embeddings(documents)
    for doc_embedding in doc_embeddings:
        print(doc_embedding)
