from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    _instance = None
    _splitter = None

    def __new__(cls, chunk_size=500, chunk_overlap=50):
        if cls._instance is None:
            cls._instance = super(Chunker, cls).__new__(cls)
            cls._instance._initialize_splitter(chunk_size, chunk_overlap)
        return cls._instance

    def _initialize_splitter(self, chunk_size, chunk_overlap):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
            is_separator_regex=False
        )

    def split_text(self, text: str, metadata: dict) -> list:
        if not text:
            return []

        docs = self._splitter.create_documents([text], metadatas=[metadata])

        clean_chunks = []
        for doc in docs:
            clean_chunks.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        return clean_chunks
