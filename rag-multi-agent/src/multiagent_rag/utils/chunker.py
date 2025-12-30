from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
            is_separator_regex=False
        )

    def split_text(self, text: str, metadata: dict) -> list:
        if not text:
            return []

        docs = self.splitter.create_documents([text], metadatas=[metadata])

        clean_chunks = []
        for doc in docs:
            clean_chunks.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        return clean_chunks


if __name__ == "__main__":
    text = """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
        Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
        nisi ut aliquip ex ea commodo consequat.
        Duis aute irure dolor in reprehenderit in voluptate velit esse
        cillum dolore eu fugiat nulla pariatur.
        Excepteur sint occaecat cupidatat non proident, sunt in culpa
        qui officia deserunt mollit anim id est laborum.
        """

    metadata = {
        "source": "example_text_source",
        "category": "Test"
    }

    chunker = Chunker(chunk_size=100, chunk_overlap=20)

    chunks = chunker.split_text(text, metadata)

    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx + 1}:")
        print("Content:", chunk["content"])
        print("Metadata:", chunk["metadata"])
        print("-" * 50)
