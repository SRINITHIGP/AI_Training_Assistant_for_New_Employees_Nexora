from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Location of our company knowledge base
KNOWLEDGE_BASE_PATH = Path("data/knowledge_base")


# Load all Markdown documents
documents = []

for file_path in KNOWLEDGE_BASE_PATH.rglob("*.md"):

    text = file_path.read_text(encoding="utf-8")

    documents.append(
        Document(
            page_content=text,
            metadata={
                "source": str(file_path)
            }
        )
    )


print(f"Loaded {len(documents)} documents.")


# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)

chunks = text_splitter.split_documents(documents)


print(f"Created {len(chunks)} chunks.")


# Show a small preview
for i, chunk in enumerate(chunks[:5]):

    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content[:500])
    print(f"Source: {chunk.metadata['source']}")