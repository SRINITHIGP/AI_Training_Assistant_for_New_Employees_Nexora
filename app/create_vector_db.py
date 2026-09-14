from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# --------------------------------------------------
# Paths
# --------------------------------------------------

KNOWLEDGE_BASE_PATH = Path("data/knowledge_base")
VECTOR_DB_PATH = "data/chroma_db"


# --------------------------------------------------
# Determine document category
# --------------------------------------------------

def get_category(file_path):

    relative_path = file_path.relative_to(KNOWLEDGE_BASE_PATH)

    folder = relative_path.parts[0]

    if folder == "company":
        return "GENERAL"

    if folder == "roles":
        return "ROLE"

    if folder in ["admin", "policies", "faq"]:
        return "ADMIN"

    return "OTHER"


# --------------------------------------------------
# 1. Load company documents
# --------------------------------------------------

documents = []

for file_path in KNOWLEDGE_BASE_PATH.rglob("*.md"):

    text = file_path.read_text(encoding="utf-8")

    category = get_category(file_path)

    documents.append(
        Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "category": category
            }
        )
    )


print(f"Loaded {len(documents)} documents.")


# --------------------------------------------------
# 2. Split documents into chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")


# --------------------------------------------------
# 3. Create OpenAI embeddings
# --------------------------------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# --------------------------------------------------
# 4. Create ChromaDB
# --------------------------------------------------

vector_store = Chroma(
    collection_name="company_knowledge",
    embedding_function=embeddings,
    persist_directory=VECTOR_DB_PATH,
)


# --------------------------------------------------
# 5. Store chunks with category metadata
# --------------------------------------------------

vector_store.add_documents(chunks)


print("Vector database created successfully!")
print(f"Stored {len(chunks)} chunks in ChromaDB.")


# --------------------------------------------------
# 6. Display category information
# --------------------------------------------------

category_counts = {}

for chunk in chunks:

    category = chunk.metadata["category"]

    category_counts[category] = category_counts.get(category, 0) + 1


print("\nChunks by category:")

for category, count in category_counts.items():

    print(f"{category}: {count}")