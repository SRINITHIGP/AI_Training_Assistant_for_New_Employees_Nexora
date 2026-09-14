from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Connect to our existing ChromaDB
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma(
    collection_name="company_knowledge",
    embedding_function=embeddings,
    persist_directory="data/chroma_db",
)


# Ask a test question
question = "How do I submit an expense claim?"


# Search for the most relevant chunks
results = vector_store.similarity_search(
    question,
    k=3
)


# Display the results
print("\nSearch results:\n")

for i, result in enumerate(results):

    print(f"--- Result {i + 1} ---")
    print(result.page_content)
    print(f"Source: {result.metadata.get('source')}")
    print()