import csv
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_FILE = PROJECT_ROOT / "evaluation" / "evaluation_set.csv"
RESULTS_FILE = PROJECT_ROOT / "evaluation" / "retrieval_results.csv"

VECTOR_DB_PATH = PROJECT_ROOT / "data" / "chroma_db"


# --------------------------------------------------
# Load ChromaDB
# --------------------------------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma(
    collection_name="company_knowledge",
    embedding_function=embeddings,
    persist_directory=str(VECTOR_DB_PATH),
)


# --------------------------------------------------
# Route mapping
# --------------------------------------------------

route_mapping = {
    "general_company": "GENERAL",
    "role_specific": "ROLE",
    "admin_policy": "ADMIN",
}


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

results = []

correct = 0
total = 0


with open(EVALUATION_FILE, "r", encoding="utf-8") as file:

    reader = csv.DictReader(file)

    for row in reader:

        question_id = row["question_id"]
        question = row["question"]
        expected_route = row["expected_route"]
        gold_source = row["gold_source"]

        # Skip non-RAG questions such as Q18-Q20
        if expected_route not in route_mapping:
            continue

        category = route_mapping[expected_route]

        # Retrieve top 3 results from the correct category
        retrieved_docs = vector_store.similarity_search(
            question,
            k=3,
            filter={"category": category}
        )

        retrieved_sources = [
            doc.metadata.get("source", "")
            for doc in retrieved_docs
        ]

        # Check whether expected source appears in Top 3
        source_found = gold_source in retrieved_sources

        if source_found:
            correct += 1

        total += 1

        results.append({
            "question_id": question_id,
            "question": question,
            "expected_route": expected_route,
            "gold_source": gold_source,
            "retrieved_sources": " | ".join(retrieved_sources),
            "source_found_in_top_3": source_found,
        })

        print(
            f"{question_id} | "
            f"Expected Source: {gold_source} | "
            f"Retrieved: {retrieved_sources} | "
            f"{'PASS' if source_found else 'FAIL'}"
        )


# --------------------------------------------------
# Save results
# --------------------------------------------------

with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as file:

    fieldnames = [
        "question_id",
        "question",
        "expected_route",
        "gold_source",
        "retrieved_sources",
        "source_found_in_top_3",
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(results)


# --------------------------------------------------
# Final score
# --------------------------------------------------

accuracy = (correct / total) * 100 if total else 0


print("\n" + "=" * 55)
print("RETRIEVAL EVALUATION COMPLETE")
print("=" * 55)

print(f"Correct retrievals: {correct}/{total}")
print(f"Top-3 Retrieval Accuracy: {accuracy:.2f}%")

print("\nResults saved to:")
print(RESULTS_FILE)