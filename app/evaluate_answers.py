import csv
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv()

client = OpenAI()

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_FILE = PROJECT_ROOT / "evaluation" / "evaluation_set.csv"
RESULTS_FILE = PROJECT_ROOT / "evaluation" / "answer_results.csv"

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
# Evaluate answers
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
        expected_behavior = row[
            "gold_answer_key_phrase_or_expected_behavior"
        ]

        # Only evaluate RAG questions in this phase
        if expected_route not in route_mapping:
            continue

        category = route_mapping[expected_route]

        # Retrieve relevant documents
        retrieved_docs = vector_store.similarity_search(
            question,
            k=3,
            filter={"category": category}
        )

        context = "\n\n".join(
            doc.page_content
            for doc in retrieved_docs
        )

        # --------------------------------------------------
        # Generate grounded answer
        # --------------------------------------------------

        response = client.responses.create(
            model="gpt-5.6-luna",
            instructions="""
You are an AI Training Assistant for a fictional company.

Answer the employee's question using ONLY the provided
company knowledge.

Rules:
- Use only information supported by the context.
- Do not use outside knowledge.
- Do not invent company information.
- Give a clear and concise answer.
- If the context does not contain enough information,
  say that the available company knowledge does not
  provide enough information.
""",
            input=f"""
Employee Question:

{question}

Company Knowledge:

{context}
"""
        )

        answer = response.output_text.strip()


        # --------------------------------------------------
        # Evaluate answer using expected key phrase/behavior
        # --------------------------------------------------

        evaluation = client.responses.create(
            model="gpt-5.6-luna",
            instructions="""
You are evaluating an AI Training Assistant answer.

Determine whether the assistant's answer correctly
addresses the employee question using the expected
answer criteria.

Return ONLY:

PASS

or

FAIL

PASS means:
- The answer contains the important information required
  by the expected answer criteria.
- The answer does not contradict the provided knowledge.

FAIL means:
- Important required information is missing.
- The answer is incorrect.
- The answer contradicts the knowledge.
- The answer invents unsupported information.

Do not explain your decision.
Return only PASS or FAIL.
""",
            input=f"""
Question:
{question}

Expected Answer Criteria:
{expected_behavior}

Assistant Answer:
{answer}

Retrieved Company Knowledge:
{context}
"""
        )

        evaluation_result = evaluation.output_text.strip().upper()

        if evaluation_result == "PASS":
            correct += 1

        total += 1


        # --------------------------------------------------
        # Save result
        # --------------------------------------------------

        results.append({
            "question_id": question_id,
            "question": question,
            "expected_route": expected_route,
            "expected_answer_criteria": expected_behavior,
            "assistant_answer": answer,
            "evaluation": evaluation_result,
        })


        print(
            f"{question_id} | "
            f"Evaluation: {evaluation_result}"
        )

        print(f"Answer: {answer}\n")


# --------------------------------------------------
# Save CSV
# --------------------------------------------------

with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as file:

    fieldnames = [
        "question_id",
        "question",
        "expected_route",
        "expected_answer_criteria",
        "assistant_answer",
        "evaluation",
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(results)


# --------------------------------------------------
# Final score
# --------------------------------------------------

accuracy = (correct / total) * 100 if total else 0


print("\n" + "=" * 55)
print("ANSWER ACCURACY EVALUATION COMPLETE")
print("=" * 55)

print(f"Correct answers: {correct}/{total}")
print(f"Answer Accuracy: {accuracy:.2f}%")

print("\nResults saved to:")
print(RESULTS_FILE)