import csv
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from app.classifier import classify_question
from app.evidence_checker import check_evidence
from app.personal_data_guard import check_sensitive_request
from app.action_guard import check_action_request
from app.ambiguity_guard import check_ambiguity
from app.followup_suggestions import generate_followups

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv()

client = OpenAI()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma(
    collection_name="company_knowledge",
    embedding_function=embeddings,
    persist_directory="data/chroma_db",
)


# --------------------------------------------------
# Expected routing
# --------------------------------------------------

route_mapping = {
    "general_company": "GENERAL",
    "role_specific": "ROLE",
    "admin_policy": "ADMIN",
    "direct_llm": "OTHER",
}


# --------------------------------------------------
# Load evaluation questions
# --------------------------------------------------

evaluation_file = Path("evaluation/evaluation_set.csv")

with open(evaluation_file, "r", encoding="utf-8") as file:
    questions = list(csv.DictReader(file))


# --------------------------------------------------
# Evaluate one question
# --------------------------------------------------

def evaluate_question(row):

    question_id = row["question_id"]
    question = row["question"]
    expected_route = route_mapping[row["expected_route"]]

    print("\n" + "=" * 70)
    print(f"{question_id} — {question}")
    print("=" * 70)

    # ----------------------------------------------
    # 1. Sensitive data guard
    # ----------------------------------------------

    sensitive_result = check_sensitive_request(question)

    # ----------------------------------------------
    # 2. Action guard
    # ----------------------------------------------

    action_result = check_action_request(question)

    # ----------------------------------------------
    # 3. Routing
    # ----------------------------------------------

    predicted_route = classify_question(question)

    print(f"Expected route:   {expected_route}")
    print(f"Predicted route:  {predicted_route}")

    route_pass = predicted_route == expected_route

    # ----------------------------------------------
    # 4. Out-of-scope / sensitive / action handling
    # ----------------------------------------------

    if sensitive_result == "SENSITIVE":

        final_behavior = "REFUSE_SENSITIVE_REQUEST"

        print("Sensitive guard:   SENSITIVE")
        print("Final behavior:   Refuse sensitive request")

    elif action_result == "ACTION":

        final_behavior = "REFUSE_ACTION"

        print("Action guard:      ACTION")
        print("Final behavior:   Explain action cannot be performed")

    elif predicted_route == "OTHER":

        final_behavior = "OUT_OF_SCOPE"

        print("Final behavior:   Politely refuse out-of-scope question")

    else:

        # ------------------------------------------
        # 5. Ambiguity check
        # ------------------------------------------

        ambiguity_result = check_ambiguity(
            question,
            "No previous conversation."
        )

        print(f"Ambiguity result:  {ambiguity_result}")

        if ambiguity_result == "AMBIGUOUS":

            final_behavior = "CLARIFICATION"

            print("Final behavior:   Ask for clarification")

        else:

            # --------------------------------------
            # 6. Retrieve knowledge
            # --------------------------------------

            results = vector_store.similarity_search(
                question,
                k=3,
                filter={"category": predicted_route}
            )

            context = "\n\n".join(
                result.page_content
                for result in results
            )

            # --------------------------------------
            # 7. Evidence check
            # --------------------------------------

            evidence_result = check_evidence(
                question,
                context
            )

            print(f"Evidence result:   {evidence_result}")

            if evidence_result == "NOT_SUPPORTED":

                final_behavior = "KNOWLEDGE_BASE_FALLBACK"

                print(
                    "Final behavior:   Explain that the information "
                    "is unavailable."
                )

            else:

                # ----------------------------------
                # 8. Generate grounded answer
                # ----------------------------------

                response = client.responses.create(
                    model="gpt-5.6-luna",
                    instructions="""
You are an AI Training Assistant for company employees.

Answer the employee's question using ONLY the provided
company knowledge.

Rules:

- Do not use outside knowledge.
- Do not invent company information.
- Be concise and helpful.
- If the information is not available, say so.
- Do not claim to perform actions.
""",
                    input=f"""
Employee Question:

{question}

Company Knowledge:

{context}
"""
                )

                answer = response.output_text.strip()

                final_behavior = "GROUNDED_ANSWER"

                print(f"Answer: {answer}")

                # ----------------------------------
                # 9. Generate follow-ups
                # ----------------------------------

                followups = generate_followups(
                    question,
                    answer,
                    context
                )

                try:

                    if isinstance(followups, str):
                        followups = json.loads(followups)

                except json.JSONDecodeError:

                    followups = []

                print("Follow-ups:")

                for followup in followups:
                    print(f"- {followup}")

    # ------------------------------------------------
    # Result
    # ------------------------------------------------

    print(f"Final behavior:   {final_behavior}")

    return {
        "id": question_id,
        "question": question,
        "expected_route": expected_route,
        "predicted_route": predicted_route,
        "route_pass": route_pass,
        "sensitive_guard": sensitive_result,
        "action_guard": action_result,
        "final_behavior": final_behavior,
    }


# --------------------------------------------------
# Run evaluation
# --------------------------------------------------

results = []

for row in questions:

    result = evaluate_question(row)

    results.append(result)


# --------------------------------------------------
# Save results
# --------------------------------------------------

output_file = Path("evaluation/end_to_end_results.csv")

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


# --------------------------------------------------
# Summary
# --------------------------------------------------

route_accuracy = (
    sum(result["route_pass"] for result in results)
    / len(results)
) * 100


print("\n")
print("=" * 70)
print("END-TO-END EVALUATION COMPLETE")
print("=" * 70)

print(f"Questions evaluated: {len(results)}")
print(f"Routing accuracy:    {route_accuracy:.2f}%")
print(f"Results saved to:    {output_file}")