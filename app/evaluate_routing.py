import csv
from pathlib import Path

from app.classifier import classify_question


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_FILE = PROJECT_ROOT / "evaluation" / "evaluation_set.csv"
RESULTS_FILE = PROJECT_ROOT / "evaluation" / "routing_results.csv"


results = []
correct = 0
total = 0


with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        question_id = row["question_id"]
        question = row["question"]
        expected_route = row["expected_route"]

        # Convert evaluation-set route names
        # into the classifier's route names.
        route_mapping = {
            "general_company": "GENERAL",
            "role_specific": "ROLE",
            "admin_policy": "ADMIN",
            "direct_llm": "OTHER",
        }

        expected_classifier_route = route_mapping[expected_route]

        predicted_route = classify_question(question)

        is_correct = predicted_route == expected_classifier_route

        if is_correct:
            correct += 1

        total += 1

        results.append({
            "question_id": question_id,
            "question": question,
            "expected_route": expected_route,
            "predicted_route": predicted_route,
            "correct": is_correct,
        })

        print(
            f"{question_id} | "
            f"Expected: {expected_classifier_route} | "
            f"Predicted: {predicted_route} | "
            f"{'PASS' if is_correct else 'FAIL'}"
        )


# Save results
with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as file:
    fieldnames = [
        "question_id",
        "question",
        "expected_route",
        "predicted_route",
        "correct",
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)


accuracy = (correct / total) * 100 if total else 0


print("\n" + "=" * 50)
print("ROUTING EVALUATION COMPLETE")
print("=" * 50)
print(f"Correct: {correct}/{total}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"\nResults saved to:")
print(RESULTS_FILE)