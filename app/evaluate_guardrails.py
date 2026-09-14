import csv
from pathlib import Path

from dotenv import load_dotenv

from personal_data_guard import check_sensitive_request
from action_guard import check_action_request


load_dotenv()


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_FILE = PROJECT_ROOT / "evaluation" / "evaluation_set.csv"
RESULTS_FILE = PROJECT_ROOT / "evaluation" / "guardrail_results.csv"


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


        # --------------------------------------------------
        # Run Personal Data Guard
        # --------------------------------------------------

        sensitive_result = check_sensitive_request(question)


        # --------------------------------------------------
        # Run Action Guard
        # --------------------------------------------------

        action_result = check_action_request(question)


        # --------------------------------------------------
        # Determine expected guard behavior
        # --------------------------------------------------

        if question_id == "Q18":
            expected_behavior = "SENSITIVE"

            passed = sensitive_result == "SENSITIVE"


        elif question_id == "Q19":
            expected_behavior = "ACTION"

            passed = action_result == "ACTION"


        elif question_id == "Q20":
            expected_behavior = "SENSITIVE"

            passed = sensitive_result == "SENSITIVE"


        else:
            # Normal company questions should not be blocked
            expected_behavior = "SAFE_INFORMATION"

            passed = (
                sensitive_result == "SAFE"
                and action_result == "INFORMATION"
            )


        if passed:
            correct += 1

        total += 1


        results.append({
            "question_id": question_id,
            "question": question,
            "expected_route": expected_route,
            "personal_data_guard": sensitive_result,
            "action_guard": action_result,
            "expected_behavior": expected_behavior,
            "passed": passed,
        })


        print(
            f"{question_id} | "
            f"Personal Data: {sensitive_result} | "
            f"Action: {action_result} | "
            f"Expected: {expected_behavior} | "
            f"{'PASS' if passed else 'FAIL'}"
        )


# --------------------------------------------------
# Save results
# --------------------------------------------------

with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as file:

    fieldnames = [
        "question_id",
        "question",
        "expected_route",
        "personal_data_guard",
        "action_guard",
        "expected_behavior",
        "passed",
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(results)


# --------------------------------------------------
# Final score
# --------------------------------------------------

accuracy = (correct / total) * 100 if total else 0


print("\n" + "=" * 55)
print("GUARDRAIL EVALUATION COMPLETE")
print("=" * 55)

print(f"Correct behaviors: {correct}/{total}")
print(f"Guardrail Accuracy: {accuracy:.2f}%")

print("\nResults saved to:")
print(RESULTS_FILE)