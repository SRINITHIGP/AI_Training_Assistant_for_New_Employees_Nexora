import json

from dotenv import load_dotenv
from openai import OpenAI

from ambiguity_guard import check_ambiguity
from followup_suggestions import generate_followups


load_dotenv()

client = OpenAI()


# --------------------------------------------------
# Test 1: Contextual question
# --------------------------------------------------

print("\n" + "=" * 55)
print("TEST 1 — CONTEXTUAL QUESTION")
print("=" * 55)

history = [
    {
        "role": "user",
        "content": "What does a Data Analyst do?"
    },
    {
        "role": "assistant",
        "content": "A Data Analyst works with business metrics, reports, dashboards, and data analysis."
    }
]

current_question = "What tools do they use?"


history_text = "\n\n".join(
    f"{'Employee' if message['role'] == 'user' else 'Assistant'}: {message['content']}"
    for message in history
)


ambiguity_result = check_ambiguity(
    current_question,
    history_text
)

print(f"Question: {current_question}")
print(f"Ambiguity result: {ambiguity_result}")


response = client.responses.create(
    model="gpt-5.6-luna",
    instructions="""
Rewrite the current employee question so it can be
understood independently using the conversation history.

Resolve references such as:
they, them, it, that, this, those.

Do not answer the question.

Return ONLY the rewritten question.
""",
    input=f"""
Previous conversation:

{history_text}

Current question:

{current_question}
"""
)

rewritten_question = response.output_text.strip()

print(f"Rewritten question: {rewritten_question}")


context_test_pass = (
    ambiguity_result == "CLEAR"
    and "Data Analyst" in rewritten_question
)

print(f"Result: {'PASS' if context_test_pass else 'FAIL'}")


# --------------------------------------------------
# Test 2: Ambiguous question
# --------------------------------------------------

print("\n" + "=" * 55)
print("TEST 2 — AMBIGUOUS QUESTION")
print("=" * 55)

ambiguous_question = "What tools do we use?"

ambiguity_result = check_ambiguity(
    ambiguous_question,
    "No previous conversation."
)

print(f"Question: {ambiguous_question}")
print(f"Ambiguity result: {ambiguity_result}")

ambiguity_test_pass = ambiguity_result == "AMBIGUOUS"

print(f"Result: {'PASS' if ambiguity_test_pass else 'FAIL'}")


# --------------------------------------------------
# Test 3: Follow-up suggestions
# --------------------------------------------------

print("\n" + "=" * 55)
print("TEST 3 — FOLLOW-UP SUGGESTIONS")
print("=" * 55)

answer = """
A Data Analyst is responsible for understanding business
metrics, producing reports, updating dashboards, and
documenting metric definitions.
"""

followup_context = """
A Data Analyst is responsible for understanding business
metrics, producing reports, updating dashboards, and
documenting metric definitions.
"""

followups = generate_followups(
    "What does a Data Analyst do?",
    answer,
    followup_context
)

# Convert JSON string to Python list if necessary
if isinstance(followups, str):
    try:
        followups = json.loads(followups)
    except json.JSONDecodeError:
        followups = []

print("Generated follow-ups:")

for suggestion in followups:
    print(f"- {suggestion}")


# Check that suggestions were generated
followup_test_pass = (
    isinstance(followups, list)
    and len(followups) > 0
    and len(followups) <= 4
)

print(f"Result: {'PASS' if followup_test_pass else 'FAIL'}")


# --------------------------------------------------
# Test 4: Follow-up relevance
# --------------------------------------------------

print("\n" + "=" * 55)
print("TEST 4 — FOLLOW-UP RELEVANCE")
print("=" * 55)

relevance_check = client.responses.create(
    model="gpt-5.6-luna",
    instructions="""
Evaluate whether the suggested follow-up questions are
relevant to the employee's original question and answer.

Return ONLY:

PASS

if the suggestions are clearly relevant.

Otherwise return:

FAIL
""",
    input=f"""
Original question:

What does a Data Analyst do?

Answer:

{answer}

Suggested follow-ups:

{json.dumps(followups)}
"""
)

relevance_result = relevance_check.output_text.strip().upper()

print(f"Relevance evaluation: {relevance_result}")

followup_relevance_pass = relevance_result == "PASS"

print(
    f"Result: "
    f"{'PASS' if followup_relevance_pass else 'FAIL'}"
)


# --------------------------------------------------
# Final summary
# --------------------------------------------------

tests = [
    context_test_pass,
    ambiguity_test_pass,
    followup_test_pass,
    followup_relevance_pass,
]

passed = sum(tests)
total = len(tests)

accuracy = (passed / total) * 100


print("\n" + "=" * 55)
print("CONVERSATION EVALUATION COMPLETE")
print("=" * 55)

print(f"Passed: {passed}/{total}")
print(f"Conversation Feature Accuracy: {accuracy:.2f}%")