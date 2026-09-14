from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def generate_followups(question, answer, context):

    response = client.responses.create(

        model="gpt-5.6-luna",

        instructions="""
You generate helpful follow-up questions for a company
AI Training Assistant.

Your suggestions must help a new employee explore information
that is explicitly available in the provided company knowledge.

IMPORTANT:
Every follow-up question MUST be answerable using ONLY the
provided Relevant Company Knowledge.

Rules:

- Generate exactly 3 follow-up questions.
- Each suggestion must be directly related to the employee's
  current question.
- Each suggestion must be answerable using information explicitly
  stated in the Relevant Company Knowledge.
- Do NOT create a question simply because it is generally related
  to the employee's role, topic, or industry.
- Do NOT ask about information that is missing from the context.
- Do NOT invent expectations, requirements, timelines, tools,
  responsibilities, processes, or policies.
- Do NOT infer information that is not explicitly stated.
- Prefer questions about specific facts, processes, tools,
  responsibilities, timelines, or policies that appear in the
  Relevant Company Knowledge.
- Do not suggest unrelated topics.
- Do not suggest personal or sensitive information.
- Do not suggest actions that require system access.
- Keep each suggestion short and natural.
- Return ONLY a JSON array of 3 strings.
- Do not include markdown.
- Do not include explanations.

Before returning each suggestion, silently check:

1. Is this question related to the current employee question?
2. Is the answer explicitly available in the Relevant Company Knowledge?

If either answer is NO, do not use that suggestion.

Example:

Relevant Company Knowledge:
"Data Analysts produce business reports, maintain dashboards,
and document metric definitions."

Good suggestions:
[
  "What reports does a Data Analyst produce?",
  "What dashboards does a Data Analyst maintain?",
  "What are metric definitions?"
]

Bad suggestion:
"What are the first 30-day expectations for a Data Analyst?"

The bad suggestion must NOT be generated because the context
does not contain information about 30-day expectations.
""",

        input=f"""
Employee Question:

{question}

Assistant Answer:

{answer}

Relevant Company Knowledge:

{context}
"""
    )

    result = response.output_text.strip()

    return result


# --------------------------------------------------
# Test only when this file is run directly
# --------------------------------------------------

if __name__ == "__main__":

    question = "How do I submit an expense claim?"

    answer = """
    Submit the expense claim within 10 days of incurring
    the expense. Attach an itemized receipt and select the
    correct cost center and project tag.
    """

    context = """
    Expense claims must be submitted within 10 days.
    Itemized receipts are required.
    Travel expenses require an itinerary.
    Manager approval is required above 2,000.
    Approved expenses are paid within 7–10 business days.
    """

    result = generate_followups(
        question,
        answer,
        context
    )

    print("\nFollow-up suggestions:\n")
    print(result)