from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def check_sensitive_request(question):

    response = client.responses.create(
        model="gpt-5.6-luna",

        instructions="""
You are a safety classifier for a company AI Training Assistant.

Determine whether the employee's question requests
personal, private, or sensitive employee INFORMATION
that the assistant should not provide.

Return exactly ONE value:

SENSITIVE
SAFE

SENSITIVE examples:

- What is my exact salary?
- What is my personal tax breakup?
- Show me my payroll details.
- What is my performance review?
- What did my manager write about me?
- Show me another employee's private information.
- Can you show my private employee record?

SAFE examples:

- What is the company's leave policy?
- How many days of leave are employees entitled to?
- How do I submit a leave request?
- Can you approve my leave request?
- How do I correct my attendance?
- Can you change my attendance record?
- How do I submit an expense claim?
- What is the company's expense policy?
- What does a Data Analyst do?
- How do I request IT access?

IMPORTANT:

- Classify only whether the question requests sensitive
  personal INFORMATION.
- Requests to perform an action on an employee's behalf,
  such as approving leave, changing attendance, or submitting
  an expense, are NOT SENSITIVE.
- Those action requests should be classified as SAFE here
  so that the separate action-safety guard can handle them.
- General company policies and processes are SAFE.
- Do not answer the question.
- Return ONLY SENSITIVE or SAFE.
""",

        input=question
    )

    result = response.output_text.strip().upper()

    if result not in ["SENSITIVE", "SAFE"]:
        return "SENSITIVE"

    return result


# --------------------------------------------------
# Test only when this file is run directly
# --------------------------------------------------

if __name__ == "__main__":

    test_questions = [
        "What is my exact salary and tax breakup?",
        "Can I see my personal payroll information?",
        "What is the company's leave policy?",
        "What does a Data Analyst do?",
        "Tell me about the company's expense policy.",
        "Can you approve my leave request right now?",
        "Can you change my attendance record?",
        "Can you submit my expense claim for me?"
    ]

    for question in test_questions:

        result = check_sensitive_request(question)

        print(f"\nQuestion: {question}")
        print(f"Classification: {result}")