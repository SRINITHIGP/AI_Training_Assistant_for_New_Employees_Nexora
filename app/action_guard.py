from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def check_action_request(question):

    response = client.responses.create(
        model="gpt-5.6-luna",

        instructions="""
You are an action-safety classifier for a company AI Training Assistant.

Determine whether the employee is asking the assistant to
PERFORM or CHANGE something in a company system or process,
rather than simply asking for information.

Return exactly ONE value:

ACTION
INFORMATION

ACTION examples:

- Approve my leave request.
- Change my attendance record.
- Submit my expense claim for me.
- Cancel my travel request.
- Update my employee details.
- Give me access to the company system.

INFORMATION examples:

- How do I submit a leave request?
- What is the leave policy?
- How do I correct an attendance entry?
- What is the expense claim process?
- How do I request IT access?

Important:

- Classify based on the employee's request.
- The assistant does not have access to company systems
  or authorization to perform employee actions.
- Do not answer the question.
- Return ONLY ACTION or INFORMATION.
""",

        input=question
    )

    result = response.output_text.strip().upper()

    if result not in ["ACTION", "INFORMATION"]:
        return "ACTION"

    return result


# --------------------------------------------------
# Test only when this file is run directly
# --------------------------------------------------

if __name__ == "__main__":

    test_questions = [
        "Can you approve my leave request right now?",
        "Can you change my attendance record?",
        "Can you submit my expense claim for me?",
        "How do I submit a leave request?",
        "What is the expense claim process?",
        "How do I correct an attendance entry?"
    ]

    for question in test_questions:

        result = check_action_request(question)

        print(f"\nQuestion: {question}")
        print(f"Classification: {result}")