from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def check_ambiguity(question, conversation_history=None):

    if conversation_history is None:
        conversation_history = []

    response = client.responses.create(
        model="gpt-5.6-luna",

        instructions="""
You are an ambiguity classifier for a company AI Training Assistant.

Determine whether the employee's question is clear enough
to answer without asking for clarification.

You may use the previous conversation to understand
references such as:

- they
- them
- it
- that
- this
- those
- the above
- the previous one

Return exactly ONE value:

CLEAR
AMBIGUOUS

CLEAR:
The question has enough context, either from the current
question or from the previous conversation, to understand
what the employee is asking.

AMBIGUOUS:
The question is too vague and the previous conversation
does not provide enough context to determine what the
employee wants.

Examples:

Previous conversation:
"What does a Data Analyst do?"

Current question:
"What tools do they use?"

Classification:
CLEAR

Previous conversation:
"What is the company's leave policy?"

Current question:
"How many days can I take?"

Classification:
CLEAR

Current question:
"What tools do we use?"

Classification:
AMBIGUOUS

Current question:
"What is the process?"

Classification:
AMBIGUOUS

Current question:
"Who should I contact?"

Classification:
AMBIGUOUS

Important:

- Use previous conversation when it provides the missing context.
- Do not use general knowledge.
- Do not answer the question.
- Return ONLY CLEAR or AMBIGUOUS.
""",

        input=f"""
Previous Conversation:

{conversation_history}

Current Employee Question:

{question}
"""
    )

    result = response.output_text.strip().upper()

    if result not in ["CLEAR", "AMBIGUOUS"]:
        return "AMBIGUOUS"

    return result


# --------------------------------------------------
# Test only when this file is run directly
# --------------------------------------------------

if __name__ == "__main__":

    test_history = [
        {
            "role": "user",
            "content": "What does a Data Analyst do?"
        },
        {
            "role": "assistant",
            "content": "A Data Analyst creates dashboards and reports."
        }
    ]

    test_questions = [
        "What tools do they use?",
        "What skills are required?",
        "What tools do we use?",
        "What is the process?"
    ]

    for question in test_questions:

        result = check_ambiguity(
            question,
            test_history
        )

        print(f"\nQuestion: {question}")
        print(f"Classification: {result}")