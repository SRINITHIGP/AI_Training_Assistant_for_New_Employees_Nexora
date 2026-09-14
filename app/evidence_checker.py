from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def check_evidence(question, context):

    response = client.responses.create(
        model="gpt-5.6-luna",

        instructions="""
You are an evidence checker for a company AI Training Assistant.

Your job is to determine whether the provided company knowledge
contains enough information to answer the employee's question.

Return exactly one of these values:

SUPPORTED
NOT_SUPPORTED

SUPPORTED:
The context contains enough relevant information to answer the
question accurately without guessing.

NOT_SUPPORTED:
The context does not contain enough information to answer the
question accurately.

Important rules:

- Use only the provided company knowledge.
- Do not use general knowledge.
- Do not assume information that is not in the context.
- If the context is only loosely related to the question,
  return NOT_SUPPORTED.
- Return ONLY SUPPORTED or NOT_SUPPORTED.
""",

        input=f"""
Company Knowledge:

{context}

Employee Question:

{question}
"""
    )

    result = response.output_text.strip().upper()

    if result not in ["SUPPORTED", "NOT_SUPPORTED"]:
        return "NOT_SUPPORTED"

    return result


# --------------------------------------------------
# Test only when this file is run directly
# --------------------------------------------------

if __name__ == "__main__":

    supported_question = "How do I submit an expense claim?"

    supported_context = """
    ## Submission Process

    1. Submit within 10 days of incurring the expense.
    2. Attach an itemized receipt.
    3. For travel attach itinerary.
    4. Choose the correct cost center and project tag.
    5. Manager approval is required for amounts above 2,000.
    """

    unsupported_question = "What products does the company offer?"

    unsupported_context = """
    ## Org Structure

    - People Operations
    - Engineering
    - Product
    - Data & Analytics
    - Customer Support
    - Finance & Procurement
    - IT & Security
    """

    print("Supported test:")
    print(check_evidence(supported_question, supported_context))

    print("\nUnsupported test:")
    print(check_evidence(unsupported_question, unsupported_context))