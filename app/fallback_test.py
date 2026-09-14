from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def generate_fallback(question):

    response = client.responses.create(
        model="gpt-5.6-luna",

        instructions="""
You are an AI Training Assistant for a company.

The company knowledge base does not contain enough
information to answer the employee's question.

Generate a short, polite response explaining that
the information is not available in the current
company knowledge base.

Important rules:

- Do not answer the original question using general knowledge.
- Do not invent company information.
- Do not pretend to have access to internal systems.
- If appropriate, suggest checking the relevant internal
  resource or contacting the appropriate company team.
- Keep the response professional and helpful.
""",

        input=question
    )

    return response.output_text.strip()


# --------------------------------------------------
# Test only when this file is run directly
# --------------------------------------------------

if __name__ == "__main__":

    question = "What products does the company offer?"

    fallback = generate_fallback(question)

    print("\nFallback response:\n")
    print(fallback)