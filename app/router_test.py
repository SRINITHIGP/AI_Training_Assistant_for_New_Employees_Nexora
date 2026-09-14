from dotenv import load_dotenv
from openai import OpenAI

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


# Load environment variables
load_dotenv()


# Create OpenAI client
client = OpenAI()


# --------------------------------------------------
# ChromaDB setup
# --------------------------------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma(
    collection_name="company_knowledge",
    embedding_function=embeddings,
    persist_directory="data/chroma_db",
)


# --------------------------------------------------
# Query classifier
# --------------------------------------------------

def classify_question(question):

    response = client.responses.create(
        model="gpt-5.6-luna",

        instructions="""
You are a query classifier for an AI Training Assistant.

Classify the employee's question into exactly ONE category:

GENERAL
ROLE
ADMIN
OTHER

GENERAL:
Questions about the company as a whole, including company overview,
products, services, teams, values, or general company information.

ROLE:
Questions about a specific employee role, job responsibilities,
team responsibilities, tools, skills, workflows, or role-specific training.

ADMIN:
Questions about onboarding, HR, leave, expenses, travel, IT access,
timesheets, attendance, security, compliance, or company policies.

OTHER:
Questions unrelated to the company, employee onboarding,
employee training, or company processes.

Return ONLY the category name.
""",

        input=question
    )

    return response.output_text.strip().upper()


# --------------------------------------------------
# Route the question
# --------------------------------------------------

def route_question(question):

    category = classify_question(question)

    print(f"\nQuestion: {question}")
    print(f"Route: {category}")

    # OTHER questions should not be searched
    if category == "OTHER":

        print("Action: Politely refuse the question.")
        return

    # Search only the category selected by the classifier
    results = vector_store.similarity_search(
        question,
        k=3,
        filter={"category": category}
    )

    print(f"Action: Search {category} knowledge base.")

    for i, result in enumerate(results):

        print(f"\n--- Retrieved Result {i + 1} ---")
        print(result.page_content[:300])
        print(f"Category: {result.metadata.get('category')}")
        print(f"Source: {result.metadata.get('source')}")


# --------------------------------------------------
# Test questions
# --------------------------------------------------

test_questions = [
    "What products does the company offer?",
    "What tools does a Data Analyst use?",
    "How do I submit an expense claim?",
    "When is the next Marvel movie coming out?"
]


for question in test_questions:

    route_question(question)