from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from dotenv import load_dotenv

from openai import OpenAI

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.evidence_checker import check_evidence
from app.fallback_test import generate_fallback
from app.personal_data_guard import check_sensitive_request
from app.action_guard import check_action_request
from app.ambiguity_guard import check_ambiguity
from app.followup_suggestions import generate_followups

# --------------------------------------------------
# Conversation memory
# --------------------------------------------------

conversation_history = []



def format_conversation_history():

    if not conversation_history:
        return "No previous conversation."

    formatted_history = []

    for message in conversation_history:

        role = message["role"]

        if role == "user":
            speaker = "Employee"
        else:
            speaker = "Assistant"

        formatted_history.append(
            f"{speaker}: {message['content']}"
        )

    return "\n\n".join(formatted_history)

def contextualize_question(question):

    history = format_conversation_history()

    if history == "No previous conversation.":
        return question

    response = client.responses.create(

        model="gpt-5.6-luna",

        instructions="""
You are a query contextualization assistant for a company
AI Training Assistant.

Rewrite the employee's current question so that it can be
understood independently using the previous conversation.

Rules:

- Preserve the employee's original meaning.
- Use previous conversation only to resolve missing context.
- Resolve references such as "they", "it", "that", "this",
  "those", "how many", and similar contextual references.
- Do not answer the question.
- Do not add information that is not present in the conversation.
- Keep the rewritten question concise.
- If the current question is already clear, return it unchanged.
- Return ONLY the rewritten question.
""",

        input=f"""
Previous Conversation:

{history}

Current Employee Question:

{question}
"""
    )

    return response.output_text.strip()


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Initialize application
# --------------------------------------------------

app = FastAPI()


@app.get("/new-chat")
def new_chat():
    conversation_history.clear()

    return {
        "message": "New chat started."
    }


client = OpenAI()


# --------------------------------------------------
# Initialize embeddings
# --------------------------------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# --------------------------------------------------
# Connect to ChromaDB
# --------------------------------------------------

vector_store = Chroma(
    collection_name="company_knowledge",
    embedding_function=embeddings,
    persist_directory="data/chroma_db",
)


# --------------------------------------------------
# Frontend
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory="frontend"),
    name="static"
)


@app.get("/")
def home():

    return FileResponse(
        "frontend/index.html"
    )


# --------------------------------------------------
# Question classifier
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
Do not provide an explanation.
""",

        input=question
    )

    result = response.output_text.strip().upper()

    if result not in [
        "GENERAL",
        "ROLE",
        "ADMIN",
        "OTHER"
    ]:
        return "OTHER"

    return result


# --------------------------------------------------
# Main chatbot endpoint
# --------------------------------------------------

@app.get("/ask")
def ask(question: str):

    # Store the employee's question
    conversation_history.append({
        "role": "user",
        "content": question
    })


    # --------------------------------------------------
    # 1. Sensitive / personal information check
    # --------------------------------------------------

    sensitive_result = check_sensitive_request(
        question
    )

    if sensitive_result == "SENSITIVE":

        def sensitive_response():

            yield (
                "I cannot provide personal or sensitive "
                "employee information. I can help with "
                "general company policies, onboarding, "
                "training, and processes."
            )

        return StreamingResponse(
            sensitive_response(),
            media_type="text/plain"
        )


    # --------------------------------------------------
    # 2. Unauthorized action check
    # --------------------------------------------------

    action_result = check_action_request(
        question
    )

    if action_result == "ACTION":

        def action_response():

            yield (
                "I cannot perform or change actions in "
                "company systems. I can explain the relevant "
                "process and guide you on what to do."
            )

        return StreamingResponse(
            action_response(),
            media_type="text/plain"
        )


    # --------------------------------------------------
    # 3. Handle simple greetings
    # --------------------------------------------------

    greeting_words = [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening"
    ]

    if question.strip().lower() in greeting_words:

        def greeting_response():
            yield (
                "Hello! 👋 Welcome to Nexora. "
                "I'm the Nexora Training Assistant, your AI onboarding "
                "and training companion. How can I help you today?"
            )

        return StreamingResponse(
            greeting_response(),
            media_type="text/plain"
        )


    # --------------------------------------------------
    # 4. Classify the question
    # --------------------------------------------------

    category = classify_question(
        question
    )

    # --------------------------------------------------
    # 5. Out-of-scope handling
    # --------------------------------------------------

    if category == "OTHER":

        def refusal_response():

            yield (
                "I cannot answer that. I am a company chatbot "
                "designed to help with company-related questions, "
                "employee onboarding, and training."
            )

        return StreamingResponse(
            refusal_response(),
            media_type="text/plain"
        )


    # --------------------------------------------------
    # 6. Ambiguity check
    # --------------------------------------------------

    ambiguity_result = check_ambiguity(
        question,
        conversation_history
    )

    if ambiguity_result == "AMBIGUOUS":

        def ambiguity_response():

            yield (
                "Could you provide a little more detail "
                "about what you need help with? For example, "
                "you can mention the specific process, role, "
                "policy, or tool you're asking about."
            )

        return StreamingResponse(
            ambiguity_response(),
            media_type="text/plain"
        )


    # --------------------------------------------------
    # 7. Contextualize question for retrieval
    # --------------------------------------------------

    retrieval_question = contextualize_question(
        question
    )


    # --------------------------------------------------
    # 8. Retrieve relevant company knowledge
    # --------------------------------------------------

    results = vector_store.similarity_search(

        retrieval_question,

        k=3,

        filter={
            "category": category
        }
        
    )


    # --------------------------------------------------
    # 9. Build context
    # --------------------------------------------------

    context = "\n\n".join(

        result.page_content

        for result in results

    )


    # --------------------------------------------------
    # 10. Evidence check
    # --------------------------------------------------

    evidence_result = check_evidence(

        question,

        context

    )


    # --------------------------------------------------
    # 11. Missing information fallback
    # --------------------------------------------------

    if evidence_result == "NOT_SUPPORTED":

        fallback = generate_fallback(
            question
        )

        def fallback_response():

            yield fallback

        return StreamingResponse(
            fallback_response(),
            media_type="text/plain"
        )


    # --------------------------------------------------
    # 12. Generate grounded answer
    # --------------------------------------------------

    def generate_response():

        stream = client.responses.create(

            model="gpt-5.6-luna",

            instructions="""
You are an AI Training Assistant for a company.

Answer the employee's question using ONLY the
company information provided in the context.

Important rules:

1. Use only the retrieved company evidence.
2. Never invent company facts, policies, procedures,
   products, roles, or processes.
3. Never fill missing information with general knowledge.
4. Do not claim access to personal employee information.
5. Do not claim to perform actions in company systems.
6. If the evidence does not support a specific detail,
   do not state that detail as fact.
7. Keep the answer clear, concise, and employee-friendly.
8. Explain processes using simple steps when appropriate.

The response must be grounded in the provided company context.
""",

            input=f"""
Previous Conversation:

{format_conversation_history()}

Company Knowledge:

{context}

Current Employee Question:

{question}
""",

            stream=True
        )


        full_response = ""

        for event in stream:

            if event.type == "response.output_text.delta":

                full_response += event.delta

                yield event.delta


        # Store the assistant's complete response
        conversation_history.append({
            "role": "assistant",
            "content": full_response
        })


        # Generate contextual follow-up suggestions
        followups = generate_followups(
            question,
            full_response,
            context
        )


        # Send follow-up suggestions to the frontend
        yield "\n\n__FOLLOWUPS__\n"
        yield followups


    return StreamingResponse(

        generate_response(),

        media_type="text/plain"

    )