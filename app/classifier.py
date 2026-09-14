from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables
load_dotenv()


# Create OpenAI client
client = OpenAI()


def classify_question(question):

    response = client.responses.create(
        model="gpt-5.6-luna",

        instructions="""
You are a query classifier for an AI Training Assistant.

Classify the employee's question into exactly ONE of these categories:

GENERAL
ROLE
ADMIN
OTHER

GENERAL:
Questions about information that applies broadly to the company as a whole.

This includes:
- company overview
- products and services
- company values
- company teams and structure
- company-wide working hours
- company-wide collaboration windows
- work model
- common internal company tools
- general communication practices

ROLE:
Questions specifically about a particular employee role, job, or team.

This includes:
- role responsibilities
- role-specific tools
- role-specific skills
- role-specific workflows
- role-specific training
- expectations for a particular role

ADMIN:
Questions about employee processes, policies, and procedures.

This includes:
- onboarding procedures
- leave and PTO
- expenses
- travel reimbursement
- IT access requests
- timesheets
- attendance
- security and compliance
- HR administrative processes
- company policies

OTHER:
Questions that are unrelated to company information, employee onboarding,
employee training, or company processes.

IMPORTANT DISTINCTIONS:

- Company-wide information → GENERAL
- Information specific to a named role → ROLE
- Employee processes or policies → ADMIN
- Unrelated or unsupported requests → OTHER

Examples:

"What are the standard company work hours?"
→ GENERAL

"What is the core collaboration window?"
→ GENERAL

"What tools are considered common internal tools during onboarding?"
→ GENERAL

"What tools do employees commonly use?"
→ GENERAL

"What tools does a Data Analyst use?"
→ ROLE

"What are the responsibilities of a Product Manager?"
→ ROLE

"How do I request PTO?"
→ ADMIN

"How do I submit an expense claim?"
→ ADMIN

"What is the turnaround time for an access request?"
→ ADMIN

"What is my exact salary breakup and tax deductions?"
→ OTHER

"Can you approve my leave request right now?"
→ OTHER

"Write a personal performance improvement plan for me."
→ OTHER

When a question could match more than one category, use this priority:

1. Role-specific information → ROLE
2. Employee process or policy → ADMIN
3. Company-wide information → GENERAL
4. Unrelated or unsupported requests → OTHER

Return ONLY the category name.
Do not provide an explanation.
""",

        input=question
    )

    return response.output_text.strip().upper()


# --------------------------------------------------
# Test questions
# --------------------------------------------------

# test_questions = [
#     "Tell me about the company.",
#     "What does our company do?",

#     "What skills are required for a Data Analyst?",
#     "What does a Product Manager do?",

#     "How many days of leave can I take?",
#     "How do I get access to company tools?",
#     "What is the travel reimbursement process?",
#     "Where do I submit my timesheet?",

#     "Who won the football match yesterday?",
#     "Write me a poem about cats."
# ]


# for question in test_questions:

#     category = classify_question(question)

#     print(f"\nQuestion: {question}")
#     print(f"Category: {category}")