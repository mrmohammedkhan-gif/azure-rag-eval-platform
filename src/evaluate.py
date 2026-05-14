from openai import AzureOpenAI
from config import *

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

def evaluate_grounding(question, docs, answer):
    context = "\n\n".join([d["content"] for d in docs])

    prompt = f"""
You are evaluating whether an AI answer is grounded in the provided context.

Return JSON only:
{{
  "grounding": "GROUNDED" or "NOT_GROUNDED",
  "reason": "short explanation"
}}

Context:
{context}

Question:
{question}

Answer:
{answer}
"""

    response = client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

def evaluate_retrieval(question, docs):
    context = "\n\n".join([d["content"] for d in docs])

    prompt = f"""
You are evaluating whether the retrieved context is relevant to the question.

Return JSON only:
{{
  "retrieval_quality": "RELEVANT" or "WEAK",
  "reason": "short explanation"
}}

Question:
{question}

Retrieved Context:
{context}
"""

    response = client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content
