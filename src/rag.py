from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from config import *

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
)

def get_embedding(text):
    response = client.embeddings.create(
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=text
    )
    return response.data[0].embedding

def retrieve(question, top_k=3):
    query_vector = get_embedding(question)

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="contentVector"
    )

    results = search_client.search(
        search_text=question,
        vector_queries=[vector_query],
        select=["source", "content"],
        top=top_k
    )

    return [{"source": r["source"], "content": r["content"]} for r in results]

def generate_answer(question, docs):
    context = "\n\n".join(
        [f"Source: {d['source']}\n{d['content']}" for d in docs]
    )

    prompt = f"""
Answer using ONLY the context below.
If the answer is not in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a grounded enterprise AI assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return {
        "answer": response.choices[0].message.content,
        "tokens": response.usage.total_tokens if response.usage else None
    }

def run_rag(question):
    docs = retrieve(question)
    result = generate_answer(question, docs)

    return {
        "question": question,
        "retrieved_docs": docs,
        "answer": result["answer"],
        "tokens": result["tokens"]
    }

if __name__ == "__main__":
    print("Running RAG test...")

    result = run_rag("What does the company require for AI systems?")

    print("\nANSWER:")
    print(result["answer"])

    print("\nTOKENS:")
    print(result["tokens"])

    print("\nSOURCES:")
    for doc in result["retrieved_docs"]:
        print("-", doc["source"])
