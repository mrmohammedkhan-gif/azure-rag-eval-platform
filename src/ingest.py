from pathlib import Path
from openai import AzureOpenAI
from config import *

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap

    return chunks

def get_embedding(text):
    response = client.embeddings.create(
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=text
    )
    return response.data[0].embedding

def load_documents(folder="data"):
    docs = []

    for file in Path(folder).glob("*.txt"):
        text = file.read_text(encoding="utf-8")

        for i, chunk in enumerate(chunk_text(text)):
            docs.append({
                "id": f"{file.stem}-{i}",
                "source": file.name,
                "content": chunk,
                "contentVector": get_embedding(chunk)
            })

    return docs

if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} chunks.")

    if docs:
        print(f"Embedding dimension: {len(docs[0]['contentVector'])}")
