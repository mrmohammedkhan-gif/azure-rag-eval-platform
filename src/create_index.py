from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)

from config import *
from ingest import load_documents

credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)

index_client = SearchIndexClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    credential=credential
)

fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="source", type=SearchFieldDataType.String),
    SearchableField(name="content", type=SearchFieldDataType.String),
    SearchField(
        name="contentVector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=3072,
        vector_search_profile_name="vector-profile"
    ),
]

vector_search = VectorSearch(
    algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
    profiles=[
        VectorSearchProfile(
            name="vector-profile",
            algorithm_configuration_name="hnsw"
        )
    ]
)

index = SearchIndex(
    name=AZURE_SEARCH_INDEX,
    fields=fields,
    vector_search=vector_search
)

print("Creating/updating Azure AI Search index...")
index_client.create_or_update_index(index)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=credential
)

print("Loading documents and creating embeddings...")
docs = load_documents()

print("Uploading documents...")
search_client.upload_documents(docs)

print(f"Uploaded {len(docs)} chunks to {AZURE_SEARCH_INDEX}.")
