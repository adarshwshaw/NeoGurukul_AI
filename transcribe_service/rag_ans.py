

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import VectorStoreIndex, StorageContext, Document
import json
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

import os
from dotenv import load_dotenv

load_dotenv()

model_name = "meta-llama/Llama-3.2-3B"

# baseurl = "https://adarshwshaw-NeoGurukul-llm.hf.space"
baseurl = "http://localhost:11434"
embed_model= OllamaEmbedding(
    model_name="llama3.2",
    base_url=baseurl,
    ollama_additional_kwargs={"mirostat": 0},
)
print("loaded llm")


# dbcreds={}
# with open("env","r") as envfile:
#     dbcreds = json.load(envfile)
uri = f"mongodb+srv://{os.getenv('su_user')}:{os.getenv('su_password')}@shaw.1iozj.mongodb.net/?retryWrites=true&w=majority&appName=Shaw"

db = MongoClient(uri)
atlas_vector_store = MongoDBAtlasVectorSearch(db, db_name='NeoGurukul_AI', \
        collection_name='transcriptions_rag', vetor_index_name='vector_index')
vector_store_context = StorageContext.from_defaults(vector_store = atlas_vector_store)
print("context created");
vector_store_index = VectorStoreIndex.from_vector_store(atlas_vector_store, embed_model=embed_model,storage_context = vector_store_context)
Settings.llm = Ollama(model="llama3.2", base_url=baseurl, request_timeout=300.0)
metadata_filters = MetadataFilters(
   filters=[ExactMatchFilter(key="metadata.classId", value="default")]
)
# Instantiate Atlas Vector Search as a retriever filters=metadata_filters,
vector_store_retriever = VectorIndexRetriever(index=vector_store_index,  similarity_top_k=2,embed_model=embed_model)


# Instantiate Atlas Vector Search as a retriever

print("engine setup")
# Pass the retriever into the query engine
query_engine = RetrieverQueryEngine(retriever=vector_store_retriever)
# Prompt the LLM
print("engine setup")
response = query_engine.query('you are a teaching assitant: summerize the text')
# response = query_engine.query('what are the Severity level in Operational SLA?')
print(response)
