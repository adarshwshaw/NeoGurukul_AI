

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import VectorStoreIndex, StorageContext, Document
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

if os.getenv("deploy_mode")=="cluster":
    base_url = "https://adarshwshaw-NeoGurukul-llm.hf.space"
else:
    base_url = "http://localhost:11434"

# baseurl = "https://adarshwshaw-NeoGurukul-llm.hf.space"

embed_model= OllamaEmbedding(
    model_name="llama3.2",
    base_url=base_url,
    ollama_additional_kwargs={"mirostat": 0},
)

Settings.llm = Ollama(model="llama3.2", base_url=base_url, request_timeout=600.0)
def get_llm_response(metadata,query):
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
    metadata_filters = MetadataFilters(
       filters=[ExactMatchFilter(key="metadata.classId", value=metadata['classId'])]
    )
    # Instantiate Atlas Vector Search as a retriever filters=metadata_filters,
    vector_store_retriever = VectorIndexRetriever(index=vector_store_index, embed_model=embed_model,filters=metadata_filters, similarity_top_k=2)
    # vector_store_retriever = VectorIndexRetriever(index=vector_store_index,  similarity_top_k=2,embed_model=embed_model)


    # Instantiate Atlas Vector Search as a retriever

    print("engine setup")
    # Pass the retriever into the query engine
    query_engine = RetrieverQueryEngine(retriever=vector_store_retriever)
    # Prompt the LLM
    print("engine setup")
    response = query_engine.query(f'you are a teaching assitant: {query}')
    db.close()
    # response = query_engine.query('what are the Severity level in Operational SLA?')
    print(response)
    return response

if __name__=='__main__':
    print("here")
    get_llm_response({"classId":"lec1"},"summarize the text")
