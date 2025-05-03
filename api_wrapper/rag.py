# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# from llama_index.llms.huggingface import HuggingFaceLLM
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.core.text_splitter import SentenceSplitter

import os
from dotenv import load_dotenv

load_dotenv()

model_name = "meta-llama/Llama-3.2-3B"

if os.getenv("deploy_mode")=="cluster":
    base_url = "https://adarshwshaw-NeoGurukul-llm.hf.space"
else:
    base_url = "http://localhost:11434"

def embed_text(metadata,text):
    embed_model = OllamaEmbedding(
        model_name="llama3.2",
        base_url=base_url,#"http://localhost:11434",
        ollama_additional_kwargs={"mirostat": 0},
    )

    uri = f"mongodb+srv://{os.getenv('su_user')}:{os.getenv('su_password')}@shaw.1iozj.mongodb.net/?retryWrites=true&w=majority&appName=Shaw"

    db = MongoClient(uri)
    atlas_vector_store = MongoDBAtlasVectorSearch(db, db_name='NeoGurukul_AI', \
            collection_name='transcriptions_rag', vetor_index_name='vector_index')

    text_splitter = SentenceSplitter(
        chunk_size=3072,       # tokens or words depending on config on model for 3.2=> 3072 , 3.2:1b=>2048
        chunk_overlap=30,
    )
    split_texts = text_splitter.split_text(text)
    docs=[]
    for chunk in split_texts:
        doc=Document(text=chunk,metadata=metadata)
        docs.append(doc)

    vector_store_context = StorageContext.from_defaults(vector_store = atlas_vector_store)
    print("context created");
    vector_store_index = VectorStoreIndex.from_documents(docs, storage_context=vector_store_context,show_progress=True, embed_model=embed_model)

    collection=db['NeoGurukul_AI']['transcriptions_rag']
    search_index_model = SearchIndexModel(
      definition={
        "fields": [
          {
            "type": "vector",
            "path": "embedding",
                    "numDimensions": 3072,
            "similarity": "cosine"
          },
          {
            "type": "filter",
            "path": "metadata.classId"
          }
        ]
      },
      name="vector_index",
      type="vectorSearch",
    )
    collection.create_search_index(model=search_index_model)
    db.close()
    print("done")
