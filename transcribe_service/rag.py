# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# from llama_index.llms.huggingface import HuggingFaceLLM
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex, StorageContext, Document
from llama_index.core.text_splitter import SentenceSplitter
import json

import os
from dotenv import load_dotenv

load_dotenv()

model_name = "meta-llama/Llama-3.2-3B"

# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,token=tok)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, 
#     torch_dtype=torch.float16,
#     device_map="auto",
#     token=tok
#     #trust_remote_code=True
# )
#
# llm = HuggingFaceLLM(
#     model=model,
#     tokenizer=tokenizer,
#     query_wrapper_prompt="Answer the following: {query_str}",
#     generate_kwargs={"temperature": 0.2},
#     max_new_tokens=1024
# )
print("loaded llm")
# embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
embed_model = OllamaEmbedding(
    model_name="llama3.2",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)


# dbcreds={}
# with open("env","r") as envfile:
#     dbcreds = json.load(envfile)
uri = f"mongodb+srv://{os.getenv('su_user')}:{os.getenv('su_password')}@shaw.1iozj.mongodb.net/?retryWrites=true&w=majority&appName=Shaw"

db = MongoClient(uri)
atlas_vector_store = MongoDBAtlasVectorSearch(db, db_name='NeoGurukul_AI', \
        collection_name='transcriptions_rag', vetor_index_name='vector_index')

text=None
with open("transcribe_service/sample.txt","r") as f:
    text=f.read()

text_splitter = SentenceSplitter(
    chunk_size=3072,       # tokens or words depending on config on model for 3.2=> 3072 , 3.2:1b=>2048
    chunk_overlap=30,
)
split_texts = text_splitter.split_text(text)
docs=[]
for chunk in split_texts:
    doc=Document(text=chunk,metadata={"classId":"default"})
    docs.append(doc)
# doc = SimpleDirectoryReader(input_files=['transcribe_service/sample.txt']).load_data()Vh
# print(doc)

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
print("done")
