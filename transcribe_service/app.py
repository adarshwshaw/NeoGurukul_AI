# Use a pipeline as a high-level helper
import pymongo
from transformers import pipeline
import gradio as gr
import torch
from pymongo.mongo_client import MongoClient
import json
import os
from dotenv import load_dotenv

load_dotenv()
langs = ["English", "Hindi"]

device= 0 if torch.cuda.is_available() else -1
print(device)

# dbcreds={}
# with open("env","r") as envfile:
#     dbcreds = json.load(envfile)

uri = f"mongodb+srv://{os.getenv('su_user')}:{os.getenv('su_password')}@shaw.1iozj.mongodb.net/?retryWrites=true&w=majority&appName=Shaw"
# Create a new client and connect to the server


def transcribe(lang,metadata, ifile):
    transcriber=None
    if lang == langs[0]:
        transcriber  = pipeline("automatic-speech-recognition", model="openai/whisper-base",chunk_length_s=30,device=device,generate_kwargs = {"language":"en","task": "translate"})
    else:
        transcriber  = pipeline("automatic-speech-recognition", model="vasista22/whisper-hindi-small",chunk_length_s=30,device=device)
        transcriber.model.config.forced_decoder_ids = transcriber.tokenizer.get_decoder_prompt_ids(language="hi", task="transcribe")

    # Path to your MP3 audio file

    client = MongoClient(uri )
    db = client.get_database("NeoGurukul_AI")
    collection = db.get_collection("transcriptions")
    print("connected to db")
    # Transcribe the MP3 audio file
    transcription = transcriber(ifile)
    obj = {"content": transcription["text"],"metadata":metadata}
    try:
        result = collection.insert_one(obj)
        
    except Exception as e:
        print(e)
        raise gr.Error(str(e))
    else:
        print("result: ",result)
    finally:
        client.close()

    return transcription["text"] 



demo = gr.Interface(
    fn=transcribe,
    inputs=[gr.Dropdown(langs, label="Language", info="Give the language of the audio"),\
            gr.JSON(label="metadata",show_label=True), \
            gr.Audio(sources=['upload'],type='filepath',label='input',show_label=True)],
    outputs=["text"],
    api_name="transcribe",
    live=False
)

demo.launch(show_api=True,debug=True)
