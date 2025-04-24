# Use a pipeline as a high-level helper
from transformers import pipeline
import gradio as gr
import torch

langs = ["English", "Hindi"]

device= 0 if torch.cuda.is_available() else -1
print(device)

def transcribe(lang, ifile):
    transcriber=None
    if lang == langs[0]:
        transcriber  = pipeline("automatic-speech-recognition", model="openai/whisper-base",chunk_length_s=30,device=device,generate_kwargs = {"language":"en","task": "translate"})
    else:
        transcriber  = pipeline("automatic-speech-recognition", model="vasista22/whisper-hindi-small",chunk_length_s=30,device=device)
        transcriber.model.config.forced_decoder_ids = transcriber.tokenizer.get_decoder_prompt_ids(language="hi", task="transcribe")

    # Path to your MP3 audio file

    # Transcribe the MP3 audio file
    transcription = transcriber(ifile)
    return transcription["text"] 



demo = gr.Interface(
    fn=transcribe,
    inputs=[gr.Dropdown(langs, label="Language", info="Give the language of the audio"),gr.Audio(sources=['upload'],type='filepath',label='input',show_label=True)],
    outputs=["text"],
    api_name="transcribe"
)

demo.launch(show_api=True,debug=True)
