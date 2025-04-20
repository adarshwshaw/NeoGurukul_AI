# Use a pipeline as a high-level helper
from transformers import pipeline
import gradio as gr

def transcribe(ifile):
    transcriber  = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

    # Path to your MP3 audio file

    # Transcribe the MP3 audio file
    transcription = transcriber(ifile)
    return transcription["text"] 


audio_file = "sample.mp3"  # Replace with the actual path to your MP3 file

demo = gr.Interface(
    fn=transcribe,
    inputs=[gr.Audio(sources=['upload'],type='filepath',label='input',show_label=True)],
    outputs=["text"],
    api_name="transcribe"
)

demo.launch(show_api=True,debug=True)
