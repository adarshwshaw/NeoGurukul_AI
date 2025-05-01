from fastapi import FastAPI,Request,HTTPException
import requests
import json
import os
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

print("deployment mode", os.getenv("deploy_mode"))

if os.getenv("deploy_mode")=="cluster":
    transcribe_uri = "https://adarshwshaw-transcribe-sample.hf.space"
else:
    transcribe_uri = "http://localhost:7860"



@app.get("/health")
async def health():
    return "Running"

def parse_gradio_res(res):
    res = res.split('\n',1)
    event = res[0].split(':')[-1]
    data = res[1].split(':',1)[-1]
    parsed = {res[0].split(':')[0].strip() : event.strip(), 'data':json.loads(data)}
    return parsed



@app.post("/transcribe")
async def transcribe(req:Request):
    data= await req.json()
    file_url=data.get("file_url")
    lang=data.get("lang")
    metadata=data.get("metadata")
    if not file_url or not lang:
        raise HTTPException(status_code=400, detail="Missing file_url or lang")
    url = transcribe_uri+"/gradio_api/call/transcribe"

    payload = json.dumps({
      "data": [
        lang,
        metadata,
        {
          "path": file_url,
          "meta": {
            "_type": "gradio.FileData"
          }
        }
      ]
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization':f'Bearer {os.getenv("hf_tok")}'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code == 503:
        raise HTTPException(status_code=503,detail={"error_msg":response.text})
    evid = json.loads(response.text)
    return {"event_id":evid['event_id']}


@app.post("/transcribe_result")
async def transcribe_result(req:Request):
    data= await req.json()
    event_id=data.get("event_id")
    res_url= "{}/gradio_api/call/transcribe/{}".format(transcribe_uri,event_id)

    headers = {
        'Content-Type': 'application/json',
        'Authorization':f'Bearer {os.getenv("hf_tok")}'
    }
    response = requests.request("GET", res_url,headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=500,detail={"event":"error","error_msg":"cannot get status"})

    res = parse_gradio_res(response.text)
    return res


