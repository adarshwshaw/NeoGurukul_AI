from fastapi import FastAPI,Request,HTTPException,BackgroundTasks
import requests
import json
import os
from dotenv import load_dotenv

from rag import embed_text
from rag_ans import get_llm_response

app = FastAPI()

load_dotenv()

print("deployment mode", os.getenv("deploy_mode"))

if os.getenv("deploy_mode")=="cluster":
    transcribe_uri = "https://adarshwshaw-transcribe-sample.hf.space"
else:
    transcribe_uri = "http://localhost:7860"


stream_outs={}
# bt=BackgroundTasks()


@app.get("/health")
async def health():
    return "Running"

def parse_gradio_res(res):
    res = res.split('\n',1)
    event = res[0].split(':')[-1]
    data = res[1].split(':',1)[-1]
    parsed = {res[0].split(':')[0].strip() : event.strip(), 'data':json.loads(data)}
    return parsed


def stream_request_handler(metadata):
    global stream_outs
    job_id=metadata['classId']
    url= "{}/gradio_api/queue/data?session_hash={}".format(transcribe_uri,job_id)
    headers = {
        'Content-Type': 'application/json',
        'Authorization':f'Bearer {os.getenv("hf_tok")}'
    }
    response = requests.request("GET", url,headers=headers,stream=True)
    flag=False
    event=None
    data=None
    for line in response.iter_lines():
        if line:
            decoded = line.decode("utf-8").strip()
            stream_outs[job_id]=json.loads(decoded.split(":",1)[-1])
            print(stream_outs)
            if 'complete' in stream_outs[job_id]['msg']:
                break;
    if stream_outs[job_id]['success']:
        stream_outs[job_id]['msg']='heartbeat'
        embed_text(metadata,stream_outs[job_id]['output']['data'][0])
        stream_outs[job_id]['msg']='complete'

@app.post("/transcribe")
async def transcribe(req:Request, bt:BackgroundTasks):
    data= await req.json()
    file_url=data.get("file_url")
    lang=data.get("lang")
    metadata=data.get("metadata")
    if not file_url or not lang:
        raise HTTPException(status_code=400, detail="Missing file_url or lang")
    url = transcribe_uri+"/gradio_api/call/transcribe"

    payload = json.dumps({
        "session_hash":metadata['classId'],
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
    bt.add_task(stream_request_handler,metadata=metadata)
    return {"event_id":evid['event_id'],"session_hash":metadata['classId']}



@app.post("/transcribe_result")
async def transcribe_result(req:Request):
    global stream_outs
    print(stream_outs)
    data= await req.json()
    event_id=data.get("event_id")
    
    if event_id not in stream_outs.keys():
        raise HTTPException(status_code=500,detail={"event":"error","error_msg":"cannot get status"})


    res = stream_outs[event_id]
    if 'complete' in res['msg']:
        if res['success']:
            res['msg']='complete'
        else:
            res['msg']='error'
            res['output']['data']=res['output']['error']
            del res['output']['error']
    #TODO: write to db and remove from dict
        del stream_outs[event_id]
    return res

@app.post("/summary")
async def summary(req:Request):
    data = await req.json()
    metadata = data.get('metadata')
    query = "Give me the summary?"
    response=None
    try:
        response = get_llm_response(metadata,query)
        response = response.response
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_msg":str(e)})
    return {"response":response}
