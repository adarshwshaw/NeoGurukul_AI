from fastapi import FastAPI,Request,HTTPException
import requests
import json

app = FastAPI()

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
    if not file_url or not lang:
        raise HTTPException(status_code=400, detail="Missing file_url or lang")
    url = "http://localhost:7860/gradio_api/call/transcribe"

    payload = json.dumps({
      "data": [
        lang,
        {
          "path": file_url,
          "meta": {
            "_type": "gradio.FileData"
          }
        }
      ]
    })
    headers = {
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    evid = json.loads(response.text)
    return {"event_id":evid['event_id']}


@app.post("/transcribe_result")
async def transcribe_result(req:Request):
    data= await req.json()
    event_id=data.get("event_id")
    res_url= "http://localhost:7860/gradio_api/call/transcribe/{}".format(event_id)
    response = requests.request("GET", res_url)
    if response.status_code != 200:
        raise HTTPException(status_code=500,details={"event":"error","error_msg":"cannot get status"})

    res = parse_gradio_res(response.text)
    return res


