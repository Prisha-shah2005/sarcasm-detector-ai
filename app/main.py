from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.predict import predict_sarcasm

app = FastAPI()

templates = Jinja2Templates(directory="frontend")


class TextInput(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(data: TextInput):
    return predict_sarcasm(data.text)