from fastapi import FastAPI
from pydantic import BaseModel
from .chatbot_engine import generate_response

app = FastAPI()


class ChatRequest(BaseModel):
    user_input: str


@app.get("/")
def home():
    return {"message": "Psychsviser API is running"}


@app.post("/chat")
def chat(request: ChatRequest):

    response = generate_response(request.user_input)

    return {"response": response}