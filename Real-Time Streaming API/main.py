from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field

app = FastAPI(title="Streaming API",version="1.0")

llm = ChatOllama(model="qwen2.5:3b")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a witty and concise assistant."),
    ("user", "{user_input}")
])

chain = prompt | llm

class queryRequest(BaseModel):
    user_input: str

def generate_tokens(user_input: str):
    for chunk in chain.stream({"user_input": user_input}):
        yield chunk.content

@app.post("/stream")
async def stream_ai(request: queryRequest):

    return StreamingResponse(
        generate_tokens(request.user_input),
        media_type="text/plain"
    )