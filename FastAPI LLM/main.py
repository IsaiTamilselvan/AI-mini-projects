from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(title="AI Coding Assistant API",version="1.0")

llm = ChatOllama(model="qwen2.5:3b")

prompt = ChatPromptTemplate.from_messages([
    ("system","you are a helpful ai coding assistant"),
    ("user","{user_prompt}")
])

chain = prompt | llm

class queryRequest(BaseModel):
    user_prompt: str 

@app.post("/ai/ask")
def ask_ai(request: queryRequest):
    response = chain.invoke({"user_prompt":request.user_prompt})
    return{
        "response":response.content
    }
