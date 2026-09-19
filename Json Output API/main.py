from fastapi import FastAPI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

app = FastAPI(title="Structured Json API",version="1.0")

llm = ChatOllama(model="qwen2.5:3b")

class codeReviewResponse(BaseModel):
    is_bug_free:bool = Field(description="true is the code has no bug or error, else false")
    bug_explanation:str = Field(description="explaint he error bug mistake in the code,as a short statement")
    optimized_code:str = Field(description="Corrected code with optimized handling of logic and perfect syntax")

structured_llm = llm.with_structured_output(codeReviewResponse)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert code reviewer. Analyze the user's code and return the response strictly matching the requested schema."),
    ("user", "Review this code and check for bugs:\n{user_input}")
])

chain = prompt | structured_llm

class queryRequest(BaseModel):
    user_input:str


@app.post("/ask")
def ask_ai(request: queryRequest):
    response = chain.invoke({"user_input":request.user_input})
    return response