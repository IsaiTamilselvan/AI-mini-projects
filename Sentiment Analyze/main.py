from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(title="Sentiment Analyze",version="1.0")

llm = ChatOllama(model="qwen2.5:3b",temperature=0)

class ProductReviewAnalysis(BaseModel):
    sentiment: str = Field(description="The sentiment of the review: Must be 'Positive' or 'Negative'")
    category: str = Field(description="The category of the issue/praise: e.g., Quality, Delivery, Pricing, or Customer Support")
    key_takeaway: str = Field(description="A short 1-line summary of what the customer is saying")

structured_llm = llm.with_structured_output(ProductReviewAnalysis)

prompt = ChatPromptTemplate([
    ("system","you are an expert e-commerce product review analyst.Analyze the user's review carefully."),
    ("user","{review_text}")
])

chain = prompt | structured_llm

class ReviewRequest(BaseModel):
    review_text: str

@app.post("/analyze-review")
def analyze_product_review(request:ReviewRequest):
    result = chain.invoke({"review_text":request.review_text})

    return {
        "status":"success",
        "analysis":result
    }