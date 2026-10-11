from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

app = FastAPI(title="chat bot",version="2.0")

llm = ChatOllama(model="qwen2.5:3b")

def get_session_history(session_id:str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat.db"
    )

prompt = ChatPromptTemplate.from_messages([
    ("system","you are a helpful and friendly AI assistant.Answer user question contextually."),
    MessagesPlaceholder(variable_name="history"),
    ("user","{user_query}")
])

base_chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="user_query",
    history_messages_key="history"
)

class ChatRequest(BaseModel):
    session_id:str
    user_query:str

@app.post("/chat")
def chat_with_persistent_memory(request:ChatRequest):
    config = {"configurable":{"session_id":request.session_id}}

    response = chain_with_history.invoke(
        {"user_input":request.user_query},
        config=config
    )

    return {
        "session_id":request.session_id,
        "reply":response.content
    }