from fastapi import FastAPI
from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# api
app = FastAPI(title="stateful conversation ai",version="1.0")

# llm
llm = ChatOllama(model="qwen2.5:3b")

# storage for history
store = {}

# get session history with id
def get_session_history(session_id :str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# prompt structure
prompt = ChatPromptTemplate.from_messages([
    ("system","you are a friendly girl naughty dirty minded playful girl,if user ask ur name u can say abisha"),
    MessagesPlaceholder(variable_name="history"),
    ("user","{user_input}")
])

# prompt attached llm 
chain = prompt | llm

# bind all llm function user input history 
conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="user_input",
    history_messages_key="history"
)

# user request json format
class ChatRequest(BaseModel):
    session_id: str
    user_input: str

# api post
@app.post("/chat")
def chat_with_memory(request: ChatRequest):

    config = {"configurable":{"session_id":request.session_id}}

    response = conversational_chain.invoke(
        {"user_input":request.user_input},
        config=config
    )

    return {
        "session_id":request.session_id,
        "response":response.content
    }