from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def run_chatbot():
    llm = ChatOllama(model="qwen2.5:3b")

    prompt = ChatPromptTemplate.from_messages([
        ("system","you are a statefull ai chatbot"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{user_input}")
    ])

    chain = prompt | llm

    chat_history =[]

    while True:
        try:
            user_input = input("\n you:")

            if user_input.lower() in ["exit","quit","bye"]:
                print("\n AI: Exiting.. bye!")
                break

            if not user_input.strip():
                continue

            response = chain.invoke({
                "chat_history": chat_history,
                "user_input":user_input
            })

            print(f"\n AI:{response.content}")

            chat_history.append(("user",user_input))
            chat_history.append(("assistant",response.content))

        except KeyboardInterrupt:
            print("\n AI: Exiting.. bye!")
            break

if __name__ == "__main__":
    run_chatbot()