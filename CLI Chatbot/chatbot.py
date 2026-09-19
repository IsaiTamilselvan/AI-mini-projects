from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

def run_cli_chatbot():
    print("---Initializing qwen model---")

    llm = ChatOllama(model ="qwen2.5:3b")

    prompt = ChatPromptTemplate.from_messages([
        ("system","you are a coding assistent to learn coding, in smart and simple way."),
        ("user","{user_input}")
    ])

    chain = prompt | llm

    print("---chatbot ready---")

    while True:
        try:
            user_input = input("you:")

            if user_input.lower() in ["bye","quit"]:
                print("AI: Exiting... bye!")
                break

            if not user_input.strip():
                continue

            response = chain.invoke({"user_input":user_input})
            print(f"AI:{response.content}")

        except KeyboardInterrupt:
            print("\n AI: Exiting... bye!")
            break

if __name__ == "__main__":
    run_cli_chatbot()