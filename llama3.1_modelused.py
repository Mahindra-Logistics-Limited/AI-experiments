from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Set up the Ollama model
llm = ChatOllama(model="llama3.1")  # or any model like "llama2", "gemma"

# Add memory to preserve context
memory = ConversationBufferMemory()

# Set up a conversation chain
chatbot = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Interact with the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = chatbot.run(user_input)
    print("Response:", response)
