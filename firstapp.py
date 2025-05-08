from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.1"
                ,temperature=0, 
                max_tokens=1000,
                stream=True)
response = llm.invoke("India is a country in South Asia.")
print(response)