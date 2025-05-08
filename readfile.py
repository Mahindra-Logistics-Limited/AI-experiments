from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA

# Step 1: Load the file (e.g., PDF or TXT)
# loader = TextLoader("example.txt")  
loader = PyPDFLoader("Constitution_India_subset.pdf")
documents = loader.load()

# Step 2: Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Step 3: Create embeddings using Ollama
embedding = OllamaEmbeddings(model="nomic-embed-text")  # Make sure this is pulled via Ollama

# Step 4: Store embeddings in Chroma
vectordb = Chroma.from_documents(docs, embedding=embedding)

# Step 5: Set up LLM (Ollama) and QA chain
llm = ChatOllama(model="llama3.1")  # Or another pulled model
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever(), chain_type="stuff")

# Step 6: Ask questions
while True:
    query = input("Ask a question (or type 'exit'): ")
    
    if query.lower() in ["exit", "quit"]:
        break
    answer = qa.run(query)
    print("Answer:", answer)
