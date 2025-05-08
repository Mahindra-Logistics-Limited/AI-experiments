import os
import hashlib
from flask import Flask, request, render_template
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "pinecone-llama3-1"

if index_name not in pc.list_indexes().names():
    print("Creating new Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )
else:
    print("Using existing Pinecone index.")

# Flask app setup
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Embedding & LLM
embedding = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.1")

def load_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return PyPDFLoader(file_path).load()
    elif ext == '.txt':
        return TextLoader(file_path).load()
    else:
        raise ValueError("Unsupported file format: only .txt and .pdf are supported")

def get_file_hash(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()

def process_file_and_answer(file_path, question):
    print("Processing file and question...")

    file_hash = get_file_hash(file_path)
    namespace = file_hash[:20]  # use first 20 chars as namespace

    documents = load_file(file_path)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # Upload to Pinecone (if not already uploaded)
    print(f"Using namespace: {namespace}")
    vectordb = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embedding,
        index_name=index_name,
        namespace=namespace
    )

    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    return qa_chain.run(question)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    print("Received /ask request")
    file = request.files.get("file")
    question = request.form.get("question")

    if not file or not question:
        return render_template("index.html", answer="Please upload a file and enter a question.")

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        answer = process_file_and_answer(file_path, question)
    except Exception as e:
        answer = f"Error: {str(e)}"
        print(answer)

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
