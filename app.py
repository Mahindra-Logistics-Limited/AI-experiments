import os
import hashlib
from flask import Flask, request, render_template
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA

UPLOAD_FOLDER = 'uploads'
VECTOR_FOLDER = 'vectorstores'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize once to avoid reloading models
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
    file_hash = get_file_hash(file_path)
    vector_dir = os.path.join(VECTOR_FOLDER, file_hash)

    print("vector_dir",vector_dir)

    if os.path.exists(vector_dir):
        # Load existing vector DB
        print("Loading existing vector DB")
        vectordb = Chroma( persist_directory=vector_dir,
                           embedding_function=embedding)
    else:
        # Load document and create new vector DB
        print("Load document and create new vector DB")
        documents = load_file(file_path)
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        vectordb = Chroma.from_documents(
            docs,
            embedding=embedding, 
            persist_directory=vector_dir
        )
        vectordb.persist()

    # QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=vectordb.as_retriever(), 
        chain_type="stuff"
    )

    return qa_chain.run(question)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():

    print('inside ask')
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

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
