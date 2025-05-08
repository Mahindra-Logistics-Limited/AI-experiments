from langchain_community.chat_models import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

from sqlalchemy import create_engine

# Step 1: Set up SQLite connection (or use your DB connection string)
engine = create_engine("sqlite:///my_database.db")
db = SQLDatabase(engine=engine)

# Step 2: Connect to Ollama LLM
llm = ChatOllama(model="mistral")  # or llama3, llama2, etc.

# Step 3: Create the SQL chain
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# Step 4: User input loop
while True:
    user_input = input("Ask your question about the database (or type 'exit'): ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = db_chain.run(user_input)
    print("Result:", response)
