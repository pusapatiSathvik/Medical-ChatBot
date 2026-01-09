from flask import Flask, render_template, jsonify, request
import os
from dotenv import load_dotenv

# Use the updated helper functions you created
from src.helper import download_embeddings
# Use the modern integration packages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
# Use langchain_classic for the legacy chain structures to avoid the ModelProfile error
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

# 1. Load Environment Variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") # Note: using your Gemini key name

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# 2. Initialize Embeddings (HuggingFace 384-dim)
embeddings = download_embeddings()

# 3. Connect to Existing Pinecone Index
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# 4. Setup Retriever (Top 3 most relevant chunks)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 5. Initialize Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# 6. Define the Medical Prompt
system_prompt = (
    "You are a Medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 7. Create the RAG Chain
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# --- Flask Routes ---

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    # Matches the 'msg' key from your HTML form/JS
    msg = request.form["msg"]
    print(f"User Query: {msg}")
    
    # Run the query through the RAG chain
    result = rag_chain.invoke({"input": msg})
    
    # Return the AI's answer
    print(f"AI Answer: {result['answer']}")
    return str(result["answer"])

if __name__ == '__main__':
    # Start the server on port 8080 as requested
    app.run(host="0.0.0.0", port=8080, debug=True)