from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from langchain_community.embeddings import FastEmbedEmbeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
# -------------------------
# App + ENV setup
# -------------------------
app = Flask(__name__)
load_dotenv()  # locally uses .env; on Render it will use Render env vars

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # holds GROQ key (tutorial constraint)

# Fail fast with a clear error (prevents silent None issues)
missing = []
if not PINECONE_API_KEY:
    missing.append("PINECONE_API_KEY")
if not GROQ_API_KEY:
    missing.append("GROQ_API_KEY (your GROQ key value)")
if missing:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(missing)}. "
        f"Set them in Render → Service → Environment."
    )

# IMPORTANT: do NOT overwrite env vars with None.
# If they're already set, leave them as-is. (No need to set them again.)


# -------------------------
# Embeddings + Vector Store
# -------------------------
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot-bge"


docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# -------------------------
# LLM (Groq)
# -------------------------
chatModel = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=GROQ_API_KEY
)


# -------------------------
# Prompt
# -------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion:\n{input}")
    ]
)


# -------------------------
# RAG Chain (LangChain 1.x)
# -------------------------
# Make sure context is computed from the user's question.
rag_chain = (
    {
        "context": retriever,
        "input": RunnablePassthrough(),
    }
    | prompt
    | chatModel
    | StrOutputParser()
)


# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "")
    if not msg.strip():
        return "Please enter a question."

    response = rag_chain.invoke(msg)
    return response


# -------------------------
# Run (Render-safe)
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

