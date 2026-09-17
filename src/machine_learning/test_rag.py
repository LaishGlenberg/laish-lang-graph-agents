from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# 1. Embeddings — sentence-transformers running on GPU
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"},
)

# 2. Test documents
docs = [
    Document(page_content="Linux Mint is an Ubuntu-based Linux distribution."),
    Document(page_content="CUDA is NVIDIA's parallel computing platform for GPUs."),
    Document(page_content="RAG stands for Retrieval-Augmented Generation."),
    Document(page_content="Chroma is an open-source vector database."),
]

# 3. Store them in Chroma
vectorstore = Chroma.from_documents(
    docs,
    embeddings,
    collection_name="rag_documents",
    persist_directory="./chroma_db",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 4. LLM via Ollama
llm = ChatOllama(model="llama3.2:3b")

# 5. Ask a question
question = "What does RAG stand for?"
retrieved = retriever.invoke(question)
context = "\n".join(d.page_content for d in retrieved)

prompt = (
    f"Answer the question using only the context below.\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {question}\n"
    f"Answer:"
)
answer = llm.invoke(prompt)
print(answer.content)