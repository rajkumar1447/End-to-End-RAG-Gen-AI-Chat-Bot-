import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# Load documents from webpage
loader = WebBaseLoader("https://en.wikipedia.org/wiki/India")
docs = loader.load()

# Split docs into chunks for embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)

# Embeddings using local HuggingFace model (all-MiniLM-L6-v2)
hf_embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Build FAISS vectorstore from documents
vectorstore = FAISS.from_documents(split_docs, hf_embedder)

# Initialize Groq Chat model (ensure model name is correct and available)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="meta-llama/llama-guard-4-12b")

template = """
Use only the following context to answer the question.
If the answer is not contained in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:
"""

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": PromptTemplate.from_template(template)},
)

st.title("Groq Chat with Local Embeddings")

query = st.text_input("Ask a question")

if query:
    retriever = vectorstore.as_retriever()
    docs_for_query = retriever.get_relevant_documents(query)

    st.write(f"Documents retrieved for query '{query}':")
    for i, d in enumerate(docs_for_query):
        st.write(f"Document chunk {i+1}:")
        st.write(d.page_content[:500])  # limit output to first 500 chars
        st.write("---")

    # Run the retrieval QA chain properly with 'query' input key
    result = qa_chain.invoke({"query": query})
    st.write("Answer:")
    st.write(result['result'])
