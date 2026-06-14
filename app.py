import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

load_dotenv()

# -------------------------------- LLM ----------------------------------------------------

llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    max_new_tokens=300,
    temperature=0.2,
)

llm = ChatHuggingFace(llm=llm_endpoint)

# -------------------------------- UI ----------------------------------------------------

st.title("NEWS RESEARCH BOT")
st.sidebar.title("NEWS ARTICLES URL")

urls = []
for i in range(5):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

urls = [u for u in urls if u.strip()]
process_url_clicked = st.sidebar.button("PROCESS URLS")

main_placeholder = st.empty()

# -------------------------------- STORAGE ------------------------------------------------

FAISS_DIR = "faiss_store"

# -------------------------------- EMBEDDINGS --------------------------------------------

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embed = get_embeddings()

# ------------------------------ PROMPTS FOR MAP AND REDUCE ------------------------------

map_template = """
You are a retrieval assistant.

Question:
{question}

Chunk:
{context}

Task:
- Extract any information that could help answer the question.
- Keep relevant code snippets.
- Keep API parameters and examples.
- Return empty only if the chunk is completely unrelated.

Relevant information:
"""

reduce_template = """
You are a strict synthesis engine.

Rules:
- Use ONLY the provided filtered chunks.
- Do NOT complete partial code unless fully present in context.
- Do NOT guess missing parts.
- If the answer is incomplete, say:
"I cannot find complete information in the provided documents."

Question:
{question}

Filtered chunks:
{context}

FINAL ANSWER (clear and complete):
"""

map_prompt = ChatPromptTemplate.from_template(map_template)
map_chain = map_prompt | llm | StrOutputParser()

reduce_prompt = ChatPromptTemplate.from_template(reduce_template)
reduce_chain = reduce_prompt | llm | StrOutputParser()

# -------------------------------- INGEST -------------------------------------------------

if process_url_clicked:
    if not urls:
        st.warning("Please enter at least one URL.")
        st.stop()

    main_placeholder.text("DATA loading... started...")

    loader = UnstructuredURLLoader(urls=urls)

    try:
        data = loader.load()
    except Exception as e:
        st.error(f"Failed: {e}")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600,
        chunk_overlap=200,
    )

    main_placeholder.text("SPLITTING DATA... started...")
    docs = text_splitter.split_documents(data)

    vec_idx = FAISS.from_documents(docs, embed)
    main_placeholder.text("CREATING EMBEDDING VECTOR... started...")

    time.sleep(2)

    vec_idx.save_local(FAISS_DIR)
    main_placeholder.text("DONE. Vector store saved.")

# -------------------------------- QUERY --------------------------------------------------

def ask(ques: str, retriever, map_chain, reduce_chain):
    docs = retriever.invoke(ques)

    # MAP step: one LLM call per chunk
    map_inputs = [
        {
            "question": ques,
            "context": doc.page_content[:1200],
        }
        for doc in docs
    ]

    filtered_chunks = map_chain.batch(map_inputs)

    # Clean empty outputs
    filtered_chunks = [
        chunk.strip()
        for chunk in filtered_chunks
        if chunk and chunk.strip()
    ]

    if not filtered_chunks:
        return {
            "answer": "I cannot find that in the provided documents.",
            "sources": [],
        }

    # REDUCE step: combine map outputs into one context
    merged_context = "\n\n".join(
        f"FC{i + 1}: {chunk[:500]}"
        for i, chunk in enumerate(filtered_chunks)
    )

    final_answer = reduce_chain.invoke(
        {
            "question": ques,
            "context": merged_context,
        }
    )

    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "UNKNOWN SOURCE")
        if source and source not in sources:
            sources.append(source)

    return {
        "answer": final_answer,
        "sources": sources,
    }

# -------------------------------- MAIN --------------------------------------------------

query = main_placeholder.text_input("question : ")

if query:
    if not os.path.exists(FAISS_DIR):
        st.warning("Please process URLs first.")
        st.stop()

    vec_idx = FAISS.load_local(
        FAISS_DIR,
        embed,
        allow_dangerous_deserialization=True,
    )

    retriever = vec_idx.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20},
    )

    result = ask(query, retriever, map_chain, reduce_chain)

    st.markdown(result["answer"])

    st.subheader("Sources")
    for src in result["sources"]:
        st.markdown(f"- [{src}]({src})")