# app.py
# ✅ Vidur Bot: Streamlit + LangChain + Groq Backend (Multi-language + PDF Upload with FAISS)

import streamlit as st
import os, urllib.parse, requests, shutil
import pypdf

# LangChain imports
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# =============================
# Load API Keys securely
# =============================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]

# =============================
# Directory setup
# =============================
dev_data_dir = "data"        # permanent PDFs (developer-added)
user_data_dir = "user_data"  # temporary PDFs (user uploads)
faiss_index_path = "faiss_index"

os.makedirs(dev_data_dir, exist_ok=True)
os.makedirs(user_data_dir, exist_ok=True)

# =============================
# Initialize LLM
# =============================
llm = ChatGroq(
    temperature=0.7,
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

# =============================
# Setup Embeddings
# =============================
embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# =============================
# Build / Load FAISS Vector DB
# =============================
def build_faiss_index():
    """Rebuild FAISS index from dev + user PDFs"""
    all_docs = []
    for folder in [dev_data_dir, user_data_dir]:
        loader = DirectoryLoader(folder, glob="*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        all_docs.extend(docs)

    if not all_docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(all_docs)

    vector_db = FAISS.from_documents(texts, embeddings)
    vector_db.save_local(faiss_index_path)
    return vector_db

def load_faiss_index():
    if os.path.exists(faiss_index_path):
        return FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        return build_faiss_index()

vector_db = load_faiss_index()
retriever = vector_db.as_retriever() if vector_db else None

# =============================
# YouTube fetcher
# =============================
def fetch_youtube(query):
    query += " According to Indian law"
    url = (
        f"https://www.googleapis.com/youtube/v3/search?"
        f"part=snippet&maxResults=5&q={urllib.parse.quote(query)}"
        f"&key={YOUTUBE_API_KEY}&type=video&regionCode=IN"
    )
    try:
        response = requests.get(url)
        items = response.json().get("items", [])
        return [
            f"[{item['snippet']['title']}](https://www.youtube.com/watch?v={item['id']['videoId']})"
            for item in items
        ]
    except:
        return []

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Vidur Bot", page_icon="⚖️", layout="wide")
st.title("⚖️ Vidur Bot – Indian Legal AI Assistant")
st.write("Ask legal questions and get responses based on Indian law.")

# Upload PDF (user only)
uploaded_file = st.file_uploader("📄 Upload a legal PDF", type=["pdf"])

pdf_text = ""
if uploaded_file:
    user_pdf_path = os.path.join(user_data_dir, uploaded_file.name)

    # Save uploaded file
    with open(user_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text for context
    try:
        reader = pypdf.PdfReader(user_pdf_path)
        pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        st.success("✅ PDF uploaded and text extracted!")
    except Exception as e:
        st.error(f"❌ Failed to read PDF: {e}")

    # Rebuild FAISS with new PDF
    vector_db = build_faiss_index()
    retriever = vector_db.as_retriever() if vector_db else None

# Delete uploaded PDFs if user clears uploader
elif not uploaded_file:
    if os.listdir(user_data_dir):  # only clear user PDFs
        shutil.rmtree(user_data_dir)
        os.makedirs(user_data_dir, exist_ok=True)
        vector_db = build_faiss_index()
        retriever = vector_db.as_retriever() if vector_db else None

# =============================
# User input
# =============================
user_lang = st.selectbox("🌐 Select Response Language", ["English", "Hindi"])
user_question = st.text_area("❓ Ask your legal question")

if st.button("Ask Vidur Bot"):
    if not user_question and not pdf_text:
        st.warning("⚠️ Please enter a question or upload a PDF.")
    elif not retriever:
        st.error("❌ No knowledge base available. Please upload or add PDFs.")
    else:
        # Prepare input
        final_input = user_question
        if pdf_text:
            final_input = (
                "The user uploaded the following legal document:\n\n"
                + pdf_text
                + "\n\nPlease explain the content and provide guidance."
            )

        # Prompt template with enforced language
        prompt_template = PromptTemplate(
            template=f"""You are Vidur Bot, a legal expert in Indian law. 
Answer queries with references to Indian Constitution Articles, BNS, and remedies where relevant. 
⚠️ IMPORTANT: Your response must be strictly in **{user_lang}**.

Context: {{context}}

User: {{question}}
Vidur Bot:""",
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template}
        )

        with st.spinner("⚖️ Vidur Bot is thinking..."):
            try:
                response = qa_chain.run(final_input) or "Sorry, I couldn't generate a response."
                st.subheader("🧑‍⚖️ Vidur Bot's Response")
                st.write(response)

                youtube_links = fetch_youtube(user_question)
                if youtube_links:
                    st.subheader("▶️ Related YouTube Resources")
                    for link in youtube_links:
                        st.markdown(link)
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
