# app.py
# ✅ Vidur Bot: Streamlit + LangChain + Groq Backend (Multi-language + PDF Upload & Safe Auto-Remove)

import streamlit as st
import os, urllib.parse, requests
import pypdf
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# =============================
# Load API Keys (from secrets/env)
# =============================
load_dotenv()
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY", os.getenv("YOUTUBE_API_KEY"))

# =============================
# Directory setup
# =============================
dev_data_dir = "data"        # permanent PDFs (developer-added)
user_data_dir = "user_data"  # temporary PDFs (user uploads)
persist_path = "chroma_db"

os.makedirs(dev_data_dir, exist_ok=True)
os.makedirs(user_data_dir, exist_ok=True)
os.makedirs(persist_path, exist_ok=True)

# =============================
# Initialize LLM
# =============================
llm = ChatGroq(
    temperature=0.7,
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

# =============================
# Setup Vector DB (RAG with dev PDFs only)
# =============================
def load_vector_db():
    if not os.path.exists(os.path.join(persist_path, "chroma.sqlite3")):
        loader = DirectoryLoader(dev_data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.split_documents(documents)
        embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma.from_documents(texts, embeddings, persist_directory=persist_path)
        vector_db.persist()
    else:
        embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma(persist_directory=persist_path, embedding_function=embeddings)
    return vector_db

vector_db = load_vector_db()
retriever = vector_db.as_retriever()

# =============================
# YouTube fetcher
# =============================
def fetch_youtube(query):
    if not YOUTUBE_API_KEY:
        return []
    query += " According to Indian law"
    url = (
        f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=5"
        f"&q={urllib.parse.quote(query)}&key={YOUTUBE_API_KEY}&type=video&regionCode=IN"
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

# Manage uploaded user file
pdf_text = ""
user_pdf_path = None

if uploaded_file is not None:
    user_pdf_path = os.path.join(user_data_dir, uploaded_file.name)
    if not os.path.exists(user_pdf_path):  # save only if new
        with open(user_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ PDF uploaded and text extracted!")

    try:
        reader = pypdf.PdfReader(user_pdf_path)
        pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        st.error(f"❌ Failed to read PDF: {e}")

else:
    # If user removes file → delete only from user_data folder
    for f in os.listdir(user_data_dir):
        try:
            os.remove(os.path.join(user_data_dir, f))
        except:
            pass
    pdf_text = ""

# =============================
# User input
# =============================
user_lang = st.selectbox("🌐 Select Response Language", ["English", "Hindi"])
user_question = st.text_area("❓ Ask your legal question")

if st.button("Ask Vidur Bot"):
    if not user_question and not pdf_text:
        st.warning("⚠️ Please enter a question or upload a PDF.")
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
