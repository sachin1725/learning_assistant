import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
import uuid

# -----------------------------
# ENV SETUP
# -----------------------------

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    st.error("GOOGLE_API_KEY not found in environment variables.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------


def get_text_from_file(uploaded_file):
    text = ""

    try:
        if uploaded_file.name.endswith(".pdf"):
            pdf_reader = PdfReader(uploaded_file)

            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

        elif uploaded_file.name.endswith(".docx"):
            doc = DocxDocument(uploaded_file)

            for para in doc.paragraphs:
                text += para.text + "\n"

        elif uploaded_file.name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")

        else:
            st.error("Unsupported file type.")
            return None

    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

    return text


def get_text_chunks(text):

    MAX_DOC_CHARS = 50000
    text = text[:MAX_DOC_CHARS]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    return splitter.split_text(text)


@st.cache_resource
def load_embeddings_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def get_vector_store(text_chunks):

    if not text_chunks:
        return None

    embeddings = load_embeddings_model()

    try:
        return FAISS.from_texts(text_chunks, embeddings)
    except Exception as e:
        st.error(f"Vector store error: {e}")
        return None


def get_gemini_response(prompt, model_name="gemini-2.5-flash"):

    try:
        model = genai.GenerativeModel(model_name)

        response = model.generate_content(prompt)

        if hasattr(response, "text"):
            return response.text

        if response.candidates:
            return response.candidates[0].content.parts[0].text

        return "No response generated."

    except Exception as e:
        return f"Error: {e}"


# -----------------------------
# FLASHCARD LOGIC
# -----------------------------


def parse_flashcards_from_llm(raw):

    cards = []

    if not raw:
        return cards

    blocks = raw.strip().split("---")

    for block in blocks:

        if "Front:" in block and "Back:" in block:

            try:

                front = block.split("Front:", 1)[1].split("Back:", 1)[0].strip()
                back = block.split("Back:", 1)[1].strip()

                if front and back:

                    cards.append({
                        "front": front,
                        "back": back,
                        "id": str(uuid.uuid4())
                    })

            except:
                pass

    return cards


def generate_and_add_flashcards(text, context="text"):

    if not text:
        return

    with st.spinner("Generating flashcards..."):

        text = text[:4000]

        prompt = f"""
Create up to 10 study flashcards from the following text.

Rules:
Each card must follow this format.

Front: question or concept
Back: answer

Separate cards using ---
Do not add extra text.

Text:
{text}
"""

        raw = get_gemini_response(prompt)

        cards = parse_flashcards_from_llm(raw)

        existing = {(c["front"], c["back"]) for c in st.session_state.flashcards}

        new_cards = [
            c for c in cards
            if (c["front"], c["back"]) not in existing
        ]

        if new_cards:
            st.session_state.flashcards.extend(new_cards)
            st.success(f"Added {len(new_cards)} flashcards.")
        else:
            st.info("No new flashcards generated.")


# -----------------------------
# STREAMLIT CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI Learning Assistant",
    layout="wide"
)

st.title("📚 AI Learning Assistant")
st.caption("Upload a document, chat with AI, and generate flashcards.")

# -----------------------------
# SESSION STATE
# -----------------------------

if "document_text" not in st.session_state:
    st.session_state.document_text = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "flashcards" not in st.session_state:
    st.session_state.flashcards = []

if "response_style" not in st.session_state:
    st.session_state.response_style = "Professional"

if "explanation_length" not in st.session_state:
    st.session_state.explanation_length = "Summary"

# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:

    st.header("📁 Document")

    uploaded = st.file_uploader(
        "Upload PDF/DOCX/TXT",
        type=["pdf", "docx", "txt"]
    )

    if uploaded and st.button("Process Document"):

        with st.spinner("Processing..."):

            text = get_text_from_file(uploaded)

            if text:

                st.session_state.document_text = text

                chunks = get_text_chunks(text)

                st.session_state.vector_store = get_vector_store(chunks)

                st.session_state.chat_history = []
                st.session_state.flashcards = []

                st.success("Document processed")

    st.divider()

    st.header("Flashcards")

    if st.button("Generate from Document"):

        generate_and_add_flashcards(
            st.session_state.document_text,
            "document"
        )

    if st.button("Clear Flashcards"):

        st.session_state.flashcards = []

# -----------------------------
# MAIN LAYOUT
# -----------------------------

chat_col, flash_col = st.columns([0.6, 0.4])

# -----------------------------
# CHAT
# -----------------------------

with chat_col:

    st.subheader("💬 Chat")

    for role, content, msg_id in st.session_state.chat_history:

        with st.chat_message(role):
            st.markdown(content)

    question = st.chat_input("Ask something")

    if question:

        user_id = str(uuid.uuid4())

        st.session_state.chat_history.append(
            ("user", question, user_id)
        )

        context = ""

        if st.session_state.vector_store:

            docs = st.session_state.vector_store.similarity_search(
                question,
                k=3
            )

            context = "\n".join(d.page_content for d in docs)

        prompt = f"""
Answer the question using the document context if available.

Context:
{context}

Question:
{question}
"""

        answer = get_gemini_response(prompt)

        st.session_state.chat_history.append(
            ("assistant", answer, str(uuid.uuid4()))
        )

        st.rerun()

# -----------------------------
# FLASHCARDS
# -----------------------------

with flash_col:

    st.subheader("🗂 Flashcards")

    if not st.session_state.flashcards:
        st.info("No flashcards yet.")

    for card in st.session_state.flashcards:

        with st.expander(card["front"]):

            st.write(card["back"])

            if st.button("Delete", key=card["id"]):

                st.session_state.flashcards = [
                    c for c in st.session_state.flashcards
                    if c["id"] != card["id"]
                ]

                st.rerun()

    st.markdown("---")