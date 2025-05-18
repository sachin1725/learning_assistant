import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
import uuid

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini API
GEMINI_CONFIGURED_SUCCESSFULLY = False
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_CONFIGURED_SUCCESSFULLY = True
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}")
        st.stop()
else:
    st.error("GOOGLE_API_KEY not found in environment variables. Please set it in Colab Secrets or .env file.")
    st.stop()

# --- Helper Functions ---
def get_text_from_file(uploaded_file):
    text = ""
    if uploaded_file is not None:
        file_name = uploaded_file.name
        if file_name.endswith(".pdf"):
            try:
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
                return None
        elif file_name.endswith(".docx"):
            try:
                doc = DocxDocument(uploaded_file)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            except Exception as e:
                st.error(f"Error reading DOCX: {e}")
                return None
        elif file_name.endswith(".txt"):
            try:
                text = uploaded_file.read().decode("utf-8")
            except Exception as e:
                st.error(f"Error reading TXT: {e}")
                return None
        else:
            st.error("Unsupported file type. Please upload PDF, DOCX, or TXT.")
            return None
    return text

def get_text_chunks(text):
    if not text:
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource
def load_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vector_store(_text_chunks):
    if not _text_chunks:
        return None
    embeddings = load_embeddings_model()
    try:
        vector_store = FAISS.from_texts(texts=_text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_gemini_response(prompt, model_name="models/gemini-1.5-flash-latest"):
    if not GEMINI_CONFIGURED_SUCCESSFULLY:
        return "Error: Gemini API was not configured successfully at startup. Check API key."
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        if hasattr(response, 'parts') and response.parts:
            if response.parts[0].text:
                return response.parts[0].text
            else:
                if response.candidates and response.candidates[0].content.parts[0].text:
                    return response.candidates[0].content.parts[0].text
                else:
                    if response.candidates and response.candidates[0].finish_reason:
                        finish_reason = response.candidates[0].finish_reason
                        if finish_reason == "SAFETY":
                            return "Content generation blocked due to safety settings."
                        elif finish_reason == "RECITATION":
                            return "Content generation blocked due to recitation policy."
                        elif finish_reason == "OTHER":
                            return "Content generation stopped for an unspecified reason."
                    return "Received an empty response or content was blocked."
        elif hasattr(response, 'text'):
            return response.text
        else:
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                return f"Content generation blocked. Reason: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}"
            return "Error: No text found in Gemini response and no block reason provided."
    except Exception as e:
        if "is not found for API version" in str(e) or "is not supported for" in str(e):
            return f"Error: The model '{model_name}' might be incorrect or not supported. Original error: {e}"
        return f"Error communicating with Gemini: {e}"

def parse_flashcards_from_llm(raw_data):
    parsed_flashcards = []
    if raw_data is None:
        st.warning("LLM returned None for flashcard generation.")
        return parsed_flashcards
    if "Error:" in raw_data or not raw_data.strip():
        st.warning(f"LLM returned an error or empty response for flashcard generation: {raw_data}")
        return parsed_flashcards
    potential_cards = raw_data.strip().split("---")
    for card_block in potential_cards:
        card_block = card_block.strip()
        if "Front:" in card_block and "Back:" in card_block:
            try:
                front_content = card_block.split("Front:", 1)[1].split("Back:", 1)[0].strip()
                back_content = card_block.split("Back:", 1)[1].strip()
                if front_content and back_content:
                    parsed_flashcards.append({"front": front_content, "back": back_content, "id": str(uuid.uuid4())})
                else:
                    st.warning(f"Skipped a card block due to empty front or back: '{card_block[:50]}...'")
            except IndexError:
                st.warning(f"Could not fully parse a flashcard block: '{card_block[:50]}...'")
        elif card_block:
            st.warning(f"Skipped a malformed card block: '{card_block[:50]}...'")
    return parsed_flashcards

def generate_and_add_flashcards(source_text, context_description="selected text"):
    if not source_text:
        st.warning("No source text provided for flashcard generation.")
        return
    with st.spinner(f"Generating flashcard(s) from {context_description}..."):
        text_limit = 4000
        if len(source_text) > text_limit:
            source_for_cards = source_text[:text_limit]
            st.info(f"Using the first ~{text_limit} characters of the {context_description} for flashcard generation.")
        else:
            source_for_cards = source_text
        prompt = f"""
        You are an AI assistant that creates helpful study flashcards.
        Based on the following text, generate a maximum number of concise flashcards that would help test the understanding of the whole answer.
        Each flashcard MUST have a 'Front:' and a 'Back:'.
        'Front:' should contain a question, key term, or concept.
        'Back:' should contain the answer or definition.
        Ensure each flashcard is clearly separated by '---' (three hyphens on a new line).
        Example format:
        Front: What is the main topic of the text?
        Back: [The main topic extracted from the text]
        ---
        Front: Define [Key Term from text].
        Back: [Definition from text].
        Strictly follow this format. Do not add any other text before or after the flashcards.
        Text to generate flashcards from:
        ---
        {source_for_cards}
        ---
        """
        raw_flashcard_data = get_gemini_response(prompt)
        new_cards = parse_flashcards_from_llm(raw_flashcard_data)
        if new_cards:
            existing_cards = {(card['front'], card['back']) for card in st.session_state.flashcards}
            new_unique_cards = [
                card for card in new_cards
                if (card['front'], card['back']) not in existing_cards
            ]
            if new_unique_cards:
                st.session_state.flashcards.extend(new_unique_cards)
                st.success(f"Added {len(new_unique_cards)} new unique flashcard(s) from {context_description}!")
            else:
                st.info(f"No new unique flashcards generated from {context_description} (all were duplicates).")
        else:
            st.warning(f"No flashcards could be generated or parsed from the {context_description}.")

# --- Streamlit UI ---
st.set_page_config(page_title="AI Learning Assistant", layout="wide")

# Initial API Key and Configuration Check
if not GEMINI_CONFIGURED_SUCCESSFULLY:
    st.error("🔴 CRITICAL ERROR: Failed to configure Gemini API. Please verify your API key in Colab Secrets or .env file, ensure it's active, and restart the runtime.")
    st.stop()

st.title("📚 AI Learning Assistant")
st.caption("Upload a document, chat with the AI, and generate flashcards to reinforce your learning!")

# Initialize session state variables
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
if "generated_flashcard_ids" not in st.session_state:
    st.session_state.generated_flashcard_ids = set()

# --- Sidebar ---
with st.sidebar:
    st.header("📁 Document Hub")
    uploaded_file = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"], key="file_uploader")
    if uploaded_file:
        if st.button("Process Uploaded Document", key="process_doc_btn"):
            with st.spinner("Processing document..."):
                doc_text = get_text_from_file(uploaded_file)
                if doc_text:
                    st.session_state.document_text = doc_text
                    text_chunks = get_text_chunks(doc_text)
                    st.session_state.vector_store = get_vector_store(text_chunks) if text_chunks else None
                    st.session_state.chat_history = []
                    st.session_state.flashcards = []
                    st.session_state.generated_flashcard_ids = set()
                    if st.session_state.vector_store:
                        st.success("Document processed successfully!")
                    else:
                        st.error("Could process document but failed to create vector store.")
                else:
                    st.error("Could not extract text from the document.")
    if st.session_state.document_text:
        st.success("✅ Document Loaded & Processed")
    
    st.divider()
    st.header("✨ Flashcard Actions")
    if st.button("Generate from Full Document", key="fc_full_doc", disabled=not st.session_state.document_text):
        generate_and_add_flashcards(st.session_state.document_text, "full document")
    if st.session_state.chat_history:
        if st.button("Generate from Conversation History", key="fc_chat_hist"):
            # Filter out assistant responses that have been individually used for flashcard generation
            full_convo_text = "\n".join([
                f"{msg[0].capitalize()}: {msg[1]}" 
                for msg in st.session_state.chat_history 
                if msg[0] == "assistant" and msg[2] not in st.session_state.generated_flashcard_ids
            ])
            if full_convo_text:
                generate_and_add_flashcards(full_convo_text, "unprocessed conversation history")
            else:
                st.info("No unprocessed conversation history available for flashcard generation.")
    if st.session_state.flashcards:
        if st.button("Clear All Flashcards", key="fc_clear_all"):
            st.session_state.flashcards = []
            st.session_state.generated_flashcard_ids = set()
            st.info("All flashcards cleared.")

# --- Main Area: Chat and Flashcards ---
chat_col, flashcard_col = st.columns([0.6, 0.4])

with chat_col:
    st.subheader("💬 Chat with your AI Tutor")
    
    # Response style and explanation length buttons
    st.markdown("**Customize Response**")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    style_options = {
        "Teach me like I'm five": "🧸",
        "Professional": "💼",
        "Exam/Interview Prep": "📝"
    }
    length_options = {
        "Brief": "🔥",
        "Summary": "📄",
        "Detailed": "📚"
    }
    
    with col1:
        if st.button(style_options["Teach me like I'm five"], key="style_five", help="Simple, fun explanation like teaching a child"):
            st.session_state.response_style = "Teach me like I'm five"
            st.rerun()
    with col2:
        if st.button(style_options["Professional"], key="style_pro", help="Clear, formal explanation for academic or workplace"):
            st.session_state.response_style = "Professional"
            st.rerun()
    with col3:
        if st.button(style_options["Exam/Interview Prep"], key="style_exam", help="Textbook-style for exams or interviews"):
            st.session_state.response_style = "Exam/Interview Prep"
            st.rerun()
    with col4:
        if st.button(length_options["Brief"], key="length_brief", help="Short and essential information"):
            st.session_state.explanation_length = "Brief"
            st.rerun()
    with col5:
        if st.button(length_options["Summary"], key="length_summary", help="Balanced overview with key points"):
            st.session_state.explanation_length = "Summary"
            st.rerun()
    with col6:
        if st.button(length_options["Detailed"], key="length_detailed", help="In-depth with examples and reasoning"):
            st.session_state.explanation_length = "Detailed"
            st.rerun()
    
    # Display selected options above chat input
    st.markdown(f"**Selected:** {style_options[st.session_state.response_style]} {st.session_state.response_style} | {length_options[st.session_state.explanation_length]} {st.session_state.explanation_length}")
    
    for i, (role, content, msg_id) in enumerate(st.session_state.chat_history):
        if content == "Thinking...":
            with st.chat_message(role):
                st.markdown(content)
            continue
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant":
                if st.button("✨ Create Flashcard", key=f"fc_msg_{msg_id}", help="Generate a flashcard from this answer"):
                    context_text = ""
                    context_desc = ""
                    if i > 0 and st.session_state.chat_history[i-1][0] == "user":
                        user_q_for_context = st.session_state.chat_history[i-1][1]
                        context_text = f"Regarding the question: \"{user_q_for_context}\"\nThe AI answered: \"{content}\""
                        context_desc = "this Q&A pair"
                    else:
                        context_text = content
                        context_desc = "the AI's response"
                    generate_and_add_flashcards(context_text, context_desc)
                    st.session_state.generated_flashcard_ids.add(msg_id)

    user_question = st.chat_input("Ask about the document or a general question:")
    if user_question:
        msg_uuid = str(uuid.uuid4())
        st.session_state.chat_history.append(("user", user_question, msg_uuid))
        st.session_state.chat_history.append(("assistant", "Thinking...", str(uuid.uuid4())))
        st.rerun()

    # Process "Thinking..." state
    for i, (role, content, msg_id) in enumerate(st.session_state.chat_history):
        if role == "assistant" and content == "Thinking...":
            with st.spinner("Generating response..."):
                prompt_context = ""
                if i > 0 and st.session_state.chat_history[i-1][0] == "user":
                    user_question = st.session_state.chat_history[i-1][1]
                    if st.session_state.vector_store:
                        docs = st.session_state.vector_store.similarity_search(user_question, k=3)
                        if docs:
                            context_str = "\n".join([doc.page_content for doc in docs])
                            prompt_context = f"Context from document:\n{context_str}\n\n"
                        else:
                            prompt_context = "(No specific context found in the document for this question)\n"
                    
                    # Customize prompt based on response style and explanation length
                    style_instruction = ""
                    if st.session_state.response_style == "Teach me like I'm five":
                        style_instruction = "Explain in a simple, fun way as if teaching a five-year-old, using analogies and avoiding complex terms."
                    elif st.session_state.response_style == "Professional":
                        style_instruction = "Provide a clear, formal, and professional explanation suitable for a workplace or academic setting."
                    elif st.session_state.response_style == "Exam/Interview Prep":
                        style_instruction = "Answer in a precise, textbook-style manner, using standard terminology and formats suitable for exams or interviews."

                    length_instruction = ""
                    if st.session_state.explanation_length == "Brief":
                        length_instruction = "Keep the response short and to the point, focusing only on the essential information."
                    elif st.session_state.explanation_length == "Summary":
                        length_instruction = "Provide a balanced overview with key points and minimal elaboration."
                    elif st.session_state.explanation_length == "Detailed":
                        length_instruction = "Give a comprehensive explanation with examples, details, and thorough reasoning."

                    final_prompt = f"{prompt_context}{style_instruction}\n{length_instruction}\nQuestion: {user_question}\nAnswer:"
                    answer = get_gemini_response(final_prompt)
                    st.session_state.chat_history[i] = ("assistant", answer, msg_id)
                    st.rerun()
            break

with flashcard_col:
    st.subheader("🗂️ Your Flashcards")
    if not st.session_state.flashcards:
        st.info("No flashcards generated yet for this session.")
    else:
        st.markdown(f"You have **{len(st.session_state.flashcards)}** flashcard(s).")
        for i, card in enumerate(st.session_state.flashcards[:]):  # Use a copy to allow modification
            with st.expander(f"**Card {i+1}:** {card['front']}"):
                st.markdown(f"**Back:** {card['back']}")
                if st.button("🗑️ Delete", key=f"delete_fc_{card['id']}"):
                    st.session_state.flashcards = [c for c in st.session_state.flashcards if c['id'] != card['id']]
                    st.rerun()
        st.markdown("---")
