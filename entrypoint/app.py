import streamlit as st
import sys
import os
import uuid
from dotenv import load_dotenv # NEW: This reads your .env file

load_dotenv()

# Tell Python to look in the root directory to find the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vectorstore import VectorStore
from src.chatbot import Chatbot

# --- Utility Functions ---
def save_uploaded_file(uploaded_file):
    file_path = os.path.join("data", "raw", "uploaded_document.pdf")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if os.path.exists(file_path):
        os.remove(file_path)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

def main():
    st.set_page_config(page_title="Document QA Bot 🤖", layout="wide")
    st.title("Document QA Bot 🤖 with Memory and Citations")

    # --- Session State Initialization ---
    # These MUST be at the very top of main() before any UI elements load
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = "session_" + str(uuid.uuid4())[:8]
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello! Please upload a PDF and enter your API keys to begin."}]
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None
    if "chatbot" not in st.session_state:
        st.session_state["chatbot"] = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state["uploaded_file_name"] = None

    # --- Sidebar for Setup ---
  
    with st.sidebar:
        st.header("Setup 🔑")
        # 1. Initialize session state variables for the keys so they aren't empty on first load
        if "cohere_key" not in st.session_state:
            st.session_state["cohere_key"] = ""
        if "pinecone_key" not in st.session_state:
            st.session_state["pinecone_key"] = ""

        # 2. Access Code Logic
        st.sidebar.header("Configuration")
        access_code = st.sidebar.text_input("Enter access code (optional)", type="password")

        if access_code:
            if access_code == os.getenv("INTERVIEW_PASSWORD"):
                st.sidebar.success("Loaded successfully!")
                # This magically auto-fills the text inputs below because they share the same 'key'
                st.session_state["cohere_key"] = os.getenv("COHERE_API_KEY", "")
                st.session_state["pinecone_key"] = os.getenv("PINECONE_API_KEY", "")
            else:
                st.sidebar.error("Incorrect Access Code. Please enter the correct code, or provide your own keys.")

        st.sidebar.markdown(
            "Don't have keys? [Get Cohere Key](https://dashboard.cohere.com/api-keys) | [Get Pinecone Key](https://app.pinecone.io/)"
        )

        # 3. The actual API Key Inputs (Notice how they link directly to the session_state keys)
        cohere_api_key = st.sidebar.text_input("Cohere API Key", type="password", key="cohere_key")
        pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password", key="pinecone_key")

        # 4. Document Upload & Processing
        uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf", key="pdf_uploader")

        if st.sidebar.button("Process Document and Initialize Chat"):
            if uploaded_file and cohere_api_key and pinecone_api_key:
                st.session_state["uploaded_file_name"] = uploaded_file.name
                # Note: Ensure save_uploaded_file handles saving to data/raw/uploaded_document.pdf
                save_uploaded_file(uploaded_file) 
                
                with st.spinner(f"Processing {uploaded_file.name} with Hybrid Search..."):
                    try:
                        file_path = os.path.join("data", "raw", "uploaded_document.pdf")
                        
                        # Initialize the VectorStore
                        vectorstore = VectorStore(
                            pdf_path=file_path, 
                            cohere_api_key=cohere_api_key, 
                            pinecone_api_key=pinecone_api_key,
                            namespace=st.session_state.get("session_id", "default-session")
                        )
                        st.session_state["vectorstore"] = vectorstore

                        # Initialize the Chatbot
                        chatbot = Chatbot(vectorstore, cohere_api_key)
                        st.session_state["chatbot"] = chatbot
                        
                        # Setup initial chat message
                        st.session_state["messages"] = [
                            {"role": "assistant", "content": f"Document **{uploaded_file.name}** processed successfully! Ask your first question."}
                        ]
                        st.sidebar.success("Initialization complete! Chat is ready.")
                    
                    except Exception as e:
                        st.sidebar.error(f"Error during initialization: {e}")
                        st.session_state["vectorstore"] = None
                        st.session_state["chatbot"] = None
                        st.session_state["messages"] = [{"role": "assistant", "content": f"ERROR: Could not initialize. Details: {e}"}]
            else:
                st.sidebar.error("Please provide both API keys and upload a PDF.")

        # --- 4. New Cleanup Section with Popup & Bright Button ---
        st.markdown("---")
        
        if "confirm_delete" not in st.session_state:
            st.session_state["confirm_delete"] = False
            
        # type="primary" makes the button bright red/blue depending on your theme
        if st.button("🗑️ End Chat & Clear Data", type="primary", use_container_width=True):
            st.session_state["confirm_delete"] = True
            st.rerun()
            
        # The confirmation popup
        if st.session_state["confirm_delete"]:
            st.warning("⚠️ **Are you sure?** This deletes your document and chat history.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Delete"):
                    if st.session_state.get("vectorstore"):
                        with st.spinner("Deleting database vectors..."):
                            st.session_state["vectorstore"].delete_namespace()
                    st.session_state.clear()
                    st.rerun()
            with col2:
                if st.button("Cancel"):
                    st.session_state["confirm_delete"] = False
                    st.rerun()

    # --- Main Chat Interface ---
    
    # FIX 3: Display all existing messages AND their sources from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # If this message has saved sources, render them in an expander
            if "sources" in message and message["sources"]:
                with st.expander("🔍 Sources (Reranked Chunks)"):
                    for i, doc in enumerate(message["sources"]):
                        st.markdown(f"**Source [{i+1}]:**")
                        st.caption(doc['text'])

    if user_query := st.chat_input("Ask a question based on the document..."):
        if st.session_state["chatbot"] is None:
            st.warning("Please initialize the chat by uploading a PDF and entering keys first.")
            st.stop()

        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Prepare history for Cohere
        chat_history_for_api = []
        for m in st.session_state.messages:
            role = m["role"]
            if role == "assistant":
                chat_history_for_api.append({"role": "Chatbot", "message": m["content"]})
            elif role == "user":
                chat_history_for_api.append({"role": "User", "message": m["content"]})
        
        with st.spinner("Thinking..."):
            response_stream, retrieved_docs = st.session_state["chatbot"].respond(
                user_query, 
                chat_history=chat_history_for_api 
            )
            
            full_response_text = ""

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                # Stream the text
                for event in response_stream:
                    if event.event_type == "text-generation":
                        full_response_text += event.text
                        message_placeholder.markdown(full_response_text + "▌")
                
                message_placeholder.markdown(full_response_text)
                
                # FIX 1 & 2: Hide sources if it's a fallback answer, and use an expander
                fallback_message = "I cannot answer this based on the provided document."
                final_sources_to_save = None

                # Only show the sources expander if the bot DID NOT use the fallback message
                if retrieved_docs and fallback_message not in full_response_text:
                    final_sources_to_save = retrieved_docs
                    with st.expander("🔍 Sources (Reranked Chunks)"):
                        for i, doc in enumerate(retrieved_docs):
                            st.markdown(f"**Source [{i+1}]:**")
                            st.caption(doc['text'])
                        
        # Save the bot's response AND the sources to the session state so they persist
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response_text,
            "sources": final_sources_to_save # This ensures the expander stays visible in history!
        })

if __name__ == "__main__":
    main()