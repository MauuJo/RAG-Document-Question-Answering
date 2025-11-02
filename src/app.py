# import streamlit as st
# from vectorstore import VectorStore
# from chatbot import Chatbot
# import uuid

# def main():
#     st.title("Document QA Bot 🤖")
#     st.write("Upload a PDF, input your API keys, and ask questions!")

#     # Initialize session state for chat history
#     if "chat_history" not in st.session_state:
#         st.session_state["chat_history"] = []

#     # Sidebar for API keys
#     with st.sidebar:
#         st.header("API Keys 🔑")
#         cohere_api_key = st.text_input("Cohere API Key", type="password")
#         pinecone_api_key = st.text_input("Pinecone API Key", type="password")

#     # Upload PDF document
#     uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

#     user_query = st.text_input("Ask a question based on the document")

#     # Process after user submits a query
#     if st.button("Submit") and uploaded_file and cohere_api_key and pinecone_api_key:
#         with st.spinner("Processing PDF..."):
#             with open("uploaded_document.pdf", "wb") as f:
#                 f.write(uploaded_file.read())

#             # Create VectorStore instance
#             vectorstore = VectorStore("uploaded_document.pdf", cohere_api_key, pinecone_api_key)

#             # Initialize chatbot
#             chatbot = Chatbot(vectorstore, cohere_api_key)

#             with st.spinner("Generating response..."):
#                 response, retrieved_docs = chatbot.respond(user_query)

#                 # Save conversation in session state
#                 st.session_state["chat_history"].append((user_query, response, retrieved_docs))

#     # Display chat history
#     if st.session_state["chat_history"]:
#         for user_query, response, retrieved_docs in st.session_state["chat_history"]:
#             st.write(f"**You:** {user_query}")

#             accumulated_response = ""
#             for event in response:
#                 if event.event_type == "text-generation":
#                     accumulated_response += event.text
#             st.write(f"**Bot:** {accumulated_response}")

# if __name__ == "__main__":
#     main()

import streamlit as st
from vectorstore import VectorStore
from chatbot import Chatbot
import uuid
import os

# --- Utility Functions ---
def save_uploaded_file(uploaded_file):
    # Ensure a clean slate for the uploaded document
    if os.path.exists("uploaded_document.pdf"):
        os.remove("uploaded_document.pdf")
    # Use getbuffer() for Streamlit's uploaded file object
    with open("uploaded_document.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

def main():
    st.set_page_config(page_title="Document QA Bot 🤖", layout="wide")
    st.title("Document QA Bot 🤖 with Memory and Citations")

    # --- Session State Initialization ---
    if "messages" not in st.session_state:
        # Use "assistant" role for Streamlit UI compatibility
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello! Please upload a PDF and enter your API keys to begin."}]
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None
    if "chatbot" not in st.session_state:
        st.session_state["chatbot"] = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state["uploaded_file_name"] = None


    # --- Sidebar for Setup (Initial Document Processing) ---
    with st.sidebar:
        st.header("Setup 🔑")
        cohere_api_key = st.text_input("Cohere API Key", type="password", key="cohere_key")
        pinecone_api_key = st.text_input("Pinecone API Key", type="password", key="pinecone_key")
        
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf", key="pdf_uploader")

        if st.button("Process Document and Initialize Chat"):
            if uploaded_file and cohere_api_key and pinecone_api_key:
                st.session_state["uploaded_file_name"] = uploaded_file.name
                save_uploaded_file(uploaded_file)
                
                with st.spinner(f"Processing {uploaded_file.name} and indexing..."):
                    try:
                        # 1. Create VectorStore instance
                        vectorstore = VectorStore("uploaded_document.pdf", cohere_api_key, pinecone_api_key)
                        st.session_state["vectorstore"] = vectorstore

                        # 2. Initialize chatbot
                        chatbot = Chatbot(vectorstore, cohere_api_key)
                        st.session_state["chatbot"] = chatbot
                        
                        # Use "assistant" role for Streamlit UI compatibility
                        st.session_state["messages"] = [
                            {"role": "assistant", "content": f"Document **{uploaded_file.name}** processed successfully! Ask your first question."}
                        ]
                        st.success("Initialization complete! Chat is ready.")
                    except Exception as e:
                        st.error(f"Error during initialization: {e}")
                        st.session_state["vectorstore"] = None
                        st.session_state["chatbot"] = None
                        st.session_state["messages"] = [{"role": "assistant", "content": f"ERROR: Could not initialize. Details: {e}"}]

            else:
                st.error("Please provide both API keys and upload a PDF.")

    # --- Main Chat Interface ---

    # Display all existing messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user query
    if user_query := st.chat_input("Ask a question based on the document..."):
        if st.session_state["chatbot"] is None:
            st.warning("Please initialize the chat by uploading a PDF and entering keys first.")
            st.stop()

        # 1. Add user message to history and display
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # >>> START: MODIFICATION FOR COHERE API ROLES <<<
        # 2. Prepare chat history for Cohere (translating roles)
        chat_history_for_api = []
        for m in st.session_state.messages:
            role = m["role"]
            if role == "assistant":
                chat_history_for_api.append({"role": "Chatbot", "message": m["content"]})
            elif role == "user":
                chat_history_for_api.append({"role": "User", "message": m["content"]})
        # >>> END: MODIFICATION <<<
        
        # 3. Get streaming response and retrieved documents
        with st.spinner("Thinking..."):
            
            # Get streaming response and retrieved docs (reranked chunks)
            response_stream, retrieved_docs = st.session_state["chatbot"].respond(
                user_query, 
                chat_history=chat_history_for_api # Pass the translated history
            )
            
            # 4. Process and display streaming response and capture citations
            full_response_text = ""

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                for event in response_stream:
                    if event.event_type == "text-generation":
                        full_response_text += event.text
                        message_placeholder.markdown(full_response_text + "▌")
                    
                message_placeholder.markdown(full_response_text)
                
                # --- Context Visibility (Context and Citations) ---
                if retrieved_docs:
                    st.markdown("---")
                    st.markdown("**🔍 Sources (Reranked Chunks):**")
                    
                    # Displaying the top chunks for verification/context
                    for i, doc in enumerate(retrieved_docs):
                        # The text from the chunk is the source material for the citation
                        st.markdown(f"**Source [{i+1}]:**")
                        st.caption(doc['text'])
                        
        # 5. Add bot response to session state history
        # Use "assistant" role for Streamlit UI compatibility
        st.session_state.messages.append({"role": "assistant", "content": full_response_text})

if __name__ == "__main__":
    main()