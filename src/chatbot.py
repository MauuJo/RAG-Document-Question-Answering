

import cohere
import uuid

class Chatbot:
    def __init__(self, vectorstore, cohere_api_key: str):
        self.vectorstore = vectorstore
        self.conversation_id = str(uuid.uuid4())
        self.co = cohere.Client(cohere_api_key)

    def respond(self, user_message: str, chat_history: list):
        
        # 1. Retrieve documents from your vector store using the user's query
        retrieved_docs = self.vectorstore.retrieve(user_message)

        # 2. Format the retrieved documents for the Cohere API
        #    Each document must be a list of dicts with a 'text' key.
        formatted_docs = [{"text": doc['text']} for doc in retrieved_docs]

        # 3. Call chat_stream once, passing documents and history
        response_stream = self.co.chat_stream(
            message=user_message,
            model="command-a-03-2025", 
            documents=formatted_docs, 
            chat_history=chat_history, 
        )
        
        # 4. Return the streaming response and the documents used
        return response_stream, retrieved_docs