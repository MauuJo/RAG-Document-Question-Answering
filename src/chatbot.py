# import cohere
# import uuid

# class Chatbot:
#     def __init__(self, vectorstore, cohere_api_key: str):
#         self.vectorstore = vectorstore
#         self.conversation_id = str(uuid.uuid4())
#         self.co = cohere.Client(cohere_api_key)

#     # def respond(self, user_message: str):
#     #     response = self.co.chat(message=user_message, model="command-r")
#     #     retrieved_docs = []
#     #     if response.search_queries:
#     #         for query in response.search_queries:
#     #             retrieved_docs.extend(self.vectorstore.retrieve(query.text))
#     #         response = self.co.chat_stream(
#     #             message=user_message,
#     #             model="command-r",
#     #             documents=retrieved_docs,
#     #             conversation_id=self.conversation_id,
#     #         )
#     #     else:
#     #         response = self.co.chat_stream(
#     #             message=user_message,
#     #             model="command-r",
#     #             conversation_id=self.conversation_id,
#     #         )
#     #     return response, retrieved_docs

#     def respond(self, user_message: str):
#         # 1. Retrieve documents from your vector store using the user's query
#         #    Note: This is the critical change. You use the user_message as the query.
#         retrieved_docs = self.vectorstore.retrieve(user_message)

#         # 2. Format the retrieved documents for the Cohere API
#         #    The Cohere Chat API expects documents as a list of dictionaries with a 'text' key.
#         formatted_docs = [{"text": doc['text']} for doc in retrieved_docs]

#         # 3. Call chat_stream once, passing the documents for RAG
#         #    The model will use these documents to generate a grounded response.
#         #    'model="command-r"' is an alias for an old model; consider using 'command-r-plus'
#         response = self.co.chat_stream(
#             message=user_message,
#             model="command-a-03-2025",
#             documents=formatted_docs, # Pass your retrieved documents here
#             conversation_id=self.conversation_id,
#         )
        
#         # 4. Return the streaming response and the documents used
#         return response, retrieved_docs

import cohere
import uuid

class Chatbot:
    def __init__(self, vectorstore, cohere_api_key: str):
        self.vectorstore = vectorstore
        self.conversation_id = str(uuid.uuid4())
        self.co = cohere.Client(cohere_api_key)

    # >>> MODIFIED TO ACCEPT CHAT HISTORY <<<
    def respond(self, user_message: str, chat_history: list):
        
        # 1. Retrieve documents from your vector store using the user's query
        retrieved_docs = self.vectorstore.retrieve(user_message)

        # 2. Format the retrieved documents for the Cohere API
        #    Each document must be a list of dicts with a 'text' key.
        formatted_docs = [{"text": doc['text']} for doc in retrieved_docs]

        # 3. Call chat_stream once, passing documents and history
        response_stream = self.co.chat_stream(
            message=user_message,
            model="command-a-03-2025", # Updated to a stable model
            documents=formatted_docs, 
            chat_history=chat_history, # Passed for memory
            # conversation_id=self.conversation_id,
        )
        
        # 4. Return the streaming response and the documents used
        return response_stream, retrieved_docs