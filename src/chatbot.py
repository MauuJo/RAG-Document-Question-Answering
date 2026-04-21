import cohere
import uuid

class Chatbot:
    def __init__(self, vectorstore, cohere_api_key: str):
        self.vectorstore = vectorstore
        self.conversation_id = str(uuid.uuid4())
        self.co = cohere.Client(cohere_api_key)

    def _route_query(self, user_message: str) -> str:
        """
        Agentic Routing: Uses a smart chat model to classify intent.
        """
        routing_prompt = f"""You are an intent classification agent.
Classify the user's message into exactly one of these two categories:
- CHAT: Casual greetings (hi, hello, hey, thanks, thank you, bye, how are you), 
        pleasantries, or simple conversational replies with NO information need.
- RAG: Any question, request for information, facts, summaries, explanations, 
       or anything that requires searching a document to answer.

User Message: "{user_message}"

Respond with ONLY ONE WORD — either CHAT or RAG. No punctuation, no explanation."""

        try:
            response = self.co.chat(
                message=routing_prompt,
                model="command-a-03-2025",
                temperature=0.0,
                # Disable RAG on the routing call itself — critical fix!
                # If your Cohere client version supports connectors, make sure
                # none are attached here. A plain .chat() call is fine.
            )
            raw = response.text.strip().upper()
            print(f"[Router raw output]: '{raw}'")

            # Be strict: only accept exact matches to avoid partial-match bugs
            if raw == "CHAT":
                return "CHAT"
            elif raw == "RAG":
                return "RAG"
            else:
                # Model returned something unexpected (e.g. "CHAT." or a sentence)
                print(f"[Router warning]: Unexpected output '{raw}', defaulting to RAG")
                return "CHAT" if "CHAT" in raw else "RAG"

        except Exception as e:
            print(f"[Routing error]: {e}")
            return "RAG"

    def respond(self, user_message: str, chat_history: list):

        # 1. Route the query
        route = self._route_query(user_message)
        print(f"[Router decision]: {route} pipeline")

        # 2. CHAT pipeline — no vector store involved
        if route == "CHAT":
            response_stream = self.co.chat_stream(
                message=user_message,
                model="command-a-03-2025",
                chat_history=chat_history,
                preamble="You are a helpful, friendly AI assistant. Respond casually and warmly.",
                temperature=0.7,
            )
            return response_stream, []  # ← Return empty list, NOT None (safer to iterate)

        # 3. RAG pipeline — query vector store
        retrieved_docs = self.vectorstore.retrieve(user_message)
        formatted_docs = [{"text": doc["text"]} for doc in retrieved_docs]

        response_stream = self.co.chat_stream(
            message=user_message,
            model="command-a-03-2025",
            documents=formatted_docs,
            chat_history=chat_history,
            temperature=0.0,
            prompt_truncation="AUTO",
            preamble=(
                "You are an expert Q&A system. Answer questions strictly using "
                "the provided documents. If the answer is not in the documents, "
                "say: 'I cannot answer this based on the provided document.'"
            ),
        )
        return response_stream, retrieved_docs