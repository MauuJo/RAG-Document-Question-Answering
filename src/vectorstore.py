import cohere
import fitz
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from pinecone_text.sparse import BM25Encoder

class VectorStore:
    # NEW: Added 'namespace' to the initialization arguments
    def __init__(self, pdf_path: str, cohere_api_key: str, pinecone_api_key: str, namespace: str):
        self.pdf_path = pdf_path
        self.co = cohere.Client(cohere_api_key)
        self.pinecone_api_key = pinecone_api_key
        self.namespace = namespace
        self.index_name = 'rag-qa-bot-hybrid' # NEW: Changed name for hybrid
        self.bm25 = BM25Encoder().default()   # NEW: Initialize sparse encoder
        self.chunks = []
        self.embeddings = []
        self.retrieve_top_k = 10
        self.rerank_top_k = 3
        self.index = None
        
        self.load_pdf()
        self.split_text() 
        self.embed_chunks()
        self.index_chunks()

    def load_pdf(self):
        self.pdf_text = self.extract_text_from_pdf(self.pdf_path)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        with fitz.open(pdf_path) as pdf:
            for page_num in range(pdf.page_count):
                page = pdf.load_page(page_num)
                text += page.get_text("text")
        return text

    def split_text(self, chunk_size=1000, chunk_overlap=100):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""], 
            length_function=len
        )
        self.chunks = text_splitter.split_text(self.pdf_text)
        
        if not self.chunks:
             print("Warning: Document splitting resulted in zero chunks.")

    def embed_chunks(self, batch_size=90):
        total_chunks = len(self.chunks)
        embed_model = "embed-multilingual-v3.0" 
        
        for i in range(0, total_chunks, batch_size):
            batch = self.chunks[i:min(i + batch_size, total_chunks)]
            batch_embeddings = self.co.embed(
                texts=batch, input_type="search_document", model=embed_model
            ).embeddings
            self.embeddings.extend(batch_embeddings)

    def index_chunks(self):
        """Indexes the embedded chunks using Pinecone Hybrid Search."""
        self.pc = Pinecone(api_key=self.pinecone_api_key)

        if not self.embeddings:
            return

        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=len(self.embeddings[0]),
                metric='dotproduct', # CRITICAL: Hybrid requires dotproduct
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        self.index = self.pc.Index(self.index_name)

        # 1. Fit the BM25 encoder to learn the vocabulary of your specific PDF
        self.bm25.fit(self.chunks)
        
        # 2. Encode all chunks into sparse keyword vectors
        sparse_vectors = self.bm25.encode_documents(self.chunks)

        # 3. Package Dense + Sparse + Metadata together
        vectors_to_upsert = []
        for i in range(len(self.chunks)):
            vectors_to_upsert.append({
                'id': str(i),
                'values': self.embeddings[i],           # Dense (Cohere)
                'sparse_values': sparse_vectors[i],     # Sparse (BM25)
                'metadata': {'text': self.chunks[i]}
            })

        # 4. Upsert to your isolated namespace
        self.index.upsert(vectors=vectors_to_upsert, namespace=self.namespace)

    def generate_queries(self, original_query: str) -> list:
        """Uses an LLM to generate alternative versions of the user's query for better recall."""
        prompt = f"""You are an AI assistant tasked with generating 3 different versions of the following user question. 
        The goal is to use these variations to retrieve relevant documents from a vector database. 
        Use different keywords and rephrase the intent.
        
        Original question: {original_query}
        
        Provide only the alternative questions, separated by newlines. Do not use bullet points or numbers."""
        
        try:
            response = self.co.chat(
                message=prompt, 
                model="command-a-03-2025", # We use light here for speed!
                temperature=0.0
            )
            # Split by newline and clean up any accidental whitespace
            queries = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
            
            # Combine the original query with the new ones
            final_queries = [original_query] + queries[:3]
            print(f"Expanded Queries: {final_queries}") # Prints to terminal so you can see the magic
            return final_queries
        except Exception as e:
            print(f"Query expansion failed: {e}")
            return [original_query] # Fallback to original if it fails
        

    def retrieve(self, query: str) -> list:
        # 1. Expand the single query into multiple variations
        expanded_queries = self.generate_queries(query)
        
        all_retrieved_texts = []
        seen_texts = set()

        # 2. Execute Hybrid Search for EVERY variation
        for q in expanded_queries:
            # Dense Vector
            query_emb = self.co.embed(texts=[q], model="embed-multilingual-v3.0", input_type="search_query").embeddings[0]
            
            # Sparse Vector (BM25)
            sparse_query = self.bm25.encode_queries(q)
            
            # Query Pinecone
            res = self.index.query(
                vector=query_emb,
                sparse_vector=sparse_query,
                top_k=5, # Reduced slightly per query so we don't overwhelm the pool
                include_metadata=True,
                namespace=self.namespace
            )
            
            # 3. Deduplicate the results
            for match in res['matches']:
                text = match['metadata']['text']
                if text not in seen_texts:
                    seen_texts.add(text)
                    all_retrieved_texts.append(text)

        print(f"Total unique chunks retrieved before reranking: {len(all_retrieved_texts)}")

        # 4. Rerank the massive pooled context using the ORIGINAL query
        if not all_retrieved_texts:
            return []
            
        rerank_results = self.co.rerank(
            query=query, # Always rerank against what the user actually asked
            documents=all_retrieved_texts,
            top_n=self.rerank_top_k,
            model="rerank-v3.5"
        )
        
        # 5. Return the top chunks formatted correctly
        final_docs = [{'text': all_retrieved_texts[result.index]} for result in rerank_results.results]
        return final_docs

    # NEW: Added the cleanup method
    def delete_namespace(self):
        """Deletes the entire namespace from Pinecone to free up space."""
        if self.index:
            try:
                self.index.delete(delete_all=True, namespace=self.namespace)
                print(f"Successfully deleted namespace: {self.namespace}")
            except Exception as e:
                print(f"Error deleting namespace {self.namespace}: {e}")

    
    def retrieve_baseline(self, query: str) -> list:
        """Naive RAG: Dense-only search, no expansion, no sparse vectors."""
        query_emb = self.co.embed(texts=[query], model="embed-multilingual-v3.0", input_type="search_query").embeddings[0]
        
        # We query the same index but skip 'sparse_vector' and 'namespace'
        # to simulate a standard, unoptimized global search
        res = self.index.query(
            vector=query_emb,
            top_k=self.rerank_top_k, # No reranking later, just take top K
            include_metadata=True
        )
        return [{'text': match['metadata']['text']} for match in res['matches']]