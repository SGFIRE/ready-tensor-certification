"""
Core RAG Assistant implementation using Google Gemini and LangChain.
"""

import json
import logging
import os
from typing import List, Dict, Any, Tuple
from datetime import datetime

import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

from .callback_handler import CustomCallbackHandler

logger = logging.getLogger(__name__)

class RAGAssistant:
    """RAG Assistant powered by Google Gemini"""
    
    def __init__(self, gemini_api_key: str = None):
        """Initialize the RAG Assistant with Gemini"""
        self.gemini_api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("Gemini API key is required. Set GOOGLE_API_KEY environment variable or pass it directly.")
        
        # Configure Gemini
        genai.configure(api_key=self.gemini_api_key)
        
        # Initialize components
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "models/embedding-001"),
            google_api_key=self.gemini_api_key
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            temperature=float(os.getenv("TEMPERATURE", 0.7)),
            google_api_key=self.gemini_api_key,
            callbacks=[CustomCallbackHandler()],
            convert_system_message_to_human=True
        )
        
        self.vectorstore = None
        self.conversation_chain = None
        
        # Memory configuration
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=5
        )
        
        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 100)),
            length_function=len
        )
        
        logger.info("RAG Assistant initialized successfully with Gemini")
    
    def load_json_knowledge_base(self, json_file_path: str) -> List[Document]:
        """Load and process JSON knowledge base"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            documents = []
            
            if isinstance(data, list):
                for i, item in enumerate(data):
                    content = self._extract_content_from_item(item, i)
                    if content:
                        doc = Document(
                            page_content=content,
                            metadata={"source": f"document_{i}", "item_id": str(i)}
                        )
                        documents.append(doc)
            
            elif isinstance(data, dict):
                for key, value in data.items():
                    content = self._extract_content_from_item(value, key)
                    if content:
                        doc = Document(
                            page_content=content,
                            metadata={"source": str(key), "item_id": str(key)}
                        )
                        documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents from JSON knowledge base")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading JSON knowledge base: {e}")
            raise
    
    def _extract_content_from_item(self, item: Any, identifier: Any) -> str:
        """Extract text content from a JSON item"""
        content_parts = []
        
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, (str, int, float)):
                    content_parts.append(f"{key}: {value}")
                elif isinstance(value, list):
                    content_parts.append(f"{key}: {', '.join(map(str, value))}")
                elif isinstance(value, dict):
                    nested_content = self._extract_content_from_item(value, key)
                    if nested_content:
                        content_parts.append(f"{key}: {nested_content}")
        
        elif isinstance(item, (str, int, float)):
            content_parts.append(f"Content: {item}")
        
        elif isinstance(item, list):
            content_parts.append(f"Items: {', '.join(map(str, item))}")
        
        return " | ".join(content_parts) if content_parts else ""
    
    def create_vectorstore(self, documents: List[Document], save_path: str = None):
        """Create and optionally save the vector store"""
        try:
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
            
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            
            if save_path:
                self.vectorstore.save_local(save_path)
                logger.info(f"Vector store saved to {save_path}")
            
            self._setup_conversation_chain()
            logger.info("Vector store created successfully")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def load_vectorstore(self, load_path: str):
        """Load existing vector store"""
        try:
            self.vectorstore = FAISS.load_local(
                load_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self._setup_conversation_chain()
            logger.info(f"Vector store loaded from {load_path}")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def _setup_conversation_chain(self):
        """Setup the conversational retrieval chain"""
        system_template = """You are a helpful AI assistant from Ready Tensor platform, designed to provide accurate and personalized responses based on the provided context. 

Use the following context to answer the user's question. If the context doesn't contain relevant information, politely say so and provide general guidance when appropriate.

Always maintain a friendly, professional tone and structure your responses clearly. When possible, reference specific information from the context to support your answers.

Context: {context}

Question: {question}

Remember: You represent Ready Tensor platform, so maintain professionalism while being helpful and conversational."""

        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": int(os.getenv("RETRIEVAL_K", 4))}
            ),
            memory=self.memory,
            return_source_documents=True,
            output_key="answer",
            combine_docs_chain_kwargs={
                "prompt": ChatPromptTemplate.from_template(system_template)
            }
        )
    
    def get_response(self, question: str) -> Tuple[str, List[str]]:
        """Get response from the RAG system"""
        try:
            if not self.conversation_chain:
                return "Sorry, the knowledge base is not loaded yet. Please load your documents first.", []
            
            result = self.conversation_chain({"question": question})
            
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    source_info = doc.metadata.get("source", "Unknown")
                    sources.append(source_info)
            
            logger.info(f"Generated response for question: {question[:50]}...")
            return result["answer"], sources
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Sorry, I encountered an error while processing your question: {str(e)}", []
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        logger.info("Conversation memory cleared")
