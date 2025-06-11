"""
Gradio interface for the RAG Assistant.
"""

import gradio as gr
from typing import List, Tuple
import logging

from .rag_assistant import RAGAssistant

logger = logging.getLogger(__name__)

# Global RAG assistant instance
rag_assistant = None

def initialize_assistant(gemini_key: str, json_file) -> str:
    """Initialize the RAG assistant with knowledge base"""
    global rag_assistant
    
    try:
        if not gemini_key:
            return "❌ Please provide your Gemini API key"
        
        if json_file is None:
            return "❌ Please upload your JSON knowledge base file"
        
        rag_assistant = RAGAssistant(gemini_api_key=gemini_key)
        documents = rag_assistant.load_json_knowledge_base(json_file.name)
        
        if not documents:
            return "❌ No documents found in the JSON file"
        
        rag_assistant.create_vectorstore(documents, "vectorstore/index")
        
        return f"✅ Assistant initialized successfully with {len(documents)} documents using Gemini!"
        
    except Exception as e:
        return f"❌ Error initializing assistant: {str(e)}"

def chat_response(message: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
    """Handle chat responses"""
    global rag_assistant
    
    if not rag_assistant:
        response = "Please initialize the assistant first by providing your Gemini API key and uploading your knowledge base."
        history.append([message, response])
        return "", history
    
    try:
        answer, sources = rag_assistant.get_response(message)
        
        if sources:
            unique_sources = list(set(sources))
            source_text = f"\n\n📚 **Sources**: {', '.join(unique_sources[:3])}"
            response = answer + source_text
        else:
            response = answer
        
        history.append([message, response])
        return "", history
        
    except Exception as e:
        error_response = f"Sorry, I encountered an error: {str(e)}"
        history.append([message, error_response])
        return "", history

def clear_chat() -> List:
    """Clear chat history"""
    global rag_assistant
    if rag_assistant:
        rag_assistant.clear_memory()
    return []

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="Ready Tensor RAG Assistant - Powered by Gemini",
        theme=gr.themes.Soft(),
        css="""
        .header {
            text-align: center;
            background: linear-gradient(90deg, #4285f4 0%, #34a853 50%, #ea4335 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .gemini-badge {
            background: linear-gradient(90deg, #4285f4, #34a853);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            display: inline-block;
            margin: 0.5rem 0;
            font-weight: bold;
        }
        """
    ) as interface:
        
        gr.HTML("""
        <div class="header">
            <h1>🤖 Ready Tensor RAG Assistant</h1>
            <div class="gemini-badge">✨ Powered by Google Gemini</div>
            <p>Your intelligent document-powered AI assistant using Google's advanced AI</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🔧 Setup")
                
                gemini_key_input = gr.Textbox(
                    label="Google Gemini API Key",
                    type="password",
                    placeholder="Enter your Gemini API key..."
                )
                
                json_file_input = gr.File(
                    label="Knowledge Base (JSON)",
                    file_types=[".json"]
                )
                
                initialize_btn = gr.Button("🚀 Initialize Assistant", variant="primary")
                status_output = gr.Textbox(label="Status", interactive=False)
                
                gr.Markdown("## 🛠️ Actions")
                clear_btn = gr.Button("🗑️ Clear Chat History", variant="secondary")
            
            with gr.Column(scale=2):
                gr.Markdown("## 💬 Chat with Your Assistant")
                
                chatbot = gr.Chatbot(
                    label="Ready Tensor Assistant (Gemini)",
                    height=500
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask me anything about your documents...",
                        scale=4,
                        show_label=False
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
        
        # Event handlers
        initialize_btn.click(
            initialize_assistant,
            inputs=[gemini_key_input, json_file_input],
            outputs=status_output
        )
        
        msg_input.submit(
            chat_response,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )
        
        send_btn.click(
            chat_response,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )
        
        clear_btn.click(clear_chat, outputs=chatbot)
    
    return interface
