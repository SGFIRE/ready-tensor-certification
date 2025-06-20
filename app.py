"""
RAG Assistant with Google Gemini - Main Application
A Gradio-based web interface for the RAG Assistant powered by Google Gemini.
"""

import os
from dotenv import load_dotenv
from src.rag_assistant import RAGAssistant
from src.interface import create_interface
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rag_assistant.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main application entry point"""
    try:
        # Create the Gradio interface
        interface = create_interface()
        
        # Launch configuration
        port = int(os.getenv('APP_PORT', 7860))
        host = os.getenv('APP_HOST', '0.0.0.0')
        
        logger.info(f"Starting RAG Assistant on {host}:{port}")
        
        # Launch the interface
        interface.launch(
            server_name=host,
            server_port=port,
            share=False,  # Set to True for public links
            debug=True,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

if __name__ == "__main__":
    main()
