"""
RAG Assistant with Google Gemini
A powerful document-based question answering system.
"""

__version__ = "1.0.0"
__author__ = "Ready Tensor"
__email__ = "support@readytensor.ai"

from .rag_assistant import RAGAssistant
from .callback_handler import CustomCallbackHandler

__all__ = ["RAGAssistant", "CustomCallbackHandler"]
