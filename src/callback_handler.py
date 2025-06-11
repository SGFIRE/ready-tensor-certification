"""
Custom callback handlers for logging and monitoring LLM interactions.
"""

import logging
from typing import Dict, Any, List
from langchain.callbacks.base import BaseCallbackHandler

logger = logging.getLogger(__name__)

class CustomCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for logging LLM interactions"""
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts generating"""
        logger.info(f"LLM started with prompt: {prompts[0][:100]}...")
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM finishes generating"""
        logger.info("LLM finished generating response")
    
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called when LLM encounters an error"""
        logger.error(f"LLM error: {error}")
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Called when chain starts"""
        logger.debug("Chain started")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when chain ends"""
        logger.debug("Chain completed")
    
    def on_chain_error(self, error: Exception, **kwargs) -> None:
        """Called when chain encounters an error"""
        logger.error(f"Chain error: {error}")
