"""
LLM Client Module
Handles communication with Ollama for text generation.
Provides a clean interface for LLM interactions.
"""

import logging
import requests
import json
from typing import List, Dict, Optional, Any, Generator
import time
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for interacting with Ollama LLM API.
    Handles text generation, streaming, and chat completions.
    """
    
    def __init__(self,
                 model: str = "qwen3:latest",
                 base_url: Optional[str] = None,
                 temperature: float = 0.4,
                 max_tokens: int = 3000,
                 timeout: int = 500):
        """
        Initialize Ollama client.

        Args:
            model: Ollama model name (e.g., 'qwen3:latest', 'gemma3:4b', 'mistral')
            base_url: Ollama API base URL
            temperature: Sampling temperature (0.0-1.0)
                - 0.0: Deterministic, focused
                - 0.7: Balanced (default)
                - 1.0: Creative, diverse
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.model = model
        if base_url:
            self.base_url = base_url.rstrip('/')
        else:
            self.base_url = get_config().ollama_base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Verify connection and model availability
        self._verify_connection()
        
        logger.info(f"OllamaClient initialized (model: {model}, temp: {temperature})")
    
    def _verify_connection(self):
        """Verify Ollama is running and model is available."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            # Check if model is available
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if not any(self.model in name for name in model_names):
                logger.warning(f"Model '{self.model}' not found. Pulling model...")
                self._pull_model()
            else:
                logger.info(f"Model '{self.model}' is available")
                
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running with: 'ollama serve'"
            )
        except Exception as e:
            logger.error(f"Error verifying Ollama connection: {str(e)}")
            raise
    
    def _pull_model(self):
        """Pull the LLM model if not available."""
        try:
            logger.info(f"Pulling model: {self.model}")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model},
                timeout=900  # 10 minutes for large models
            )
            response.raise_for_status()
            logger.info(f"Successfully pulled model: {self.model}")
        except Exception as e:
            raise RuntimeError(f"Failed to pull model '{self.model}': {str(e)}")
    
    def generate(self,
                prompt: str,
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None,
                system_prompt: Optional[str] = None,
                stop_sequences: Optional[List[str]] = None,
                json_mode: bool = False) -> str:
        """
        Generate text from a prompt (non-streaming).
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            system_prompt: System message to guide model behavior
            stop_sequences: List of sequences where generation should stop
            json_mode: Whether to enforce JSON output (Ollama feature)
        
        Returns:
            Generated text
        """
        logger.debug(f"Generating with prompt: {prompt[:100]}...")
        
        # Prepare request
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
            }
        }
        
        # Enable JSON mode if requested
        if json_mode:
            request_data["format"] = "json"
        
        # Add system prompt if provided
        if system_prompt:
            request_data["system"] = system_prompt
        
        # Add stop sequences if provided
        if stop_sequences:
            request_data["options"]["stop"] = stop_sequences
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            elapsed = time.time() - start_time
            
            result = response.json()
            generated_text = result.get('response', '')
            
            logger.info(f"Generation completed in {elapsed:.2f}s ({len(generated_text)} chars)")
            
            return generated_text
            
        except requests.exceptions.Timeout:
            logger.error("Generation timeout")
            raise TimeoutError(f"Generation timed out after {self.timeout}s")
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
    def generate_stream(self,
                       prompt: str,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        """
        Generate text with streaming (yields tokens as they're generated).
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            system_prompt: System message
        
        Yields:
            Generated text chunks
        """
        logger.debug(f"Streaming generation with prompt: {prompt[:100]}...")
        
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
            }
        }
        
        if system_prompt:
            request_data["system"] = system_prompt
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if 'response' in chunk:
                        yield chunk['response']
                    
                    # Check if done
                    if chunk.get('done', False):
                        break
                        
        except Exception as e:
            logger.error(f"Streaming generation failed: {str(e)}")
            raise
    
    def chat(self,
            messages: List[Dict[str, str]],
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None) -> str:
        """
        Chat completion with conversation history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                Example: [
                    {'role': 'system', 'content': 'You are a helpful assistant'},
                    {'role': 'user', 'content': 'Hello!'},
                    {'role': 'assistant', 'content': 'Hi there!'},
                    {'role': 'user', 'content': 'How are you?'}
                ]
            temperature: Override default temperature
            max_tokens: Override default max tokens
        
        Returns:
            Assistant's response
        """
        logger.debug(f"Chat completion with {len(messages)} messages")
        
        request_data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
            }
        }
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=request_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            elapsed = time.time() - start_time
            
            result = response.json()
            assistant_message = result.get('message', {}).get('content', '')
            
            logger.info(f"Chat completed in {elapsed:.2f}s")
            
            return assistant_message
            
        except Exception as e:
            logger.error(f"Chat completion failed: {str(e)}")
            raise
    
    def chat_stream(self,
                   messages: List[Dict[str, str]],
                   temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None) -> Generator[str, None, None]:
        """
        Chat completion with streaming.
        
        Args:
            messages: List of message dicts
            temperature: Override default temperature
            max_tokens: Override default max tokens
        
        Yields:
            Response chunks
        """
        request_data = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=request_data,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if 'message' in chunk and 'content' in chunk['message']:
                        yield chunk['message']['content']
                    
                    if chunk.get('done', False):
                        break
                        
        except Exception as e:
            logger.error(f"Streaming chat failed: {str(e)}")
            raise
    
    def extract_json(self, text: str) -> Optional[Dict]:
        """
        Extract JSON from LLM response (handles markdown code blocks).
        
        Args:
            text: LLM response text
        
        Returns:
            Parsed JSON dict or None if parsing fails
        """
        # Try to find JSON in markdown code blocks
        import re
        
        # Pattern 1: ```json ... ```
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            # Pattern 2: ``` ... ```
            json_match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # Pattern 3: Raw JSON
                json_text = text
        
        # Pattern 4: Try to find first { and last }
        if not json_match:
            try:
                start_idx = text.find('{')
                end_idx = text.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_text = text[start_idx:end_idx+1]
            except Exception:
                pass
        
        # Try to parse
        try:
            return json.loads(json_text.strip())
        except json.JSONDecodeError as e:
            logger.warning("Failed to extract JSON from response")
            logger.debug(f"JSON parse error: {str(e)}")
            logger.debug(f"Response preview (first 500 chars): {text[:500]}")
            logger.debug(f"Extracted text (first 500 chars): {json_text[:500]}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Model information dictionary
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model},
                timeout=10
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {}


# Utility functions
def create_llm_client(model: str = "qwen3:latest",
                     temperature: float = 0.4) -> OllamaClient:
    """
    Convenience function to create an LLM client.

    Args:
        model: Model name (default: qwen3:latest)
        temperature: Sampling temperature

    Returns:
        OllamaClient instance
    """
    return OllamaClient(model=model, temperature=temperature)