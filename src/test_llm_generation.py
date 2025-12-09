
import logging
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.agents.llm_client import OllamaClient
from src.utils.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_llm():
    print("Testing LLM Client...")
    
    try:
        # Initialize client
        config = get_config()
        print(f"Configured Base URL: {config.ollama_base_url}")
        
        client = OllamaClient(base_url=config.ollama_base_url)
        print(f"Client initialized with Base URL: {client.base_url}")
        
        # Test 1: Simple Generation
        print("\nTest 1: Simple Generation")
        response = client.generate(prompt="Say 'Hello, World!'", max_tokens=50)
        print(f"Response: {response}")
        
        # Test 2: JSON Generation
        print("\nTest 2: JSON Generation")
        json_prompt = """
        Generate a JSON object with a 'message' field containing 'Hello JSON'.
        Ensure the response is valid JSON.
        """
        response = client.generate(prompt=json_prompt, max_tokens=100, json_mode=True)
        print(f"Raw Response: {response}")
        
        json_data = client.extract_json(response)
        print(f"Parsed JSON: {json_data}")
        
        if json_data and json_data.get('message') == 'Hello JSON':
            print("\nSUCCESS: LLM is working and generating JSON.")
        else:
            print("\nWARNING: JSON parsing failed or content mismatch.")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_llm()
