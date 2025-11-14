"""
Agents Package
Handles LLM interactions for resume analysis and ranking.
"""

from src.agents.llm_client import OllamaClient, create_llm_client
from src.agents.analyzer import ResumeAnalyzer, CandidateAnalysis, create_analyzer
from src.agents.prompt_templates import (
    PromptTemplates, 
    PromptBuilder, 
    PromptVariations,
    get_system_prompt,
    build_custom_prompt
)

__all__ = [
    # LLM Client
    'OllamaClient',
    'create_llm_client',
    
    # Analyzer
    'ResumeAnalyzer',
    'CandidateAnalysis',
    'create_analyzer',
    
    # Prompt Templates
    'PromptTemplates',
    'PromptBuilder',
    'PromptVariations',
    'get_system_prompt',
    'build_custom_prompt',
]

__version__ = '1.0.0'