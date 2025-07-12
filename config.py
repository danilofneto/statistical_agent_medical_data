# ========================================
# 3. config.py - CONFIGURAÇÕES
# ========================================

"""
Arquivo: config.py
Configurações centralizadas do projeto
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentConfig:
    """Configurações do Agente Estatístico"""
    
    # LLM Settings
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.1
    max_tokens: Optional[int] = None
    
    # Dataset Settings
    default_test_size: float = 0.2
    random_state: int = 42
    missing_threshold: float = 0.1
    
    # Analysis Settings
    max_features_importance: int = 10
    confidence_threshold: float = 0.7
    
    # File Paths
    data_dir: str = "data"
    results_dir: str = "results"
    models_dir: str = "results/models"
    plots_dir: str = "results/plots"
    
    # API Keys
    openai_api_key: Optional[str] = None
    
    def __post_init__(self):
        """Configuração pós-inicialização"""
        # Tentar obter API key do ambiente
        if not self.openai_api_key:
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Criar diretórios se não existirem
        for directory in [self.data_dir, self.results_dir, self.models_dir, self.plots_dir]:
            os.makedirs(directory, exist_ok=True)

# Instância global de configuração
CONFIG = AgentConfig()