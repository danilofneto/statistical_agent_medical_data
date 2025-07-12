# ========================================
# 4. utils.py - FUNÇÕES UTILITÁRIAS
# ========================================

"""
Arquivo: utils.py
Funções utilitárias para o projeto
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import json

def setup_plotting_style():
    """Configura estilo padrão para plots"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

def validate_dataset(data: pd.DataFrame) -> Dict[str, Any]:
    """Valida dataset antes da análise"""
    validation = {
        'shape': data.shape,
        'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
        'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': data.select_dtypes(include=['object', 'category']).columns.tolist(),
        'has_target': False,
        'recommendations': []
    }
    
    # Verificar possíveis variáveis alvo
    binary_cols = []
    for col in validation['numeric_columns']:
        if data[col].nunique() == 2 and set(data[col].unique()).issubset({0, 1, np.nan}):
            binary_cols.append(col)
    
    if binary_cols:
        validation['has_target'] = True
        validation['potential_targets'] = binary_cols
        validation['recommendations'].append(f"Variáveis alvo potenciais: {binary_cols}")
    
    # Recomendações de limpeza
    high_missing_cols = [col for col, pct in validation['missing_percentage'].items() if pct > 20]
    if high_missing_cols:
        validation['recommendations'].append(f"Considere remover colunas com muitos valores faltantes: {high_missing_cols}")
    
    return validation

def save_analysis_report(results: Dict[str, Any], filename: str):
    """Salva relatório de análise em formato markdown"""
    report = f"""# Relatório de Análise Estatística

## Informações Gerais
- **Data/Hora:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Tipo de Análise:** {results.get('results', {}).get('analysis_type', 'N/A')}
- **Confiança:** {results.get('results', {}).get('confidence', 0):.2%}

## Resultados

### Interpretação
{results.get('results', {}).get('interpretation', 'Não disponível')}

### Recomendações
"""
    
    recommendations = results.get('results', {}).get('recommendations', [])
    for i, rec in enumerate(recommendations, 1):
        report += f"{i}. {rec}\n"
    
    report += f"""
### Modelos Utilizados
{', '.join(results.get('models', {}).keys()) if results.get('models') else 'Nenhum'}

### Explicabilidade
- **SHAP disponível:** {'Sim' if 'shap' in results.get('results', {}).get('explainability', {}) else 'Não'}
- **LIME disponível:** {'Sim' if 'lime' in results.get('results', {}).get('explainability', {}) else 'Não'}
"""
    
    with open(f"results/{filename}.md", "w", encoding="utf-8") as f:
        f.write(report)