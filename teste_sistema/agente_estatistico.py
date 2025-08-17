#
# Arquivo: agente_estatistico_real.py
# Descrição: Versão do agente adaptada para carregar e analisar um dataset real (UCI Heart Disease).
#
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Libs para Análise Preditiva e XAI
from sklearn.linear_model import LogisticRegression

# Libs para Inferência Causal
from dowhy import CausalModel

# Libs para Programação Probabilística (PyMC v4+)
import pymc as pm
import arviz as az
import pytensor.tensor as pt

print("Dependências do Agente Estatístico importadas com sucesso.")

def carregar_e_preparar_heart_disease_data():
    """
    Carrega e prepara o dataset "Heart Disease" da UCI.
    Esta função encapsula o pré-processamento necessário para dados reais.
    """
    print("Carregando dataset real 'Heart Disease' da UCI...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Nomes das colunas conforme a documentação do dataset
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    try:
        data = pd.read_csv(url, names=column_names, na_values='?')
        
        # Pré-processamento
        data = data.dropna() # Remove linhas com valores faltantes
        data['target'] = (data['target'] > 0).astype(int)  # Alvo binário: 0=sem doença, 1=com doença
        
        # Mapeamento para melhor interpretabilidade (opcional, mas recomendado)
        data['sex_label'] = data['sex'].map({0: 'Feminino', 1: 'Masculino'})
        
        print(f"✅ Dataset carregado e preparado com sucesso. Dimensões: {data.shape}")
        return data
        
    except Exception as e:
        print(f"❌ Erro ao carregar ou processar o dataset: {e}")
        return None

class AgenteEstatistico:
    """
    Um agente de IA que realiza diferentes tipos de análises estatísticas
    em um determinado conjunto de dados.
    """
    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Os dados de entrada devem ser um DataFrame do Pandas.")
        self.data = data.copy()
        print(f"Agente Estatístico inicializado com {self.data.shape[0]} linhas de dados.")

    def analisar(self, analysis_type: str, **kwargs):
        """Ponto de entrada principal para realizar uma análise."""
        print(f"\n>>> [Agente Estatístico] Iniciando análise do tipo: '{analysis_type}'")
        if analysis_type == 'preditiva':
            return self._run_predictive_analysis(**kwargs)
        elif analysis_type == 'causal':
            return self._run_causal_analysis(**kwargs)
        else:
            return {"error": f"Tipo de análise desconhecido: {analysis_type}"}

    def _run_predictive_analysis(self, target_column: str, feature_columns: list):
        """Realiza uma análise preditiva usando Regressão Logística e XAI."""
        print("   - Treinando modelo de Regressão Logística...")
        X = self.data[feature_columns]
        y = self.data[target_column]
        model = LogisticRegression(random_state=42, max_iter=1000).fit(X, y)
        
        importances = pd.DataFrame(data=model.coef_[0], index=feature_columns, columns=["Importância"])
        importances = importances.sort_values(by="Importância", ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances.index, y=importances['Importância'])
        plt.title(f"XAI: Importância das Variáveis para Prever '{target_column}'")
        plt.ylabel("Coeficiente da Regressão Logística")
        plt.xticks(rotation=45, ha='right')
        
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        viz_base64 = base64.b64encode(buf.getbuffer()).decode("ascii")
        plt.close()
        
        summary_text = f"Modelo de Regressão Logística treinado para prever '{target_column}'. As variáveis mais influentes foram: {', '.join(importances.index[:3])}."
        
        return {"analysis_type": "Preditiva (com XAI)", "summary": summary_text, "visualization_b64": viz_base64}

    def _run_causal_analysis(self, treatment_column: str, outcome_column: str, common_causes: list):
        """Realiza uma análise de inferência causal usando DoWhy."""
        print("   - Construindo modelo causal com DoWhy...")
        model = CausalModel(data=self.data, treatment=treatment_column, outcome=outcome_column, common_causes=common_causes)
        identified_estimand = model.identify_effect()
        estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_matching")
        ate = estimate.value
        
        summary_text = (
            f"Análise de Inferência Causal concluída.\n"
            f"Efeito causal médio de '{treatment_column}' sobre '{outcome_column}' é de aproximadamente {ate:.4f}.\n"
            f"Isso sugere que a variável de tratamento tem um efeito {'positivo' if ate > 0 else 'negativo'} no resultado, após controlar por {', '.join(common_causes)}."
        )
        
        return {"analysis_type": "Inferência Causal", "summary": summary_text, "estimated_effect": ate, "visualization_b64": None}

def salvar_relatorio_html(resultado: dict, nome_arquivo: str):
    """Salva o resultado de uma análise em um arquivo HTML simples para visualização."""
    titulo = resultado.get("analysis_type", "Relatório de Análise")
    resumo = resultado.get("summary", "Nenhum resumo disponível.")
    viz_b64 = resultado.get("visualization_b64")
    resumo_html = resumo.replace('\n', '<br>')
    imagem_html = f'<h2>Visualização</h2>\n<img src="data:image/png;base64,{viz_b64}" alt="Gráfico da Análise">' if viz_b64 else ""
    html_content = f"""
    <!DOCTYPE html><html lang="pt-BR"><head><meta charset="UTF-8"><title>{titulo}</title>
    <style>body{{font-family:sans-serif;margin:40px;}}.container{{max-width:800px;margin:auto;padding:20px;border:1px solid #ddd;border-radius:8px;}}img{{max-width:100%;}}</style>
    </head><body><div class="container"><h1>{titulo}</h1><h2>Resumo da Análise</h2><p>{resumo_html}</p>{imagem_html}</div></body></html>"""
    with open(nome_arquivo, "w", encoding="utf-8") as f: f.write(html_content)
    print(f"✅ Relatório salvo em: {os.path.abspath(nome_arquivo)}")

if __name__ == "__main__":
    # 1. Carregar e preparar os dados reais
    dados_reais = carregar_e_preparar_heart_disease_data()
    
    if dados_reais is not None:
        # 2. Inicializar o agente com os dados reais
        agente_estatistico = AgenteEstatistico(dados_reais)
        
        # 3. Executar Análise Preditiva
        # Vamos prever a doença cardíaca ('target') com base em outras variáveis clínicas
        resultado_preditivo = agente_estatistico.analisar(
            'preditiva',
            target_column='target',
            feature_columns=['age', 'sex', 'trestbps', 'chol', 'thalach']
        )
        salvar_relatorio_html(resultado_preditivo, "relatorio_preditivo_real.html")
        
        # 4. Executar Inferência Causal
        # Qual o efeito causal do sexo ('sex') na doença cardíaca ('target'), controlando pela idade ('age')?
        resultado_causal = agente_estatistico.analisar(
            'causal',
            treatment_column='sex',
            outcome_column='target',
            common_causes=['age']
        )
        salvar_relatorio_html(resultado_causal, "relatorio_causal_real.html")
