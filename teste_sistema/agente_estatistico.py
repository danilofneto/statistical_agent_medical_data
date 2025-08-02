#
# Arquivo: agente_estatistico.py
# Descrição: Versão final e limpa, pronta para ser importada pelo main.py
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

class AgenteEstatistico:
    """
    Um agente de IA que realiza diferentes tipos de análises estatísticas
    em um determinado conjunto de dados.
    """
    def __init__(self, data: pd.DataFrame):
        """
        Inicializa o agente com o conjunto de dados.
        :param data: Um DataFrame do Pandas com os dados a serem analisados.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Os dados de entrada devem ser um DataFrame do Pandas.")
        # Lida com valores faltantes de forma simples para garantir que os modelos rodem
        self.data = data.dropna().copy()
        print(f"Agente Estatístico inicializado. Dados carregados com {self.data.shape[0]} linhas (após remover valores faltantes).")

    def analisar(self, analysis_type: str, **kwargs):
        """
        Ponto de entrada principal para realizar uma análise.
        Funciona como um dispatcher que chama o método de análise apropriado.
        """
        print(f"\n>>> [Agente Estatístico] Iniciando análise do tipo: '{analysis_type}'")
        if analysis_type == 'preditiva':
            return self._run_predictive_analysis(**kwargs)
        elif analysis_type == 'causal':
            return self._run_causal_analysis(**kwargs)
        elif analysis_type == 'probabilistica':
            return self._run_probabilistic_analysis(**kwargs)
        else:
            return {
                "error": f"Tipo de análise desconhecido: {analysis_type}",
                "summary": "Nenhuma análise foi executada.",
                "visualization_b64": None
            }

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
        plt.title("XAI: Importância das Variáveis no Modelo Preditivo")
        plt.ylabel("Coeficiente da Regressão Logística")
        plt.xticks(rotation=45, ha='right')
        
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        viz_base64 = base64.b64encode(buf.getbuffer()).decode("ascii")
        plt.close()
        
        summary_text = (
            f"Modelo de Regressão Logística treinado para prever '{target_column}'.\n"
            f"As variáveis mais influentes (positiva ou negativamente) foram: {', '.join(importances.index)}.\n"
            "O gráfico mostra o peso de cada variável na decisão do modelo."
        )
        
        return {
            "analysis_type": "Preditiva (com XAI)",
            "summary": summary_text,
            "visualization_b64": viz_base64
        }

    def _run_causal_analysis(self, treatment_column: str, outcome_column: str, common_causes: list):
        """Realiza uma análise de inferência causal usando DoWhy."""
        print("   - Construindo modelo causal com DoWhy...")
        model = CausalModel(
            data=self.data,
            treatment=treatment_column,
            outcome=outcome_column,
            common_causes=common_causes
        )
        identified_estimand = model.identify_effect()
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.propensity_score_matching"
        )
        ate = estimate.value
        
        summary_text = (
            f"Análise de Inferência Causal concluída.\n"
            f"Efeito causal médio do tratamento ('{treatment_column}') sobre o resultado ('{outcome_column}') "
            f"é de aproximadamente {ate:.4f}.\n"
            f"Isso sugere que o tratamento tem um efeito {'positivo' if ate > 0 else 'negativo'} no resultado, "
            "após controlar pelas variáveis de confusão."
        )
        
        return {
            "analysis_type": "Inferência Causal",
            "summary": summary_text,
            "estimated_effect": ate,
            "visualization_b64": None # Análise causal não gera gráfico neste exemplo
        }

    def _run_probabilistic_analysis(self, target_column: str, predictor_columns: list):
        """
        Realiza uma análise probabilística usando PyMC (Regressão Logística Bayesiana).
        """
        print("   - Construindo modelo Bayesiano com PyMC...")
        
        with pm.Model() as probabilistic_model:
            intercept = pm.Normal("intercept", mu=0, sigma=10)
            betas = pm.Normal("betas", mu=0, sigma=10, shape=len(predictor_columns))
            predictors_data = self.data[predictor_columns].values
            logit_p = intercept + pt.dot(predictors_data, betas)
            y_obs = pm.Bernoulli(target_column, logit_p=logit_p, observed=self.data[target_column].values)
            trace = pm.sample(2000, tune=1000, cores=1, random_seed=42)
            
        az.plot_posterior(trace, var_names=['betas'])
        plt.suptitle("Distribuições a Posteriori dos Coeficientes do Modelo", y=1.02)
        
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        viz_base64 = base64.b64encode(buf.getbuffer()).decode("ascii")
        plt.close()
        
        summary_text = (
            "Modelo de Regressão Logística Bayesiana treinado.\n"
            f"O gráfico mostra as distribuições de credibilidade para os coeficientes das variáveis: {', '.join(predictor_columns)}.\n"
            "Isso nos permite quantificar a incerteza sobre o efeito de cada variável."
        )
        
        return {
            "analysis_type": "Programação Probabilística",
            "summary": summary_text,
            "visualization_b64": viz_base64
        }

def salvar_relatorio_html(resultado: dict, nome_arquivo: str):
    """
    Salva o resultado de uma análise em um arquivo HTML simples para visualização.
    Esta função é exportada para ser usada pelo main.py.
    """
    titulo = resultado.get("analysis_type", "Relatório de Análise")
    resumo = resultado.get("summary", "Nenhum resumo disponível.")
    viz_b64 = resultado.get("visualization_b64")

    resumo_html = resumo.replace('\n', '<br>')
    imagem_html = f'<h2>Visualização</h2>\n<img src="data:image/png;base64,{viz_b64}" alt="Gráfico da Análise">' if viz_b64 else ""

    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <title>{titulo}</title>
        <style>
            body {{ font-family: sans-serif; line-height: 1.6; margin: 40px; }}
            .container {{ max-width: 800px; margin: auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{titulo}</h1>
            <h2>Resumo da Análise</h2>
            <p>{resumo_html}</p>
            {imagem_html}
        </div>
    </body>
    </html>
    """
    
    with open(nome_arquivo, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✅ Relatório de análise estatística salvo em: {os.path.abspath(nome_arquivo)}")


if __name__ == "__main__":
    # Este bloco serve apenas para testar o agente de forma isolada.
    # O script principal que orquestra tudo é o main.py.
    print("Este script contém a classe 'AgenteEstatistico'.")
    print("Para testar o sistema completo, execute 'main.py'.")

