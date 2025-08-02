#
# Arquivo: agente_relatorios.py (Versão com importação corrigida)
# Descrição: Agente de IA que usa o LLM para texto e lógica Python para gráficos.
#
import json
import base64
from io import BytesIO
from typing import TypedDict, List, Dict

# Libs de Visualização
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt

# Libs do LangChain e LangGraph
from langchain_community.llms import Ollama
# CORREÇÃO: Removida a importação de 'Graph' que não é mais usada.
from langgraph.graph import StateGraph, END

print("Dependências do Agente de Relatórios importadas com sucesso.")

# --- 1. CONFIGURAÇÃO DO MODELO E PROMPTS ---

# Use um modelo que você tenha disponível, como 'medllama2' ou 'llama3'
OLLAMA_MODEL = "llama3" 
try:
    llm = Ollama(model=OLLAMA_MODEL, temperature=0.1)
    llm.invoke("Responda com 'OK' se estiver funcionando.")
    print(f"Agente de Relatórios conectado ao modelo '{OLLAMA_MODEL}' via Ollama.")
except Exception as e:
    print(f"ERRO: Não foi possível conectar ao Ollama. Verifique se ele está em execução e se o modelo '{OLLAMA_MODEL}' foi baixado (`ollama run {OLLAMA_MODEL}`).")
    exit()

# PROMPT REFORÇADO (Nó 1): Instruções mais diretas para o resumo narrativo.
MASTER_PROMPT_TEMPLATE = """ATENÇÃO: Sua resposta deve ser exclusivamente em Português do Brasil.

Você é um assistente de IA especialista em medicina. Sua única tarefa é gerar um resumo narrativo a partir dos dados clínicos em JSON abaixo. Siga estritamente a estrutura de formatação com os títulos "Resumo Geral", "Sinais Vitais", e "Resultados Laboratoriais".

**DADOS CLÍNICOS:**
```json
{clinical_data}
```

**RELATÓRIO GERADO:**
"""

# --- 2. DEFINIÇÃO DAS FERRAMENTAS DE VISUALIZAÇÃO ---

def create_interactive_line_chart(data: dict, title: str) -> str:
    """Cria um gráfico de linhas interativo com Plotly e retorna como HTML."""
    try:
        df = pd.DataFrame(data)
        x_axis = df.columns[0]
        y_axes = df.columns[1:]
        fig = px.line(df, x=x_axis, y=y_axes, title=title, markers=True, template="plotly_white")
        fig.update_layout(legend_title_text='Métricas')
        return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        return f"<p><i>Erro ao gerar gráfico de linhas: {e}</i></p>"

def create_static_bar_chart(data: dict, title: str) -> str:
    """Cria um gráfico de barras estático com Seaborn e retorna como uma tag <img> em base64."""
    try:
        df = pd.DataFrame(list(data.items()), columns=['Métrica', 'Valor'])
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x='Métrica', y='Valor')
        plt.title(title)
        plt.ylabel("Valor")
        plt.xticks(rotation=15, ha='right')
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close()
        data_b64 = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f'<img src="data:image/png;base64,{data_b64}" alt="{title}"/>'
    except Exception as e:
        return f"<p><i>Erro ao gerar gráfico de barras: {e}</i></p>"


# --- 3. DEFINIÇÃO DO AGENTE COM LANGGRAPH (ARQUITETURA SIMPLIFICADA) ---

class AgentState(TypedDict):
    clinical_data: Dict
    summary_text: str
    visualizations: List[str]
    final_report: str

def analysis_node(state: AgentState):
    """Nó 1: Analisa os dados e gera o resumo em texto com o LLM."""
    print(">>> [Agente de Relatórios - Nó 1] Analisando dados...")
    prompt = MASTER_PROMPT_TEMPLATE.format(clinical_data=json.dumps(state['clinical_data']))
    summary = llm.invoke(prompt)
    return {"summary_text": summary.strip()}

def deterministic_visualization_node(state: AgentState):
    """Nó 2: Gera gráficos de forma determinística com base na estrutura dos dados."""
    print(">>> [Agente de Relatórios - Nó 2] Gerando gráficos...")
    visualizations = []
    data = state.get("clinical_data", {})

    if "sinais_vitais" in data and isinstance(data["sinais_vitais"], dict):
        visualizations.append(create_interactive_line_chart(data["sinais_vitais"], "Evolução dos Sinais Vitais"))

    if "labs" in data and isinstance(data["labs"], dict):
        visualizations.append(create_static_bar_chart(data["labs"], "Resultados Laboratoriais"))

    return {"visualizations": visualizations}

def report_compilation_node(state: AgentState):
    """Nó 3: Compila o texto e as visualizações em um relatório HTML final."""
    print(">>> [Agente de Relatórios - Nó 3] Compilando relatório...")
    summary_html = "<br>".join(state.get('summary_text', '').splitlines())
    visualizations_html = "".join(state.get('visualizations', []))

    report_html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Relatório Clínico</title>
    <style>
        body {{ font-family: sans-serif; line-height: 1.6; margin: 0 auto; max-width: 900px; padding: 20px; }}
        h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px;}}
        .report-section {{ margin-bottom: 30px; padding: 20px; background-color: #f8f9f9; border: 1px solid #d5dbdb; border-radius: 8px; }}
        img {{ max-width: 100%; height: auto; display: block; margin: 20px auto; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        .plotly-graph-div {{ margin: 20px auto; }}
    </style>
</head>
<body>
    <h1>Relatório Clínico Gerado por IA</h1>
    <div class="report-section">
        <h2>Análise Narrativa</h2>
        <p>{summary_html}</p>
    </div>
    <div class="report-section">
        <h2>Visualizações de Dados</h2>
        {visualizations_html if visualizations_html else "<p><i>Nenhuma visualização foi gerada.</i></p>"}
    </div>
</body>
</html>"""
    return {"final_report": report_html}

# Construção do Grafo
workflow = StateGraph(AgentState)
workflow.add_node("analise", analysis_node)
workflow.add_node("geracao_determinista_viz", deterministic_visualization_node)
workflow.add_node("compilacao", report_compilation_node)

# Definição das Arestas
workflow.set_entry_point("analise")
workflow.add_edge("analise", "geracao_determinista_viz")
workflow.add_edge("geracao_determinista_viz", "compilacao")
workflow.add_edge("compilacao", END)

# Compilação do App
app = workflow.compile()
print("Grafo do Agente de Relatórios compilado com sucesso.")

# --- 4. FUNÇÃO PRINCIPAL PARA SER CHAMADA PELO main.py ---

def generate_clinical_report(patient_data: dict) -> str:
    """Função principal que encapsula o agente LangGraph."""
    initial_state = {"clinical_data": patient_data}
    final_state = app.invoke(initial_state)
    return final_state.get("final_report", "<h1>Erro: Não foi possível gerar o relatório.</h1>")

if __name__ == "__main__":
    # Este bloco serve apenas para testar o agente de forma isolada.
    print("Este script contém a função 'generate_clinical_report'.")
    print("Para testar o sistema completo, execute 'main.py'.")
