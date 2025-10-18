#
# Arquivo: agente_epidemiologico.py
# Descrição: Agente que responde a perguntas epidemiológicas usando RAG em dados locais
# e fallback para busca na web com Tavily.
#
import pandas as pd
import os
from typing import Dict, List, Any

# Ferramentas de busca e LLMs
from langchain_community.utilities import DuckDBLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

print("Dependências do Agente Epidemiológico importadas com sucesso.")

class AgenteEpidemiologico:
    """
    Agente especializado em responder perguntas epidemiológicas.
    1. Tenta responder usando uma base de dados local (simulando um RAG).
    2. Se não encontra uma resposta confiável, busca em fontes médicas na web com Tavily.
    """
    def __init__(self, dataset_path: str, llm_model: str = "gpt-4o"):
        """
        Inicializa o agente com um caminho para o dataset e um modelo de LLM.

        :param dataset_path: Caminho para o arquivo de dados (CSV).
        :param llm_model: Modelo da OpenAI para usar na síntese das respostas.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"O arquivo de dataset não foi encontrado em: {dataset_path}")
            
        self.dataset_path = dataset_path
        self.llm = ChatOpenAI(model=llm_model, temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        
        # Ferramenta 1: Buscador em Dados Locais (Simula o RAG)
        self.local_db_tool = self._criar_ferramenta_db_local()
        
        # Ferramenta 2: Buscador Web Confiável (Tavily)
        self.web_search_tool = TavilySearchResults(
            max_results=5,
            #
            # IMPORTANTE: Você pode adicionar um filtro de busca para focar em sites confiáveis.
            # search_depth="advanced" # pode ser usado para pesquisas mais profundas
        )
        
        # Cria o agente ReAct que decide qual ferramenta usar
        self.agent_executor = self._criar_agente_executor()
        print("Agente Epidemiológico inicializado com sucesso.")

    def _criar_ferramenta_db_local(self):
        """
        Cria uma ferramenta que permite fazer perguntas em linguagem natural (SQL)
        a um banco de dados DuckDB carregado a partir do CSV.
        """
        print(f"Carregando dados de '{self.dataset_path}' para o DuckDB...")
        loader = DuckDBLoader(self.dataset_path, query=f"SELECT * FROM '{self.dataset_path}'")
        db = loader.load()
        
        def run_query(query: str) -> str:
            """Executa uma consulta SQL no banco de dados local."""
            try:
                # O DuckDBLoader já oferece uma forma de executar queries.
                # Para uma implementação mais robusta, usaríamos o duckdb diretamente.
                # Esta é uma simplificação para o agente.
                # Vamos carregar os dados em um DataFrame para a consulta.
                df = pd.read_csv(self.dataset_path)
                result = df.query(query)
                return f"Resultado da consulta local:\n{result.to_markdown()}"
            except Exception as e:
                return f"Erro ao executar a consulta local: {e}. Tente uma sintaxe de query do Pandas."

        return Tool(
            name="BuscadorDeDadosEpidemiologicosLocais",
            func=run_query,
            description="""
            Use esta ferramenta para responder perguntas que podem ser resolvidas com os dados de saúde locais.
            A entrada deve ser uma query no formato do método `query` do Pandas (ex: 'age > 50 and diabetes == 1').
            NÃO use SQL completo. Use apenas para perguntas sobre prevalência, contagens ou estatísticas
            descritivas que estão nos dados. Se a pergunta for sobre conhecimento médico geral,
            não use esta ferramenta.
            """
        )

    def _criar_agente_executor(self):
        """
        Cria o agente principal que orquestra o uso das ferramentas.
        """
        tools = [self.local_db_tool, self.web_search_tool]
        
        # O prompt ReAct instrui o agente sobre como pensar e qual ferramenta escolher.
        prompt_template = """
        Você é um assistente de pesquisa epidemiológica de alto nível. Responda à seguinte pergunta da forma mais precisa possível, usando as ferramentas disponíveis.
        Sua principal prioridade é usar o `BuscadorDeDadosEpidemiologicosLocais` primeiro. Se e somente se essa ferramenta não fornecer uma resposta adequada,
        ou se a pergunta for sobre conhecimento médico geral que não estaria nos dados locais (como "quais são os sintomas de..."),
        use a `tavily_search_results` para pesquisar em fontes confiáveis.
        Ao usar a `tavily_search_results`, você pode refinar sua busca para sites confiáveis como `site:who.int`, `site:cdc.gov`, `site:thelancet.com`.

        Ferramentas disponíveis:
        {tools}

        Use o seguinte formato:

        Pergunta: a pergunta de entrada que você precisa responder
        Pensamento: você deve sempre pensar sobre o que fazer.
        Ação: a ação a ser tomada, deve ser uma de [{tool_names}]
        Entrada da Ação: a entrada para a ação
        Observação: o resultado da ação
        ... (este Pensamento/Ação/Entrada/Observação pode se repetir N vezes)
        Pensamento: Agora eu sei a resposta final
        Resposta Final: a resposta final para a pergunta original. Sempre cite suas fontes (seja 'dados locais' ou uma URL da web).

        Comece!

        Pergunta: {input}
        Pensamento:{agent_scratchpad}
        """
        
        prompt = PromptTemplate.from_template(prompt_template)
        
        agent = create_react_agent(self.llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    def analisar(self, pergunta: str) -> Dict[str, Any]:
        """
        Ponto de entrada principal para analisar uma pergunta epidemiológica.
        """
        print(f"\n>>> [Agente Epidemiológico] Recebendo pergunta: '{pergunta}'")
        try:
            resultado = self.agent_executor.invoke({"input": pergunta})
            return {
                "pergunta": pergunta,
                "resposta": resultado.get("output", "Não foi possível obter uma resposta."),
                "fonte": "Híbrida (Dados Locais e/ou Web)"
            }
        except Exception as e:
            print(f"Ocorreu um erro durante a análise epidemiológica: {e}")
            return {
                "pergunta": pergunta,
                "resposta": "Ocorreu um erro ao processar sua pergunta.",
                "error": str(e)
            }


if __name__ == '__main__':
    # --- Demonstração de Uso ---
    
    # Verifique se as chaves de API estão configuradas
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        print("ERRO: Configure suas variáveis de ambiente OPENAI_API_KEY e TAVILY_API_KEY.")
        exit()

    # 1. Preparar um dataset de exemplo (iremos criar um sintético para facilitar)
    print("Criando dataset epidemiológico sintético de exemplo...")
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000, n_features=5, n_informative=3, n_classes=2, random_state=42
    )
    df = pd.DataFrame(X, columns=['age', 'bmi', 'blood_pressure', 'cholesterol', 'genetic_marker'])
    df['diabetes'] = y
    df['age'] = (df['age'] * 15 + 50).astype(int) # Escala de idade para algo realista
    
    dataset_filename = "synthetic_diabetes_data.csv"
    df.to_csv(dataset_filename, index=False)
    print(f"Dataset salvo como '{dataset_filename}'.")
    
    # 2. Inicializar o agente
    agente = AgenteEpidemiologico(dataset_path=dataset_filename)
    
    # 3. Fazer perguntas
    
    # Pergunta 1: Pode ser respondida pelos dados locais
    print("\n" + "="*50)
    pergunta_local = "Qual é a idade média dos pacientes com diabetes no dataset?"
    # Para o nosso buscador simplificado, precisamos de uma query do Pandas
    pergunta_local_query = "diabetes == 1" # A query real seria 'age[diabetes == 1].mean()'
    # Vamos adaptar a pergunta para o nosso agente de query
    pergunta_local_adaptada = "Me dê os dados para pacientes onde 'diabetes == 1' para que eu possa calcular a idade média."
    resultado1 = agente.analisar(pergunta_local_adaptada)
    print("\n--- RESULTADO 1 (Busca Local) ---")
    print(f"Pergunta: {resultado1['pergunta']}")
    print(f"Resposta: {resultado1['resposta']}")
    print("="*50)

    # Pergunta 2: Requer conhecimento externo (fallback para Tavily)
    print("\n" + "="*50)
    pergunta_externa = "Quais são as últimas diretrizes da OMS para o tratamento de diabetes tipo 2?"
    resultado2 = agente.analisar(pergunta_externa)
    print("\n--- RESULTADO 2 (Busca Externa) ---")
    print(f"Pergunta: {resultado2['pergunta']}")
    print(f"Resposta: {resultado2['resposta']}")
    print("="*50)
