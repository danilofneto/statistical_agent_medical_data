#
# Arquivo: agente_epidemiologico.py
# Descrição: Agente com RAG vetorial (FAISS) e RAG estruturado (Pandas/DuckDB).
#
import pandas as pd
import os
from typing import Dict, List, Any

# Ferramentas de busca, LLMs e RAG
from langchain_community.utilities import DuckDBLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

print("Dependências do Agente Epidemiológico importadas com sucesso.")

class AgenteEpidemiologico:
    """
    Agente especializado em responder perguntas epidemiológicas usando três ferramentas:
    1. RAG Estruturado (Pandas): Para consultas em tabelas locais (ex: 'qual a média de idade?')
    2. RAG Vetorial (FAISS): Para perguntas conceituais em artigos (ex: 'quais as diretrizes?')
    3. Busca Externa (Tavily): Para informações da web em tempo real.
    """
    def __init__(self, dataset_path: str, vector_index_path: str, llm_model: str = "gpt-4o"):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"O arquivo de dataset (CSV) não foi encontrado em: {dataset_path}")
        if not os.path.exists(vector_index_path):
            raise FileNotFoundError(f"O índice vetorial FAISS não foi encontrado em: {vector_index_path}. Execute 'build_index.py' primeiro.")
            
        self.dataset_path = dataset_path
        self.vector_index_path = vector_index_path
        self.llm = ChatOpenAI(model=llm_model, temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        
        # Ferramenta 1: Buscador em Dados Locais (CSV)
        self.local_db_tool = self._criar_ferramenta_db_local()
        
        # Ferramenta 2: RAG Vetorial (FAISS)
        self.rag_tool = self._criar_ferramenta_rag_vetorial()
        
        # Ferramenta 3: Buscador Web Confiável (Tavily)
        self.web_search_tool = TavilySearchResults(max_results=3)
        
        # Cria o agente ReAct que decide qual ferramenta usar
        self.agent_executor = self._criar_agente_executor()
        print("Agente Epidemiológico (com RAG vetorial) inicializado com sucesso.")

    def _criar_ferramenta_db_local(self):
        """Cria a ferramenta para consultar dados estruturados (CSV/Pandas)."""
        print(f"Carregando dados de '{self.dataset_path}' para a ferramenta de dados locais...")
        
        def run_query(query: str) -> str:
            """Executa uma consulta no formato Pandas `query` no CSV."""
            try:
                df = pd.read_csv(self.dataset_path)
                result = df.query(query)
                return f"Resultado da consulta local:\n{result.to_markdown()}"
            except Exception as e:
                return f"Erro ao executar a consulta local: {e}. Tente uma sintaxe de query do Pandas."

        return Tool(
            name="BuscadorDeDadosLocais",
            func=run_query,
            description="""
            Use esta ferramenta para perguntas estatísticas sobre dados locais estruturados (CSV).
            A entrada deve ser uma query no formato do método `query` do Pandas (ex: 'age > 50 and diabetes == 1').
            Use apenas para perguntas sobre contagens, médias, prevalências, etc., que estão no CSV.
            """
        )

    def _criar_ferramenta_rag_vetorial(self):
        """Cria a ferramenta de RAG que busca em documentos (índice FAISS)."""
        print(f"Carregando índice vetorial FAISS de '{self.vector_index_path}'...")
        try:
            # Carrega o modelo de embedding
            model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
            embeddings = SentenceTransformerEmbeddings(model_name=model_name)
            
            # Carrega o índice FAISS salvo
            db = FAISS.load_local(self.vector_index_path, embeddings, allow_dangerous_deserialization=True)
            retriever = db.as_retriever(search_kwargs={"k": 3}) # Retorna os 3 chunks mais relevantes
            
            def run_rag_query(query: str) -> str:
                """Executa uma busca semântica no índice vetorial."""
                docs = retriever.invoke(query)
                return f"Contexto encontrado nos documentos locais:\n" + "\n---\n".join([doc.page_content for doc in docs])

            return Tool(
                name="BuscadorDeArtigosMedicos",
                func=run_rag_query,
                description="""
                Use esta ferramenta para perguntas conceituais ou sobre diretrizes médicas,
                como 'quais são os sintomas de...', 'qual o tratamento para...', 
                'o que diz a diretriz sobre...'.
                Busca em artigos médicos e diretrizes salvas localmente.
                """
            )
        except Exception as e:
            print(f"ERRO CRÍTICO ao carregar o índice FAISS: {e}")
            print("Execute 'build_index.py' para criar o índice.")
            # Retorna uma ferramenta "vazia" para não quebrar o agente
            return Tool(
                name="BuscadorDeArtigosMedicos",
                func=lambda q: "Erro: Índice FAISS não carregado.",
                description="Erro: Índice FAISS não carregado."
            )

    def _criar_agente_executor(self):
        """
        Cria o agente principal que orquestra o uso das ferramentas.
        """
        tools = [self.local_db_tool, self.rag_tool, self.web_search_tool]
        
        prompt_template = """
        Você é um assistente de pesquisa epidemiológica de alto nível. Responda à seguinte pergunta da forma mais precisa possível.
        Você tem três ferramentas à sua disposição:

        1. `BuscadorDeDadosLocais`: Use para perguntas estatísticas sobre dados locais (ex: 'qual a média de idade?', 'quantos pacientes fumam?'). A entrada deve ser uma query Pandas.
        2. `BuscadorDeArtigosMedicos`: Use para perguntas conceituais ou sobre diretrizes (ex: 'quais os sintomas de...', 'qual o tratamento recomendado para...').
        3. `tavily_search_results`: Use como último recurso se as ferramentas locais não fornecerem uma resposta, ou para informações muito recentes (ex: 'notícias de hoje').

        **Prioridade:** Tente sempre usar `BuscadorDeDadosLocais` ou `BuscadorDeArtigosMedicos` primeiro.

        Use o seguinte formato:

        Pergunta: a pergunta de entrada que você precisa responder
        Pensamento: Devo analisar a pergunta. É uma consulta estatística (BuscadorDeDadosLocais), uma consulta conceitual (BuscadorDeArtigosMedicos) ou uma busca geral (tavily)?
        Ação: a ação a ser tomada, deve ser uma de [{tool_names}]
        Entrada da Ação: a entrada para a ação
        Observação: o resultado da ação
        ... (este Pensamento/Ação/Entrada/Observação pode se repetir N vezes)
        Pensamento: Agora eu sei a resposta final
        Resposta Final: a resposta final para a pergunta original. Sempre cite sua fonte (seja 'Dados Locais', 'Artigos Médicos Locais' ou uma URL da web).

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
                "fonte": "Híbrida (IA + Ferramentas)"
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

    # 1. Preparar os dados (Execute build_index.py primeiro!)
    dataset_filename = "synthetic_diabetes_data.csv"
    index_folder = "faiss_index"

    if not os.path.exists(index_folder) or not os.path.exists(dataset_filename):
        print("ERRO: Arquivos de dados ou índice não encontrados.")
        print("Por favor, execute 'build_index.py' e o script anterior para criar 'synthetic_diabetes_data.csv' primeiro.")
        exit()
    
    # 2. Inicializar o agente
    agente = AgenteEpidemiologico(dataset_path=dataset_filename, vector_index_path=index_folder)
    
    # 3. Fazer perguntas de teste
    
    # Pergunta 1: Teste do RAG Estruturado (Pandas)
    print("\n" + "="*50)
    pergunta_local = "Quantos pacientes com idade acima de 60 anos (age > 60) têm diabetes (diabetes == 1)?"
    resultado1 = agente.analisar(pergunta_local)
    print("\n--- RESULTADO 1 (Busca Estruturada) ---")
    print(f"Resposta: {resultado1['resposta']}")
    print("="*50)

    # Pergunta 2: Teste do RAG Vetorial (FAISS)
    print("\n" + "="*50)
    pergunta_conceitual = "O que este documento diz sobre o tratamento de diabetes?"
    resultado2 = agente.analisar(pergunta_conceitual)
    print("\n--- RESULTADO 2 (RAG Vetorial) ---")
    print(f"Resposta: {resultado2['resposta']}")
    print("="*50)

    # Pergunta 3: Teste da Busca Externa (Tavily)
    print("\n" + "="*50)
    pergunta_externa = "Quais são as últimas notícias sobre a cura do diabetes tipo 1? site:cdc.gov"
    resultado3 = agente.analisar(pergunta_externa)
    print("\n--- RESULTADO 3 (Busca Externa) ---")
    print(f"Resposta: {resultado3['resposta']}")
    print("="*50)
