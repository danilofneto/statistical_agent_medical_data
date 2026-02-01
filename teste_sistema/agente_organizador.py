#
# Arquivo: agente_organizador.py
# Descrição: Agente central que interpreta os prompts do usuário e roteia para o agente especialista correto.
#
import json
from typing import Dict, Any

# Libs do LangChain para interagir com o modelo
from langchain_community.llms import Ollama

# Importe as classes dos seus agentes especialistas
# Supondo que cada agente está em seu próprio arquivo .py
# from agente_relatorios import AgenteDeRelatorios (Exemplo)
# from agente_estatistico import AgenteEstatistico (Exemplo)
# from agente_imagens import AgenteDeImagens (Exemplo)

print("Dependências importadas com sucesso.")

class AgenteOrganizador:
    """
    Agente que atua como um orquestrador, analisando o prompt do usuário
    e decidindo qual agente especialista deve ser acionado.
    """
    def __init__(self, model_name: str = "llama3"):
        """
        Inicializa o agente com um modelo de linguagem geral para raciocínio.
        Modelos como Llama 3 ou Mistral são boas escolhas para esta tarefa.

        :param model_name: O nome do modelo a ser usado via Ollama.
        """
        self.model_name = model_name
        try:
            self.llm = Ollama(model=self.model_name, temperature=0.0, format="json", base_url="http://127.0.0.1:11434")
            self.llm.invoke("Responda apenas com 'OK'")
            print(f"Agente Organizador inicializado com sucesso, conectado ao modelo '{self.model_name}'.")
        except Exception as e:
            print(f"ERRO: Não foi possível conectar ao Ollama. Verifique se ele está em execução e se o modelo '{self.model_name}' foi baixado (`ollama run {self.model_name}`).")
            exit()

        # Define as "ferramentas" que o organizador pode chamar.
        # Cada ferramenta corresponde a um agente especialista.
        self.tools_description = """
        [
            {
                "tool_name": "analise_de_imagem",
                "description": "Usado para responder perguntas sobre uma imagem médica específica. Ideal para tarefas de VQA (Visual Question Answering). Requer o caminho da imagem e uma pergunta.",
                "parameters": ["image_path", "question"]
            },
            {
                "tool_name": "analise_estatistica",
                "description": "Usado para realizar análises estatísticas, preditivas ou causais em um conjunto de dados. Requer um conjunto de dados e o tipo de análise (preditiva, causal, etc.).",
                "parameters": ["dataset", "analysis_type", "params"]
            },
            {
                "tool_name": "geracao_de_relatorio",
                "description": "Usado para criar um relatório clínico resumido a partir de dados estruturados (JSON) de um paciente.",
                "parameters": ["patient_data"]
            },
            {
                "tool_name": "conversa_geral",
                "description": "Usado para responder perguntas gerais, saudações ou quando nenhuma outra ferramenta é apropriada.",
                "parameters": ["question"]
            }
        ]
        """

    def _create_routing_prompt(self, user_prompt: str) -> str:
        """
        Cria o "meta-prompt" que instrui o LLM a atuar como um roteador.
        """
        prompt = f"""Você é um roteador inteligente em um sistema de IA para saúde. Sua tarefa é analisar a pergunta do usuário e decidir qual das seguintes ferramentas é a mais apropriada para respondê-la.
Sua resposta DEVE ser um único objeto JSON e nada mais.

**Ferramentas Disponíveis:**
{self.tools_description}

**Pergunta do Usuário:**
"{user_prompt}"

**Instruções:**
1.  Leia a pergunta do usuário com atenção.
2.  Escolha a `tool_name` mais adequada da lista de ferramentas.
3.  Extraia os parâmetros necessários da pergunta do usuário. Se um parâmetro não for mencionado, deixe seu valor como `null`.
4.  Retorne um único objeto JSON com a `tool_name` e um dicionário de `arguments`.

**Exemplo 1:**
Pergunta do Usuário: "Analise a imagem 'rx_torax.jpg' e me diga se há sinais de pneumonia."
Sua Resposta JSON:
{{
    "tool_name": "analise_de_imagem",
    "arguments": {{
        "image_path": "rx_torax.jpg",
        "question": "Há sinais de pneumonia?"
    }}
}}

**Exemplo 2:**
Pergunta do Usuário: "Com base nos dados dos pacientes, crie um modelo preditivo para risco cardiovascular."
Sua Resposta JSON:
{{
    "tool_name": "analise_estatistica",
    "arguments": {{
        "dataset": "dados_pacientes",
        "analysis_type": "preditiva",
        "params": {{"target_column": "risco_cardiovascular"}}
    }}
}}

**Exemplo 3:**
Pergunta do Usuário: "Olá, como você está?"
Sua Resposta JSON:
{{
    "tool_name": "conversa_geral",
    "arguments": {{
        "question": "Olá, como você está?"
    }}
}}

**Agora, analise a pergunta do usuário e forneça sua resposta em formato JSON.**
"""
        return prompt

    def rotear_prompt(self, user_prompt: str) -> Dict[str, Any]:
        """
        Ponto de entrada do agente. Recebe o prompt do usuário e retorna a decisão de roteamento.
        """
        print(f"\n>>> Agente Organizador recebendo o prompt: '{user_prompt}'")
        
        # 1. Criar o prompt de roteamento
        routing_prompt = self._create_routing_prompt(user_prompt)
        
        # 2. Chamar o LLM para obter a decisão
        print("   - Solicitando decisão de roteamento ao LLM...")
        try:
            response_text = self.llm.invoke(routing_prompt)
            decision = json.loads(response_text)
            print(f"   - Decisão recebida: {decision}")
            return decision
        except (json.JSONDecodeError, TypeError) as e:
            print(f"   - ERRO: O LLM não retornou um JSON válido. Resposta: {response_text}")
            return {
                "tool_name": "conversa_geral",
                "arguments": {"question": user_prompt, "error": "Falha no roteamento."}
            }
        except Exception as e:
            print(f"   - ERRO inesperado ao comunicar com o LLM: {e}")
            return {
                "tool_name": "conversa_geral",
                "arguments": {"question": user_prompt, "error": "Erro de comunicação com o LLM."}
            }

# --- FUNÇÃO DE EXECUÇÃO E DEMONSTRAÇÃO ---

if __name__ == "__main__":
    print("="*60)
    print("🚀 DEMONSTRAÇÃO DO AGENTE ORGANIZADOR")
    print("="*60)
    
    # 1. Inicializar o agente
    # Certifique-se de ter um modelo como Llama 3 rodando: `ollama run llama3`
    agente_organizador = AgenteOrganizador(model_name="llama3")
    
    # 2. Simular diferentes prompts de usuário
    prompts_de_teste = [
        "Por favor, analise a imagem em '/path/to/image/brain_mri.png' e verifique se há anomalias.",
        "Gere um relatório clínico para o paciente com os seguintes dados: {'id': 123, 'fc': 88, 'spo2': 97}",
        "Execute uma análise causal nos dados de pacientes para ver o efeito do tratamento na melhora clínica.",
        "Bom dia, qual a previsão do tempo para Guarapari?",
        "Qual o principal fator de risco para doença cardíaca nos dados fornecidos?",
        "Compare o raio-x 'case1.jpg' com 'case2.jpg'." # Um prompt mais complexo
    ]
    
    # 3. Executar o roteamento para cada prompt
    for prompt in prompts_de_teste:
        decisao_de_roteamento = agente_organizador.rotear_prompt(prompt)
        
        # Em um sistema real, você usaria um 'match' ou 'if/elif' aqui para chamar o agente real.
        # Exemplo:
        # if decisao_de_roteamento['tool_name'] == 'analise_de_imagem':
        #     agente_imagens.analisar_imagem(**decisao_de_roteamento['arguments'])
        # elif ...
        
        print("-" * 50)
        
    print("\nDemonstração concluída. O agente determinou a ferramenta e os parâmetros para cada prompt.")

