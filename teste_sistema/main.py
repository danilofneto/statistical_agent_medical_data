#
# Arquivo: main.py
# Descrição: Script principal para testar a integração e orquestração de todos os agentes.
#
import pandas as pd
import numpy as np
import time
from datetime import datetime

# --- Importação dos Agentes ---
# Importe as classes principais de cada um dos seus arquivos de agente.
# Se os arquivos tiverem nomes diferentes, ajuste as importações.

from agente_organizador import AgenteOrganizador
# Para os agentes especialistas, vamos criar Mocks (simulações) para este teste,
# para que o script funcione de forma independente. Você pode substituir pelos seus imports reais.

# from agente_imagens import AgenteDeImagens
# from agente_estatistico import AgenteEstatistico
# from agente_iot import AgenteIoT, simular_dispositivo_iot

print("Dependências importadas com sucesso.")


# --- Mocks (Simulações) dos Agentes Especialistas ---
# Estes mocks simulam o comportamento dos seus agentes reais.
# Isso nos permite testar o Agente Organizador sem depender 100% dos outros scripts.

class MockAgenteDeImagens:
    def analisar_imagem(self, image_path: str, question: str):
        print(f"--- [MOCK Agente de Imagens] Análise solicitada para '{image_path}' ---")
        return {
            "image_path": image_path,
            "question": question,
            "answer": f"Análise simulada da imagem '{image_path}': A imagem parece mostrar achados consistentes com a pergunta '{question}'."
        }

class MockAgenteEstatistico:
    def __init__(self, data):
        self.data = data
    def analisar(self, analysis_type: str, **kwargs):
        print(f"--- [MOCK Agente Estatístico] Análise '{analysis_type}' solicitada ---")
        return {
            "analysis_type": analysis_type,
            "summary": f"Resultado da análise estatística simulada do tipo '{analysis_type}'. Parâmetros recebidos: {kwargs}",
            "visualization_b64": "simulated_base64_string"
        }

class MockAgenteIoT:
    def __init__(self, paciente_id, limiares):
        self.paciente_id = paciente_id
        self.limiares = limiares
    def monitorar_ponto_de_dados(self, ponto_de_dados):
        print(f"--- [MOCK Agente IoT] Monitorando dados: {ponto_de_dados} ---")
        # Simula um alerta ocasional
        if ponto_de_dados['heart_rate_bpm'] > self.limiares['hr_max']:
            return {"status": "alerta", "motivo": "Taquicardia detectada."}
        return {"status": "normal"}


class SistemaMultiAgente:
    """
    Classe principal que inicializa e orquestra todos os agentes.
    """
    def __init__(self):
        print("="*60)
        print("🚀 Inicializando o Sistema Multiagente...")
        print("="*60)
        
        # 1. Inicializar o cérebro do sistema
        self.agente_organizador = AgenteOrganizador(model_name="llama3")
        
        # 2. Inicializar os agentes especialistas (usando os Mocks para este teste)
        # Em produção, você inicializaria suas classes reais aqui.
        self.agente_imagens = MockAgenteDeImagens()
        
        # O agente estatístico precisa de dados, vamos criar um mock
        mock_data = pd.DataFrame(np.random.rand(100, 4), columns=['idade', 'bmi', 'tratamento', 'resultado'])
        self.agente_estatistico = MockAgenteEstatistico(data=mock_data)
        
        self.agente_iot = MockAgenteIoT("Paciente-Demo", {'hr_max': 100})
        
        print("\n✅ Todos os agentes foram inicializados com sucesso.")

    def processar_solicitacao(self, prompt_usuario: str):
        """
        Processa uma única solicitação do usuário, desde o roteamento até a execução.
        """
        # 1. O Agente Organizador decide o que fazer
        decisao = self.agente_organizador.rotear_prompt(prompt_usuario)
        
        tool_name = decisao.get("tool_name")
        arguments = decisao.get("arguments", {})
        
        resultado_final = None
        
        print(f"\n>>> Executando a ferramenta decidida: '{tool_name}'")
        
        # 2. O sistema executa a ferramenta (agente) apropriada
        if tool_name == "analise_de_imagem":
            if "image_path" in arguments and "question" in arguments:
                resultado_final = self.agente_imagens.analisar_imagem(**arguments)
            else:
                resultado_final = {"error": "Parâmetros 'image_path' e 'question' são necessários."}

        elif tool_name == "analise_estatistica":
            # Em um sistema real, o 'dataset' seria carregado aqui
            resultado_final = self.agente_estatistico.analisar(**arguments)

        elif tool_name == "geracao_de_relatorio":
            # Este agente ainda não foi implementado como mock, vamos simular
            print("--- [MOCK Agente de Relatórios] ---")
            resultado_final = {"relatorio": f"Relatório simulado gerado com os dados: {arguments.get('patient_data')}"}
        
        elif tool_name == "conversa_geral":
            resultado_final = {"resposta": "Sou um sistema de IA focado em saúde. Como posso ajudar com análises de imagens ou dados?"}
            
        else:
            resultado_final = {"error": f"Ferramenta desconhecida ou decisão de roteamento inválida: '{tool_name}'"}
            
        # 3. Exibir o resultado final da execução
        print("\n--- RESULTADO FINAL DA SOLICITAÇÃO ---")
        print(json.dumps(resultado_final, indent=2, ensure_ascii=False))
        print("-" * 60)


if __name__ == "__main__":
    sistema = SistemaMultiAgente()
    
    # Lista de prompts para testar todas as funcionalidades do sistema
    prompts_de_teste = [
        "Analise a imagem em 'data/images/chest_xray_01.png' e procure por nódulos pulmonares.",
        "Execute uma análise preditiva para 'risco_diabetes' usando o dataset de pacientes.",
        "Qual o efeito causal do 'medicamento_X' no 'desfecho_Y'?",
        "Gere um relatório para o paciente João Silva, idade 45, FC 78bpm.",
        "Olá, tudo bem?",
        "O que é inferência causal?" # Deve ser roteado para conversa geral
    ]
    
    for prompt in prompts_de_teste:
        sistema.processar_solicitacao(prompt)
        time.sleep(2) # Pausa para facilitar a leitura dos logs

