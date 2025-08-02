#
# Arquivo: main.py
# Descrição: Script principal corrigido para testar a integração de todos os agentes.
#
import pandas as pd
import numpy as np
import time
import json
import os

# --- Importação dos Agentes Reais e Mocks ---
from agente_organizador import AgenteOrganizador

# Importa o Agente Estatístico REAL
# Certifique-se de que o nome do arquivo é 'agente_estatistico.py'
from agente_estatistico import AgenteEstatistico, salvar_relatorio_html

# Importa o Agente de Relatórios REAL
# Certifique-se de que o nome do arquivo é 'agente_relatorios.py'
from agente_relatorios import generate_clinical_report

print("Dependências e agentes importados com sucesso.")


# --- Mocks (Simulações) para agentes ainda não integrados ---
class MockAgenteDeImagens:
    def analisar_imagem(self, image_path: str, question: str):
        print(f"--- [MOCK Agente de Imagens] Análise solicitada para '{image_path}' ---")
        return {
            "image_path": image_path,
            "question": question,
            "answer": f"Análise simulada da imagem '{image_path}': A imagem parece mostrar achados consistentes com a pergunta '{question}'."
        }


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
        
        # 2. Inicializar os agentes especialistas
        self.agente_imagens = MockAgenteDeImagens()
        
        # O agente estatístico precisa de dados para ser inicializado
        dados_simulados = pd.DataFrame({
            'idade': np.random.normal(55, 18, 1000),
            'bmi': np.random.normal(26, 5, 1000),
            'smoking': np.random.binomial(1, 0.2, 1000),
            'treatment': np.random.binomial(1, 0.5, 1000),
            'cardiovascular_event': np.random.binomial(1, 0.3, 1000)
        })
        self.agente_estatistico = AgenteEstatistico(data=dados_simulados)
        
        print("\n✅ Todos os agentes foram inicializados com sucesso.")

    def processar_solicitacao(self, prompt_usuario: str):
        """
        Processa uma única solicitação do usuário, desde o roteamento até a execução.
        """
        decisao = self.agente_organizador.rotear_prompt(prompt_usuario)
        
        tool_name = decisao.get("tool_name")
        arguments = decisao.get("arguments", {})
        
        resultado_final = None
        
        print(f"\n>>> Executando a ferramenta decidida: '{tool_name}'")
        
        if tool_name == "analise_de_imagem":
            resultado_final = self.agente_imagens.analisar_imagem(**arguments)

        elif tool_name == "analise_estatistica":
            analysis_type = arguments.get("analysis_type")
            params = arguments.get("params", {})
            
            # *** CORREÇÃO APLICADA AQUI ***
            # Alinha o nome do parâmetro: o organizador envia 'features', mas o estatístico espera 'feature_columns'.
            if 'features' in params:
                params['feature_columns'] = params.pop('features')
            
            if analysis_type and params:
                # Chama o agente estatístico real com os argumentos corrigidos e desempacotados
                resultado_analise = self.agente_estatistico.analisar(analysis_type=analysis_type, **params)
                
                # Gera um relatório HTML para a análise estatística
                nome_arquivo = f"relatorio_{analysis_type}.html"
                salvar_relatorio_html(resultado_analise, nome_arquivo)
                resultado_final = {"status": "Análise estatística concluída.", "relatorio_salvo_em": nome_arquivo}
            else:
                resultado_final = {"error": "Parâmetros 'analysis_type' e 'params' são necessários para a análise estatística."}

        elif tool_name == "geracao_de_relatorio":
            print("--- [REAL Agente de Relatórios] Gerando relatório clínico... ---")
            dados_paciente = arguments.get("patient_data", {})
            if dados_paciente:
                html_report = generate_clinical_report(dados_paciente)
                nome_arquivo = "relatorio_clinico_gerado.html"
                with open(nome_arquivo, "w", encoding="utf-8") as f:
                    f.write(html_report)
                resultado_final = {"status": "Relatório clínico gerado com sucesso.", "relatorio_salvo_em": nome_arquivo}
            else:
                resultado_final = {"error": "Dados do paciente não foram fornecidos."}
        
        elif tool_name == "conversa_geral":
            resultado_final = {"resposta": "Sou um sistema de IA focado em saúde. Como posso ajudar com análises de imagens ou dados?"}
            
        else:
            resultado_final = {"error": f"Ferramenta desconhecida ou decisão de roteamento inválida: '{tool_name}'"}
            
        print("\n--- RESULTADO FINAL DA SOLICITAÇÃO ---")
        print(json.dumps(resultado_final, indent=2, ensure_ascii=False))
        print("-" * 60)


if __name__ == "__main__":
    sistema = SistemaMultiAgente()
    
    prompts_de_teste = [
        "Gere um relatório para o paciente com os seguintes dados: {'paciente_id': 'P-999', 'sinais_vitais': {'hora': [0, 4, 8], 'fc_bpm': [70, 75, 72]}, 'labs': {'glicose': 98, 'colesterol': 210}}",
        "Analise a imagem em 'data/images/chest_xray_01.png' e procure por nódulos pulmonares.",
        "Execute uma análise preditiva para 'cardiovascular_event' usando as features 'idade', 'bmi' e 'smoking'.",
        "Olá, tudo bem?"
    ]
    
    for prompt in prompts_de_teste:
        sistema.processar_solicitacao(prompt)
        time.sleep(2)
