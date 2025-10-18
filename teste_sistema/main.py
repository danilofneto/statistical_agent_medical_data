#
# Arquivo: main.py
# Descrição: Versão final com depuração e tratamento de erros aprimorados.
#
import pandas as pd
import numpy as np
import time
import json
import os

# --- Importação dos Agentes Reais ---
from agente_organizador import AgenteOrganizador
from agente_estatistico import AgenteEstatistico, salvar_relatorio_html
from agente_relatorios import generate_clinical_report
from agente_imagens import AgenteDeImagens 

print("Dependências e agentes importados com sucesso.")

class SistemaMultiAgente:
    """
    Classe principal que inicializa e orquestra todos os agentes.
    """
    def __init__(self):
        print("="*60)
        print("🚀 Inicializando o Sistema MultiAgente...")
        print("="*60)
        
        self.agente_organizador = AgenteOrganizador(model_name="llama3")
        
        # *** MUDANÇA PRINCIPAL: USA O AGENTE DE IMAGENS REAL ***
        #self.agente_imagens = AgenteDeImagens(model_name="alibayram/medgemma:4b")
        #self.agente_imagens = AgenteDeImagens(model_name="llava:7b")  # Usando o modelo Llava 7B como exemplo
        self.agente_imagens = AgenteDeImagens(model_name="medllama2:7b-q5_1")  # Usando o modelo Llava 7B como exemplo
        dados_simulados = pd.DataFrame({
            'idade': np.random.normal(55, 18, 1000), 'bmi': np.random.normal(26, 5, 1000),
            'smoking': np.random.binomial(1, 0.2, 1000), 'treatment': np.random.binomial(1, 0.5, 1000),
            'cardiovascular_event': np.random.binomial(1, 0.3, 1000)
        })
        self.agente_estatistico = AgenteEstatistico(data=dados_simulados)
        
        print("\n✅ Todos os agentes foram inicializados com sucesso.")

    def processar_solicitacao(self, prompt_usuario: str, caminho_imagem: str = None):
        """
        Processa uma única solicitação do usuário, desde o roteamento até a execução.
        """
        try:
            decisao = self.agente_organizador.rotear_prompt(prompt_usuario)
            
            tool_name = decisao.get("tool_name")
            arguments = decisao.get("arguments", {})
            
            # Se um caminho de imagem foi passado pelo backend, use-o
            if caminho_imagem and 'image_path' not in arguments:
                 arguments['image_path'] = caminho_imagem

            resultado_final = None
            
            print(f"\n>>> Executando a ferramenta decidida: '{tool_name}'")
            
            if tool_name == "analise_de_imagem":
                print("   - Chamando Agente de Imagens...")
                resultado_final = {
                    "agent": "AgenteDeImagens",
                    "data": self.agente_imagens.analisar_imagem(**arguments)
                }
                print("   - Agente de Imagens concluiu.")

            elif tool_name == "analise_estatistica":
                print("   - Chamando Agente Estatístico...")
                analysis_type = arguments.get("analysis_type")
                params = arguments.get("params", {})
                if 'features' in params: params['feature_columns'] = params.pop('features')
                
                resultado_analise = self.agente_estatistico.analisar(analysis_type=analysis_type, **params)
                nome_arquivo = f"relatorio_{analysis_type}.html"
                salvar_relatorio_html(resultado_analise, nome_arquivo)
                resultado_final = {"agent": "AgenteEstatistico", "data": {"status": "Análise concluída.", "relatorio_salvo_em": nome_arquivo}}
                print("   - Agente Estatístico concluiu.")

            elif tool_name == "geracao_de_relatorio":
                print("   - Chamando Agente de Relatórios...")
                dados_paciente = arguments.get("patient_data", {})
                html_report = generate_clinical_report(dados_paciente)
                nome_arquivo = "relatorio_clinico_gerado.html"
                with open(nome_arquivo, "w", encoding="utf-8") as f: f.write(html_report)
                resultado_final = {"agent": "AgenteDeRelatorios", "data": {"status": "Relatório gerado.", "relatorio_salvo_em": nome_arquivo, "html_report": html_report}}
                print("   - Agente de Relatórios concluiu.")
            
            else: # Inclui conversa_geral
                print("   - Roteado para Conversa Geral.")
                resultado_final = {"agent": "ConversaGeral", "data": {"answer": "Sou um sistema de IA focado em saúde. Como posso ajudar com análises de imagens ou dados?"}}
            
            print(">>> Processamento da solicitação concluído. Retornando resultado.")
            return resultado_final

        except Exception as e:
            print(f"!!!!!!!!!! ERRO INESPERADO DURANTE O PROCESSAMENTO !!!!!!!!!!")
            print(f"Erro: {e}")
            # Retorna um erro claro para o frontend
            return {
                "agent": "Sistema",
                "data": {
                    "error": "Ocorreu um erro interno no servidor.",
                    "details": str(e)
                }
            }


if __name__ == "__main__":
    sistema = SistemaMultiAgente()
    
    # O teste agora é feito pelo frontend e app.py
    print("\n" + "="*60)
    print("Sistema pronto. Inicie o servidor com 'python app.py' e acesse o frontend.html")
    print("="*60)
