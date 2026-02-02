import time
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from agente_organizador import AgenteOrganizador
from quantum_router import AgenteOrganizadorQuantico

def avaliar_melhora():
    # 1. Dataset de Teste (Ground Truth - O que o Llama-3 "Professor" decidiria)
    # Crie uma lista com 50 exemplos reais
    dataset = [
        ("Dor no peito irradiando para o braço esquerdo", "analise_estatistica"),
        ("Veja esta mancha na pele", "analise_de_imagem"),
        ("Bom dia", "conversa_geral"),
        # ... mais 47 exemplos ...
    ]
    prompts = [x[0] for x in dataset]
    gabarito = [x[1] for x in dataset]

    # --- AVALIAÇÃO DO AGENTE NOVO (QUÂNTICO/OTIMIZADO) ---
    print(">>> Testando Agente Quântico...")
    agente_novo = AgenteOrganizadorQuantico() # Seu modelo PyTorch
    
    start_time = time.time()
    preds_novo = []
    for p in prompts:
        # Supondo que o método retorne apenas o nome da ferramenta para o teste
        res = agente_novo.rotear_prompt(p) 
        preds_novo.append(res['tool_name'])
    end_time = time.time()
    
    tempo_novo = end_time - start_time
    acc_novo = accuracy_score(gabarito, preds_novo)
    
    # --- RESULTADOS ---
    print(f"\nResultados da Comparação:")
    print(f"1. Acurácia: {acc_novo:.2%} (Quanto mais próximo de 100%, melhor o aprendizado)")
    print(f"2. Tempo Total: {tempo_novo:.4f}s para {len(prompts)} prompts")
    print(f"3. Velocidade: {tempo_novo/len(prompts)*1000:.2f} ms/prompt")
    
    # Detalhe por classe (F1-Score) 
    print("\nRelatório Detalhado (Precision/Recall/F1):")
    print(classification_report(gabarito, preds_novo))

if __name__ == "__main__":
    avaliar_melhora()