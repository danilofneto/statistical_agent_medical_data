import torch
import torch.nn as nn
import torch.optim as optim
from quantum_router import AgenteOrganizadorQuantico
import json
    
    
def treinar():
    print(">>> Iniciando Treinamento do Agente Router...")
    agente = AgenteOrganizadorQuantico()
    
    # 1. Dados de Treino (Exemplos simples para ele aprender padrões)
    # Formato: (Texto do Prompt, Índice da Ferramenta Correta)
    # 0: Imagem, 1: Estatistica, 2: Relatorio, 3: Conversa

    # dados_treino = [
    #     ("Analise esta imagem de raio-x", 0),
    #     ("Verifique a mancha nesta foto", 0),
    #     ("Tem tumor nesta tomografia?", 0),
    #     ("Dê uma olhada neste exame visual", 0),
        
    #     ("Qual a média de idade dos pacientes?", 1),
    #     ("Faça uma análise preditiva de risco", 1),
    #     ("Calcule o desvio padrão da pressão arterial", 1),
    #     ("Quantos pacientes tiveram alta?", 1),
        
    #     ("Gere um resumo do paciente João", 2),
    #     ("Crie um relatório clínico", 2),
    #     ("Resuma o histórico médico", 2),
    #     ("Exporte os dados do paciente para texto", 2),
        
    #     ("Bom dia, tudo bem?", 3),
    #     ("Olá, quem é você?", 3),
    #     ("Qual a capital do Brasil?", 3),
    #     ("Conte uma piada", 3)
    # ]
    
    print("Carregando dataset robusto...")
    with open("prompts_clinicos.json", "r", encoding="utf-8") as f:
        dados_json = json.load(f)
        
    # Converter para o formato de tuplas esperado pelo loop
    dados_treino = [(item["texto"], item["label"]) for item in dados_json]
    print(f"Total de amostras para treino: {len(dados_treino)}")
    
    # Configuração de Treinamento
    #optimizer = optim.Adam(agente.router_model.parameters(), lr=0.001)
    
    # Configuração de Treinamento
    optimizer = optim.Adam(agente.router_model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 150
    
    agente.router_model.train() # Modo de treino
    
    print(f"Treinando por {epochs} épocas...")
    for epoch in range(epochs):
        total_loss = 0
        for texto, label_idx in dados_treino:
            # 1. Zerar gradientes
            optimizer.zero_grad()
            
            # 2. Forward (Gerar Embedding -> Passar no Modelo)
            # Nota: Precisamos fazer manualmente o embedding aqui para ter o gradiente
            #inputs = agente.embedder.encode(texto, convert_to_tensor=True).to(agente.device).unsqueeze(0)
            raw_embedding = agente.embedder.encode(texto, convert_to_tensor=True)
            inputs = raw_embedding.detach().clone().to(agente.device).unsqueeze(0)            
            target = torch.tensor([label_idx], device=agente.device)
            
            # 3. Predição e Erro
            outputs = agente.router_model(inputs) # Logits/Probabilidades
            loss = loss_fn(outputs, target)
            
            # 4. Backward (Aprender)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Época {epoch+1}/{epochs} - Erro: {total_loss:.4f}")
            
    print(">>> Treinamento concluído!")
    
    # Salvar o cérebro treinado
    torch.save(agente.router_model.state_dict(), "pesos_agente_treinado.pth")
    print("Modelo salvo em 'pesos_agente_treinado.pth'")

if __name__ == "__main__":
    treinar()