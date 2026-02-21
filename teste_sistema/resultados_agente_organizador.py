import time
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from agente_organizador import AgenteOrganizador
from quantum_router import AgenteOrganizadorQuantico

def plotar_matriz_confusao(y_true, y_pred, classes):
    """Gera e salva a matriz de confusão em alta resolução para a tese."""
    # Calcula a matriz
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Configura o tamanho e o estilo da figura (Padrão artigo científico)
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="white")
    
    # Cria o heatmap com o Seaborn (Tons de azul são padrão para publicações)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar=True, annot_kws={"size": 14})
    
    # Ajusta os rótulos e título
    plt.title('Matriz de Confusão - Agente Integrador (Quantum-Inspired)', fontsize=16, pad=15)
    plt.xlabel('Predição do Agente', fontsize=14, labelpad=10)
    plt.ylabel('Classe Real (Ground Truth)', fontsize=14, labelpad=10)
    
    # Rotaciona os labels do eixo X para não encavalarem
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    # Ajusta o layout para não cortar as margens
    plt.tight_layout()
    
    # Salva em alta resolução (300 dpi) nos formatos PDF e PNG
    plt.savefig('matriz_confusao.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('matriz_confusao.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Gráficos salvos: 'matriz_confusao.pdf' e 'matriz_confusao.png'")

def avaliar_melhora():
    # 1. Dataset de Teste (Ground Truth - O que o Llama-3 "Professor" decidiria)
    # Crie uma lista com 50 exemplos reais
    dataset = [
        # --- ANÁLISE ESTATÍSTICA / CIÊNCIA DE DADOS (14 exemplos) ---
        ("Crie uma análise de coorte por mês e ano para avaliar a retenção de pacientes no tratamento.", "analise_estatistica"),
        ("Construa um modelo de série temporal para prever o volume de internações com base no histórico da data-base.", "analise_estatistica"),
        ("Cruze o dataframe de pacientes com os dados de desfecho clínico e remova as duplicatas.", "analise_estatistica"),
        ("Como posso interpretar as métricas agregadas deste modelo de risco, onde o MAE é 45 e o RMSE é 92?", "analise_estatistica"),
        ("Calcule a correlação de Pearson entre o IMC, idade e a taxa de readmissão hospitalar.", "analise_estatistica"),
        ("Treine um modelo preditivo utilizando árvores de decisão para identificar pacientes com alto risco de sepse.", "analise_estatistica"),
        ("Extraia a taxa média de mortalidade agrupada por comorbidade e código do posto de atendimento.", "analise_estatistica"),
        ("Faça uma análise exploratória de dados sobre as contratações de crédito de saúde.", "analise_estatistica"),
        ("Analise se houve diferença na renda ajustada para mais ou para menos após a alta do paciente.", "analise_estatistica"),
        ("Identifique as features mais importantes para o modelo usando SHAP values.", "analise_estatistica"),
        ("Gere uma curva ROC para avaliar a precisão do classificador de doenças cardíacas.", "analise_estatistica"),
        ("Teste a hipótese nula de que o novo medicamento não reduz a febre (calcule o p-valor).", "analise_estatistica"),
        ("Agrupe os perfis de risco dos pacientes usando o algoritmo K-means.", "analise_estatistica"),
        ("Avalie a variância da pressão sistólica dos pacientes da UTI na última semana.", "analise_estatistica"),

        # --- ANÁLISE DE IMAGEM (12 exemplos) ---
        ("Veja esta mancha na pele.", "analise_de_imagem"),
        ("Analise o raio-x de tórax do paciente do leito 5 e me diga se há padrão de vidro fosco.", "analise_de_imagem"),
        ("Verifique se há traços de fratura de fêmur nesta radiografia.", "analise_de_imagem"),
        ("Tem sinais de pneumonia intersticial nesta TC?", "analise_de_imagem"),
        ("Avalie esta ressonância magnética do crânio em busca de isquemia.", "analise_de_imagem"),
        ("Compare o volume do nódulo neste ultrassom com o exame do mês passado.", "analise_de_imagem"),
        ("Há presença de microcalcificações suspeitas nesta mamografia?", "analise_de_imagem"),
        ("Identifique a área e o perímetro da lesão cutânea nesta fotografia clínica.", "analise_de_imagem"),
        ("Qual o grau de infiltração celular na imagem desta lâmina histopatológica?", "analise_de_imagem"),
        ("Verifique o padrão do fluxo vascular na angiografia contrastada.", "analise_de_imagem"),
        ("Analise o fundo de olho buscando sinais de retinopatia diabética grave.", "analise_de_imagem"),
        ("Este eletrocardiograma apresenta supradesnivelamento do segmento ST?", "analise_de_imagem"),

        # --- GERAÇÃO DE RELATÓRIO (12 exemplos) ---
        ("Gere o resumo de alta estruturado do paciente João da Silva.", "geracao_de_relatorio"),
        ("Transforme o JSON com os dados vitais do paciente 405 em um texto legível para o médico.", "geracao_de_relatorio"),
        ("Crie um relatório clínico com a evolução das últimas 72 horas.", "geracao_de_relatorio"),
        ("Resuma as anotações diárias de enfermagem em um único parágrafo objetivo.", "geracao_de_relatorio"),
        ("Compile a dosagem e horários das medicações de hoje em um relatório de turno.", "geracao_de_relatorio"),
        ("Gere um laudo preliminar com base nos achados textuais registrados na triagem.", "geracao_de_relatorio"),
        ("Estruture a queixa principal e a história pregressa neste formato de formulário padrão.", "geracao_de_relatorio"),
        ("Escreva um sumário executivo da condição clínica da UTI Cardiológica.", "geracao_de_relatorio"),
        ("Formate a lista de resultados dos exames laboratoriais em um relatório descritivo corrido.", "geracao_de_relatorio"),
        ("Prepare a folha de rosto do prontuário para a transferência do paciente para o bloco cirúrgico.", "geracao_de_relatorio"),
        ("Faça um relatório consolidando as alergias e comorbidades crônicas registradas.", "geracao_de_relatorio"),
        ("Traduza essas marcações sistêmicas em um parecer médico compreensível.", "geracao_de_relatorio"),

        # --- CONVERSA GERAL (12 exemplos) ---
        ("Bom dia", "conversa_geral"),
        ("Olá, como funciona a sua arquitetura de múltiplos agentes?", "conversa_geral"),
        ("Quem desenvolveu este sistema de inteligência artificial?", "conversa_geral"),
        ("Qual a previsão do tempo para amanhã na minha cidade?", "conversa_geral"),
        ("Me conte uma curiosidade histórica sobre a descoberta da penicilina.", "conversa_geral"),
        ("Boa tarde, preciso de ajuda com um caso clínico complexo.", "conversa_geral"),
        ("Obrigado pelas respostas, foram muito úteis.", "conversa_geral"),
        ("Quais são as ferramentas e integrações que você suporta?", "conversa_geral"),
        ("Como faço para resetar o contexto desta conversa?", "conversa_geral"),
        ("Até logo, vou encerrar o turno agora.", "conversa_geral"),
        ("Tudo bem por aí com os servidores?", "conversa_geral"),
        ("Me explique brevemente o que é um LLM.", "conversa_geral")
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

    # --- GERAR GRÁFICO PARA A TESE ---
    classes_ordenadas = [
        "analise_de_imagem", 
        "analise_estatistica", 
        "geracao_de_relatorio", 
        "conversa_geral"
    ]
    plotar_matriz_confusao(gabarito, preds_novo, classes_ordenadas)

if __name__ == "__main__":
    avaliar_melhora()