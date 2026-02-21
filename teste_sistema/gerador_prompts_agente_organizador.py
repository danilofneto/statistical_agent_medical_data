import json
import random
import itertools

def gerar_dataset():
    random.seed(42) # Para reprodutibilidade acadêmica
    dataset = []

    # ==========================================
    # CLASSE 0: ANÁLISE DE IMAGEM
    # ==========================================
    verbos_img = ["Analise", "Verifique", "Avalie", "Examine", "Cheque", "Dê uma olhada em"]
    tipos_img = ["esta imagem de raio-x", "esta tomografia", "esta ressonância magnética", "este ultrassom", "o rx", "a TC"]
    alvos_img = ["para ver se tem tumor", "em busca de anomalias", "verificando fraturas", "para identificar lesões", "buscando nódulos"]
    
    for v, t, a in itertools.product(verbos_img, tipos_img, alvos_img):
        dataset.append({"texto": f"{v} {t} {a}", "label": 0})
        # Variações mais curtas
        dataset.append({"texto": f"{v} {t}", "label": 0})

    # ==========================================
    # CLASSE 1: ANÁLISE ESTATÍSTICA E PREDITIVA
    # ==========================================
    verbos_est = ["Calcule", "Gere", "Faça", "Qual é", "Estime", "Construa"]
    metricas_est = ["a média de idade", "o risco preditivo", "um modelo de regressão", "a distribuição", "a análise de coorte", "uma série temporal do realizado"]
    alvos_est = ["dos pacientes", "na base de dados", "no dataframe", "para o risco cardiovascular", "dos internados"]
    
    for v, m, a in itertools.product(verbos_est, metricas_est, alvos_est):
        dataset.append({"texto": f"{v} {m} {a}", "label": 1})
        dataset.append({"texto": f"Preciso que você {v.lower()} {m} {a}", "label": 1})

    # ==========================================
    # CLASSE 2: GERAÇÃO DE RELATÓRIO
    # ==========================================
    verbos_rel = ["Gere", "Crie", "Escreva", "Resuma", "Exporte", "Monte"]
    tipos_rel = ["um resumo clínico", "um relatório médico", "o prontuário", "o histórico de alta", "os dados estruturados"]
    alvos_rel = ["do paciente", "deste caso", "do leito 4", "para o sistema", "em formato texto"]
    
    for v, t, a in itertools.product(verbos_rel, tipos_rel, alvos_rel):
        dataset.append({"texto": f"{v} {t} {a}", "label": 2})
        dataset.append({"texto": f"{t.capitalize()} {a}, por favor", "label": 2})

    # ==========================================
    # CLASSE 3: CONVERSA GERAL (Greetings / Fallback)
    # ==========================================
    frases_gerais = [
        "Bom dia, tudo bem?", "Olá, quem é você?", "Qual a capital do Brasil?",
        "Conte uma piada", "Boa tarde", "Como você pode me ajudar?",
        "Qual o sentido da vida?", "Teste de sistema", "Oi", "Tudo certo por aqui?"
    ]
    for frase in frases_gerais * 10: # Multiplicando para balancear as classes
        dataset.append({"texto": frase, "label": 3})

    # ==========================================
    # EMBARALHAR E SALVAR
    # ==========================================
    random.shuffle(dataset)
    
    # Exportar para JSON
    with open("prompts_clinicos.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
        
    print(f"Dataset gerado com sucesso! Total de exemplos: {len(dataset)}")
    
    # Mostrar distribuição
    dist = {0: 0, 1: 0, 2: 0, 3: 0}
    for item in dataset:
        dist[item['label']] += 1
    print(f"Distribuição das classes: {dist}")

if __name__ == "__main__":
    gerar_dataset()