# 
# Arquivo: datasus_ingest.py
# Descrição: Coleta automática de dados do DATASUS (CSV) e PDFs
# médicos do Ministério da Saúde, salvando-os para uso no RAG.
# 

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Biblioteca da Fiocruz para acessar dados públicos do SUS
#from pysus.online_data import SIH
from pysus.online_data.SIA import download as download_sia

# --- Configurações ---
DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "datasus_docs")
CSV_OUTPUT = os.path.join(DATA_DIR, "datasus_diabetes.csv")

os.makedirs(PDF_DIR, exist_ok=True)

# ============================================================
# Coleta de dados estruturados do DATASUS (internações por diabetes)
# ============================================================
def baixar_dados_sih(ano=2023):
    print(f"Baixando dados do SIH/SUS (internações) para {ano}...")
    # ufs = ['AC','AL','AM','AP','BA','CE','DF','ES','GO','MA','MG','MS','MT',
    #        'PA','PB','PE','PI','PR','RJ','RN','RO','RR','RS','SC','SE','SP','TO']
    ufs = ['MG']  # Teste com apenas MG para agilizar

    # dfs = []
    # for uf in ufs:
    #     try:
    #         df = SIH().download('MG', 2023)
    #         df["UF"] = uf
    #         dfs.append(df)
    #     except Exception as e:
    #         print(f"Falha ao baixar dados de {uf}: {e}")

    # dados_sih = pd.concat(dfs, ignore_index=True)
    # print(f"Dados consolidados: {len(dados_sih)} registros totais.")
    # dados_sih.to_csv(CSV_OUTPUT, index=False)
    # print(f"Arquivo salvo em: {CSV_OUTPUT}")
    
    dados_sih = download_sia('MG', 2023, months=list(range(1,2)), groups=["BI"])
    print(f"Dados consolidados: {len(dados_sih)} registros totais.")

def baixar_pdfs_diabetes():
    print("Buscando PDFs do Ministério da Saúde sobre Diabetes...")
    url = "https://www.gov.br/saude/pt-br/assuntos/saude-de-a-a-z/d/diabetes"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    pdf_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.pdf')]
    print(f"Encontrados {len(pdf_links)} PDFs.")

    for link in pdf_links:
        nome = link.split('/')[-1]
        if not link.startswith("http"):
            link = "https://www.gov.br" + link
        caminho = os.path.join(PDF_DIR, nome)
        try:
            r = requests.get(link)
            with open(caminho, 'wb') as f:
                f.write(r.content)
            print(f"PDF salvo: {nome}")
        except Exception as e:
            print(f"Erro ao baixar {link}: {e}")

    print(f"Todos os PDFs salvos em: {PDF_DIR}")


if __name__ == "__main__":
    baixar_dados_sih(2023)
    baixar_pdfs_diabetes()
    print("\nColeta concluída. Agora execute: python build_index.py")
