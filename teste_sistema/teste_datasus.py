# arquivo: datasus_ingest.py
# descrição: coleta e integra dados SIH + SIA do DATASUS em CSVs normalizados

import pandas as pd
from pysus.online_data.SIH import download as download_sih
from pysus.online_data.SIA import download as download_sia
from tqdm import tqdm
import os

# ======================================================
# CONFIGURAÇÕES
# ======================================================
OUTPUT_DIR = "data/dados_datasus/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Grupos de doenças (CIDs relevantes)
CID_GROUPS = {
    "diabetes": ["E10", "E11", "E12", "E13", "E14"],
    "hipertensao": ["I10", "I11", "I12", "I13", "I15"],
    "doencas_cardiacas": ["I20", "I21", "I22", "I25", "I50"],
    "doencas_respiratorias": ["J40", "J44", "J45", "J46"],
    "neoplasias": ["C50", "C61", "C34", "C18"],
}

UFS = ["SP", "RJ", "MG", "RS"]
ANOS = [2023, 2024]


# ======================================================
# FUNÇÕES
# ======================================================
MESES = list(range(1, 13))  # 1 a 12
GRUPOS = ["all"]             # baixa todos os grupos do DATASUS

def coletar_sih(uf: str, ano: int) -> pd.DataFrame:
    """Baixa e processa dados hospitalares (SIH) do DATASUS."""
    print(f"📥 Coletando SIH para {uf}-{ano}...")
    try:
        df = download_sih(
            uf=uf, 
            year=ano, 
            months=MESES, 
            groups=GRUPOS
        )
        df["fonte"] = "SIH"
        df["uf"] = uf
        df["ano"] = ano
        
        return df
    except Exception as e:
        print(f"⚠️ Erro ao coletar SIH {uf}-{ano}: {e}")
        return pd.DataFrame()


def coletar_sia(uf: str, ano: int) -> pd.DataFrame:
    """Baixa e processa dados ambulatoriais (SIA) do DATASUS."""
    print(f"📥 Coletando SIA para {uf}-{ano}...")
    try:
        df = download_sia(
            uf=uf,
            year=ano,
            months=MESES,
            groups=GRUPOS
        )
        df["fonte"] = "SIA"
        df["uf"] = uf
        df["ano"] = ano
        return df
    except Exception as e:
        print(f"⚠️ Erro ao coletar SIA {uf}-{ano}: {e}")
        return pd.DataFrame()


def filtrar_por_cids(df: pd.DataFrame, grupos: dict) -> pd.DataFrame:
    """Filtra registros de acordo com grupos de CID definidos."""
    # identifica a coluna de diagnóstico (muda entre SIH/SIA)
    col_cid = None
    for c in ["DIAG_PRINC", "PA_CIDPRI"]:
        if c in df.columns:
            col_cid = c
            break

    if not col_cid:
        print("⚠️ Nenhuma coluna de CID encontrada, pulando filtro.")
        return df

    dfs = []
    for grupo, cids in grupos.items():
        mask = df[col_cid].astype(str).str.startswith(tuple(cids), na=False)
        df_tmp = df.loc[mask].copy()
        df_tmp["grupo_cid"] = grupo
        dfs.append(df_tmp)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def salvar_csv(df: pd.DataFrame, nome: str):
    """Salva DataFrame como CSV."""
    path = os.path.join(OUTPUT_DIR, nome)
    df.to_csv(path, index=False)
    print(f"💾 Dados salvos em {path} ({len(df)} linhas)")


def coletar_sih_sia_multiuf(ufs, anos, grupos_cid):
    """Coleta dados SIH e SIA para múltiplos estados e anos."""
    sih_total, sia_total = [], []

    for uf in tqdm(ufs, desc="Unidades Federativas"):
        for ano in anos:
            # --- SIH ---
            df_sih = coletar_sih(uf, ano)
            if not df_sih.empty:
                df_sih_filtrado = filtrar_por_cids(df_sih, grupos_cid)
                if not df_sih_filtrado.empty:
                    sih_total.append(df_sih_filtrado)

            # --- SIA ---
            df_sia = coletar_sia(uf, ano)
            if not df_sia.empty:
                df_sia_filtrado = filtrar_por_cids(df_sia, grupos_cid)
                if not df_sia_filtrado.empty:
                    sia_total.append(df_sia_filtrado)

    # Consolida e salva resultados
    if sih_total:
        df_sih_final = pd.concat(sih_total, ignore_index=True)
        salvar_csv(df_sih_final, "dados_sih.csv")

    if sia_total:
        df_sia_final = pd.concat(sia_total, ignore_index=True)
        salvar_csv(df_sia_final, "dados_sia.csv")

    print("✅ Coleta e integração SIH + SIA finalizadas com sucesso.")


# ======================================================
# EXECUÇÃO PRINCIPAL
# ======================================================
if __name__ == "__main__":
    coletar_sih_sia_multiuf(UFS, ANOS, CID_GROUPS)
