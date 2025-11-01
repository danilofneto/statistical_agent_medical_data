#
# Arquivo: build_index.py
# Descrição: Script único para criar o banco de dados vetorial (FAISS)
# a partir de TODOS os documentos em uma pasta.
#
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuração ---
DOCUMENTS_FOLDER_PATH = "meus_documentos" # Pasta onde seus PDFs e TXTs estão
INDEX_PATH = "faiss_index" # Pasta onde o índice vetorial será salvo
MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO" # Modelo de embedding focado em medicina

def create_vector_store():
    """
    Lê TODOS os documentos de uma pasta, divide em pedaços,
    cria embeddings e salva em um índice FAISS.
    """
    if not os.path.exists(DOCUMENTS_FOLDER_PATH):
        print(f"ERRO: Pasta de documentos não encontrada em '{DOCUMENTS_FOLDER_PATH}'.")
        print("Por favor, crie esta pasta e adicione seus arquivos (.pdf, .txt).")
        return

    print(f"Iniciando a indexação de todos os documentos da pasta: {DOCUMENTS_FOLDER_PATH}...")

    # 1. Carregar os documentos da pasta
    # Mapeia extensões de arquivo para as classes de Loader corretas
    loader_map = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        # Você pode adicionar mais loaders aqui (ex: .docx)
    }
    
    loader = DirectoryLoader(
        DOCUMENTS_FOLDER_PATH,
        glob="**/*.*", # Carrega todos os arquivos
        use_multithreading=True,
        show_progress=True,
        loader_map=loader_map
    )
    
    docs = loader.load()
    
    if not docs:
        print("Nenhum documento encontrado na pasta. Encerrando.")
        return
        
    print(f"Total de {len(docs)} documentos carregados.")

    # 2. Dividir os documentos em "chunks" (pedaços)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Documentos divididos em {len(chunks)} pedaços (chunks).")

    # 3. Criar o modelo de embedding
    # Usando o mesmo modelo focado em PubMed que usamos anteriormente
    print(f"Carregando o modelo de embedding: {MODEL_NAME}...")
    embeddings = SentenceTransformerEmbeddings(model_name=MODEL_NAME)

    # 4. Criar o índice FAISS a partir dos chunks
    print("Criando o banco de dados vetorial FAISS...")
    db = FAISS.from_documents(chunks, embeddings)

    # 5. Salvar o índice no disco
    db.save_local(INDEX_PATH)
    print(f"✅ Índice FAISS salvo com sucesso na pasta: '{INDEX_PATH}'")

if __name__ == "__main__":
    create_vector_store()

