#
# Arquivo: agente_imagens.py
# Descrição: Agente especializado em analisar imagens médicas usando um modelo de linguagem multimodal (VQA).
#
import base64
from io import BytesIO
from PIL import Image
import requests
import os

# Libs do LangChain para interagir com o modelo multimodal via Ollama
from langchain_community.llms import Ollama

print("Dependências importadas com sucesso.")

class AgenteDeImagens:
    """
    Agente que utiliza um modelo de linguagem e visão para responder perguntas
    sobre imagens médicas (Visual Question Answering - VQA).
    """
    def __init__(self, model_name: str = "llava:7b"):
        """
        Inicializa o agente com um modelo multimodal servido pelo Ollama.

        :param model_name: O nome do modelo VQA a ser usado (ex: 'llava:7b', 'alibayram/medgemma:4b').
        """
        self.model_name = model_name
        try:
            # A classe Ollama do LangChain pode interagir com modelos multimodais
            # passando a imagem como parte da chamada.
            self.llm = Ollama(model=self.model_name, temperature=0.1)
            # Testa a conexão
            self.llm.invoke("Olá")
            print(f"Agente de Imagens inicializado com sucesso, conectado ao modelo '{self.model_name}'.")
        except Exception as e:
            print(f"ERRO: Não foi possível conectar ao Ollama. Verifique se ele está em execução e se o modelo '{self.model_name}' foi baixado (`ollama pull {self.model_name}`).")
            print(f"Detalhe do erro: {e}")
            exit()

    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Converte uma imagem de um arquivo para uma string Base64.
        Este é o formato que o Ollama espera para receber imagens no prompt.
        """
        try:
            with Image.open(image_path) as img:
                # Converte para RGB para garantir consistência
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Erro ao codificar a imagem {image_path}: {e}")
            return None

    def analisar_imagem(self, image_path: str, question: str):
        """
        Ponto de entrada principal do agente. Analisa uma imagem e responde a uma pergunta sobre ela.
        Implementa a técnica de "Cache-Augmented Generation" enviando a imagem diretamente no prompt.

        :param image_path: Caminho para o arquivo da imagem.
        :param question: A pergunta a ser feita sobre a imagem.
        :return: Um dicionário com a resposta e informações da análise.
        """
        print(f"\n>>> Iniciando análise da imagem: '{os.path.basename(image_path)}'")
        print(f"    Pergunta: '{question}'")

        if not os.path.exists(image_path):
            return {"error": f"Arquivo de imagem não encontrado em: {image_path}"}

        # 1. Codificar a imagem para Base64
        print("   - Codificando imagem para Base64...")
        base64_image = self._encode_image_to_base64(image_path)
        if not base64_image:
            return {"error": "Falha ao processar a imagem."}

        # 2. Chamar o modelo multimodal com a imagem e a pergunta
        print(f"   - Enviando imagem e pergunta para o modelo '{self.model_name}'...")
        try:
            # A biblioteca do LangChain para Ollama permite passar imagens no parâmetro `images`
            response_text = self.llm.invoke(question, images=[base64_image])
            print("   - Resposta recebida do modelo.")
        except Exception as e:
            print(f"   - ERRO ao chamar o modelo: {e}")
            return {"error": "Ocorreu um erro ao comunicar com o modelo de visão."}

        # 3. Estruturar a saída
        resultado = {
            "image_path": image_path,
            "question": question,
            "answer": response_text.strip()
        }
        
        return resultado

# --- FUNÇÃO DE EXECUÇÃO E DEMONSTRAÇÃO ---

def baixar_imagem_exemplo(url: str, nome_arquivo: str):
    """Baixa uma imagem da web se ela não existir localmente."""
    if not os.path.exists(nome_arquivo):
        print(f"Baixando imagem de exemplo de '{url}'...")
        try:
            response = requests.get(url)
            response.raise_for_status() # Lança um erro para respostas ruins (4xx ou 5xx)
            with open(nome_arquivo, 'wb') as f:
                f.write(response.content)
            print(f"✅ Imagem salva como '{nome_arquivo}'.")
        except requests.exceptions.RequestException as e:
            print(f"❌ Falha ao baixar a imagem: {e}")
            return False
    return True


if __name__ == "__main__":
    print("="*60)
    print("🚀 DEMONSTRAÇÃO DO AGENTE DE IMAGENS MÉDICAS (VQA)")
    print("="*60)
    
    # 1. Preparar o ambiente
    # URL de um raio-x de tórax com pneumonia (fonte: Wikimedia Commons, CC BY-SA 4.0)
    IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/c/c1/Pneumonia_x-ray.jpg"
    IMAGE_PATH = "raio_x_exemplo.jpg"
    
    if not baixar_imagem_exemplo(IMAGE_URL, IMAGE_PATH):
        exit()

    # 2. Inicializar o agente
    # Certifique-se de ter o LLaVA rodando no Ollama: `ollama run llava:7b`
    agente_imagens = AgenteDeImagens(model_name="llava:7b")
    
    # 3. Definir a tarefa (pergunta)
    pergunta_clinica = "Esta radiografia de tórax parece normal ou há sinais de anormalidades como consolidação ou infiltrados, sugestivos de pneumonia?"
    
    # 4. Executar a análise
    resultado_analise = agente_imagens.analisar_imagem(IMAGE_PATH, pergunta_clinica)
    
    # 5. Exibir o resultado de forma clara
    print("\n" + "-"*60)
    print("📋 RESULTADO DA ANÁLISE DE IMAGEM")
    print("-" * 60)
    if "error" in resultado_analise:
        print(f"Ocorreu um erro: {resultado_analise['error']}")
    else:
        print(f"🖼️ Imagem Analisada: {resultado_analise['image_path']}")
        print(f"❓ Pergunta Feita: {resultado_analise['question']}")
        print("\n🧠 Resposta do Modelo:")
        print(resultado_analise['answer'])
    print("-" * 60)
    
    # Exemplo de como a saída pode ser usada por outro agente
    print("\n--- EXEMPLO DE INTEGRAÇÃO ---")
    print("O dicionário de resultado pode ser enviado para o Agente de Relatórios para ser incluído no prontuário do paciente.")

