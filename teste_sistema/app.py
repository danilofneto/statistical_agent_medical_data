#
# Arquivo: app.py
# Descrição: Versão atualizada que lida com imagens reais do frontend.
#
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import base64
import os

# Importe a classe principal do seu sistema e o agente real
from main import SistemaMultiAgente
from agente_imagens import AgenteDeImagens # Importa o agente real

print("Inicializando o servidor Flask...")

app = Flask(__name__)
CORS(app)

print("Carregando o Sistema MultiAgente... Isso pode levar um momento.")
sistema_multi_agente = SistemaMultiAgente()
print("✅ Sistema MultiAgente pronto para receber solicitações.")


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Endpoint da API que recebe a pergunta e a imagem, processa com os agentes reais e retorna o resultado.
    """
    print("\n--- Nova solicitação recebida no endpoint /analyze ---")
    
    data = request.get_json()
    prompt_usuario = data.get('prompt', '')
    imagem_base64 = data.get('imageBase64')

    if not prompt_usuario and not imagem_base64:
        return jsonify({"error": "Nenhum prompt ou imagem fornecido."}), 400

    # Se uma imagem foi enviada, o Agente Organizador precisa saber o caminho dela.
    # Vamos salvar a imagem temporariamente e adicionar o caminho ao prompt.
    caminho_imagem_temporaria = None
    if imagem_base64:
        try:
            # Cria um diretório temporário se não existir
            os.makedirs("temp_images", exist_ok=True)
            caminho_imagem_temporaria = os.path.join("temp_images", "temp_image.jpg")
            
            # Decodifica a string Base64 e salva o arquivo
            image_data = base64.b64decode(imagem_base64)
            with open(caminho_imagem_temporaria, 'wb') as f:
                f.write(image_data)
            
            # Adiciona a informação do caminho ao prompt para o organizador
            prompt_usuario = f"{prompt_usuario} [imagem anexada em: '{caminho_imagem_temporaria}']"
            print(f"   - Imagem recebida e salva em: {caminho_imagem_temporaria}")

        except Exception as e:
            print(f"   - ERRO ao salvar imagem temporária: {e}")
            return jsonify({"error": "Falha ao processar a imagem recebida."}), 500
            
    # O sistema processa a solicitação (agora com o caminho da imagem, se houver)
    resultado_final = sistema_multi_agente.processar_solicitacao(prompt_usuario, caminho_imagem_temporaria)

    # Limpa a imagem temporária após o uso
    if caminho_imagem_temporaria and os.path.exists(caminho_imagem_temporaria):
        os.remove(caminho_imagem_temporaria)
        print(f"   - Imagem temporária removida: {caminho_imagem_temporaria}")

    print(f"--- Enviando resposta para o frontend ---")
    return jsonify(resultado_final)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
