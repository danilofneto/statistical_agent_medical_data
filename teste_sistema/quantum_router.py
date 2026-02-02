import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

# ==============================================================================
# PARTE 1: O MODELO MATEMÁTICO (Quantum-Inspired Neural Network)
# ==============================================================================

class QuantumInspiredRouter(nn.Module):
    """
    Implementação PyTorch da arquitetura QD-LLM para roteamento de agentes.
    Simula o fluxo de informação de um circuito quântico variacional (VQC).
    """
    def __init__(self, input_dim=768, n_qubits=11, n_layers=2, n_tools=4):
        super(QuantumInspiredRouter, self).__init__()
        
        self.n_qubits = n_qubits
        
        # 1. Camada de Redução Dimensional (Eq. 1 do LaTeX)
        # Transforma o embedding do texto (ex: 768 dim) para o espaço de qubits (11 dim)
        self.reduction_layer = nn.Linear(input_dim, n_qubits)
        
        # 2. Ansatz Variacional (Eq. 3 do LaTeX)
        # Simulamos a evolução unitária U(theta) como camadas de rotação e mistura.
        # Em um simulador real, seriam portas quânticas. Aqui, usamos camadas densas
        # com ativações periódicas para mimetizar a natureza ondulatória (rotações).
        self.ansatz_layers = nn.ModuleList()
        for _ in range(n_layers):
            # "Rotação" e "Emaranhamento" simulados
            layer = nn.Sequential(
                nn.Linear(n_qubits, n_qubits),
                nn.Tanh(), # Não-linearidade suave
                nn.Linear(n_qubits, n_qubits) # Mistura (Entanglement simulado)
            )
            self.ansatz_layers.append(layer)
            
        # 3. Medição (Eq. 4 do LaTeX)
        # Projeta o estado final dos qubits nas probabilidades das ferramentas
        self.measurement_layer = nn.Linear(n_qubits, n_tools)

    def forward(self, x):
        """
        Passo de inferência do modelo.
        x: Vetor de embedding do prompt (Batch Size, Input Dim)
        """
        # --- Etapa A: Codificação de Dados (Data Encoding) ---
        # Z = W * E + b
        z = self.reduction_layer(x)
        
        # Simulando a codificação Rx(z): Normalizamos para [0, 2pi] para atuar como ângulos
        angles = torch.sigmoid(z) * 2 * np.pi 
        
        # Representação do Estado Quântico inicial (Simulação simplificada de amplitudes)
        # |psi> = cos(theta/2)|0> - i*sin(theta/2)|1>
        # Aqui trabalhamos com a representação real das características
        state = torch.cos(angles) 

        # --- Etapa B: Evolução Variacional (Ansatz) ---
        # Aplica as camadas de "portas lógicas" aprendíveis
        for layer in self.ansatz_layers:
            # Conexão residual para estabilidade (comum em QNNs profundas)
            state = state + layer(state)
            
        # --- Etapa C: Medição e Decisão ---
        # Mapeia o estado final para as logits das ferramentas
        logits = self.measurement_layer(state)
        
        # Retorna probabilidades (Softmax)
        return F.softmax(logits, dim=1)

# ==============================================================================
# PARTE 2: AGENTE ORGANIZADOR ATUALIZADO
# ==============================================================================

class AgenteOrganizadorQuantico:
    def __init__(self, embedding_model_name="sentence-transformers/all-mpnet-base-v2"):
        print(">>> Inicializando Agente Organizador Híbrido...")

        
        # # 1. Carregar modelo de Embedding (para transformar texto em vetor)
        # # (Em produção, use o mesmo que você usa no FAISS)
        # try:
        #     from sentence_transformers import SentenceTransformer
        #     self.embedder = SentenceTransformer(embedding_model_name)
        #     self.input_dim = self.embedder.get_sentence_embedding_dimension()
        # except ImportError:
        #     raise ImportError("Instale: pip install sentence-transformers")

        # ### CORREÇÃO 1: Detectar se há GPU disponível
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   - Dispositivo de processamento: {self.device}")

        # 1. Carregar modelo de Embedding
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(embedding_model_name)
            
            # ### CORREÇÃO 2: Mover o modelo de embedding para o dispositivo correto
            self.embedder.to(self.device)
            
            self.input_dim = self.embedder.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError("Instale: pip install sentence-transformers")

        # 2. Definição das Ferramentas (Classes de Saída)
        self.tools = [
            "analise_de_imagem",    # Classe 0
            "analise_estatistica",  # Classe 1
            "geracao_de_relatorio", # Classe 2
            "conversa_geral"        # Classe 3
        ]
        
        # 3. Instanciar o "Cérebro Quântico"
        self.router_model = QuantumInspiredRouter(
            input_dim=self.input_dim, 
            n_qubits=11, 
            n_layers=4, 
            n_tools=len(self.tools)
        )

        # ### CORREÇÃO 3: Enviar o modelo PyTorch para a GPU (se disponível)
        self.router_model.to(self.device)
        
        self.router_model.eval()
        print(f"Sistema pronto. Router operando com {self.router_model.n_qubits} qubits virtuais.")

        # NOTA: Em um caso real, você carregaria os pesos treinados aqui.
        # self.router_model.load_state_dict(torch.load("pesos_treinados_qd_llm.pth"))
        #self.router_model.eval() # Modo de inferência
        #print(f"Sistema pronto. Router operando com {self.router_model.n_qubits} qubits virtuais.")

    def extrair_parametros_com_llm(self, prompt, tool_name):
        """
        Função auxiliar: Se o roteador decidir a ferramenta, usamos um LLM pequeno
        apenas para extrair os JSONs (tarefa mais simples e rápida que decidir).
        """
        # Simulação para o exemplo (aqui você chamaria o Ollama se quisesse extrair dados)
        return {"raw_prompt": prompt, "status": "extraido_via_llm_lite"}

    # def rotear_prompt(self, user_prompt: str):
    #     print(f"\n[Input]: '{user_prompt}'")
        
    #     # 1. Gerar Embedding (Vetor Clássico)
    #     embedding_vector = self.embedder.encode(user_prompt, convert_to_tensor=True)
    #     # Ajustar forma para (1, input_dim)
    #     embedding_vector = embedding_vector.unsqueeze(0)
        
    #     # 2. Inferência no Modelo Quântico (Forward Pass)
    #     with torch.no_grad():
    #         probabilities = self.router_model(embedding_vector)
        
    #     # 3. Interpretar Resultado
    #     best_tool_idx = torch.argmax(probabilities, dim=1).item()
    #     confidence = probabilities[0][best_tool_idx].item()
    #     selected_tool = self.tools[best_tool_idx]
        
    #     print(f"   [Processamento Quântico]:")
    #     #print(f"   - Estado Reduzido (11 dim): {self.router_model.reduction_layer(embedding_vector).numpy()[0][:4]}...")
    #     vetor_reduzido = self.router_model.reduction_layer(embedding_vector)
    #     print(f"   - Estado Reduzido (11 dim): {vetor_reduzido.detach().cpu().numpy()[0][:4]}...")
    #     print(f"   - Confiança: {confidence:.2%}")
    #     print(f"   - Decisão: {selected_tool.upper()}")

    #     # 4. Construir Resposta
    #     result = {
    #         "tool_name": selected_tool,
    #         "confidence": confidence,
    #         "arguments": self.extrair_parametros_com_llm(user_prompt, selected_tool)
    #     }
    #     return result

    def rotear_prompt(self, user_prompt: str):
        # 1. Gerar Embedding
        embedding_vector = self.embedder.encode(user_prompt, convert_to_tensor=True)
        
        # Garantir que o tensor vá para o dispositivo correto (GPU/CPU)
        embedding_vector = embedding_vector.to(self.device)
        
        # Ajustar forma para (1, input_dim)
        embedding_vector = embedding_vector.unsqueeze(0)
        
        # 2. Inferência no Modelo Quântico (TUDO dentro do no_grad)
        with torch.no_grad():
            # Processamento principal
            probabilities = self.router_model(embedding_vector)
            
            # --- DEBUG VISUAL (Opcional) ---
            # Aqui calculamos o estado reduzido apenas para visualizar.
            # Como estamos dentro do torch.no_grad(), não haverá erro de gradiente.
            vetor_reduzido = self.router_model.reduction_layer(embedding_vector)
            
            # .cpu().numpy() move os dados da GPU para a RAM para poder imprimir
            print(f"   - Estado Reduzido (11 dim): {vetor_reduzido.cpu().numpy()[0][:4]}...")
        
        # 3. Interpretar Resultado
        best_tool_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][best_tool_idx].item()
        selected_tool = self.tools[best_tool_idx]

        # 4. Construir Resposta
        result = {
            "tool_name": selected_tool,
            "confidence": confidence,
            "arguments": self.extrair_parametros_com_llm(user_prompt, selected_tool)
        }
        return result

# ==============================================================================
# DEMONSTRAÇÃO
# ==============================================================================

if __name__ == "__main__":
    # Inicializa o agente
    agente = AgenteOrganizadorQuantico()
    
    # Testes
    test_prompts = [
        "Analise esta tomografia para ver se tem tumor.",
        "Qual a média de idade dos pacientes na base de dados?",
        "Gere um resumo clínico do paciente João Silva.",
        "Bom dia, tudo bem?"
    ]
    
    print("="*60)
    print("INICIANDO ROTEAMENTO OTIMIZADO")
    print("="*60)

    for prompt in test_prompts:
        decisao = agente.rotear_prompt(prompt)
        # print(json.dumps(decisao, indent=2))