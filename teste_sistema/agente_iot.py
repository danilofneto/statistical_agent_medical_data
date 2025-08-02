#
# Arquivo: agente_iot.py
# Descrição: Agente especializado em monitorar dados de dispositivos IoT em tempo real e detectar anomalias.
#
import time
import numpy as np
from datetime import datetime
from collections import deque

print("Dependências importadas com sucesso.")

def simular_dispositivo_iot(paciente_em_risco=False):
    """
    Um gerador que simula um dispositivo IoT (smartwatch) enviando dados a cada 2 segundos.
    Ele produz dados normais na maior parte do tempo, mas pode gerar anomalias.
    
    :param paciente_em_risco: Se True, aumenta a probabilidade de gerar anomalias.
    :yield: Um dicionário com os dados da leitura.
    """
    freq_cardiaca_base = 75
    spo2_base = 98
    
    while True:
        # Gera dados base com um pouco de ruído normal
        fc = np.random.normal(freq_cardiaca_base, 5)
        spo2 = np.random.normal(spo2_base, 0.5)
        
        # Chance de gerar uma anomalia
        chance_anomalia = 0.15 if paciente_em_risco else 0.05
        
        if np.random.rand() < chance_anomalia:
            tipo_anomalia = np.random.choice(['taquicardia', 'bradicardia', 'hipoxia', 'queda_subita_fc'])
            print(f"    (Simulador: Gerando anomalia do tipo '{tipo_anomalia}'...)")
            
            if tipo_anomalia == 'taquicardia':
                fc = np.random.normal(140, 10) # Frequência muito alta
            elif tipo_anomalia == 'bradicardia':
                fc = np.random.normal(40, 5)  # Frequência muito baixa
            elif tipo_anomalia == 'hipoxia':
                spo2 = np.random.normal(88, 2) # Saturação de oxigênio baixa
            elif tipo_anomalia == 'queda_subita_fc':
                # Simula uma queda brusca, o agente deve detectar a variação
                freq_cardiaca_base = 50 

        # Garante que os valores fiquem dentro de um intervalo razoável
        fc = np.clip(fc, 30, 200)
        spo2 = np.clip(spo2, 80, 100)
        
        leitura = {
            "timestamp": datetime.now(),
            "heart_rate_bpm": int(fc),
            "spo2_percent": round(spo2, 1)
        }
        
        yield leitura
        time.sleep(2) # Simula o intervalo entre leituras


class AgenteIoT:
    """
    Agente que monitora um fluxo de dados de saúde e detecta anomalias
    com base em regras pré-definidas.
    """
    def __init__(self, paciente_id: str, limiares: dict):
        """
        Inicializa o agente com os limiares específicos do paciente.
        
        :param paciente_id: Identificador do paciente.
        :param limiares: Dicionário com os valores críticos. Ex:
                         {'hr_max': 100, 'hr_min': 50, 'spo2_min': 92, 'hr_delta_max': 30}
        """
        self.paciente_id = paciente_id
        self.limiares = limiares
        # Usa um deque para armazenar o histórico recente de leituras (ex: últimos 10)
        self.historico_recente = deque(maxlen=10)
        print(f"Agente IoT inicializado para o paciente '{self.paciente_id}' com limiares definidos.")

    def _verificar_anomalia(self, dados_atuais: dict) -> (str, str):
        """
        Lógica interna para verificar se os dados atuais representam uma anomalia.
        Retorna uma tupla (tipo_alerta, mensagem) se uma anomalia for encontrada, senão None.
        """
        fc = dados_atuais['heart_rate_bpm']
        spo2 = dados_atuais['spo2_percent']

        # 1. Verificação de limiares absolutos
        if fc > self.limiares['hr_max']:
            return "Taquicardia", f"Frequência cardíaca ({fc} bpm) acima do limiar máximo ({self.limiares['hr_max']} bpm)."
        if fc < self.limiares['hr_min']:
            return "Bradicardia", f"Frequência cardíaca ({fc} bpm) abaixo do limiar mínimo ({self.limiares['hr_min']} bpm)."
        if spo2 < self.limiares['spo2_min']:
            return "Hipoxia", f"Saturação de oxigênio ({spo2}%) abaixo do limiar crítico ({self.limiares['spo2_min']}%)."

        # 2. Verificação de mudanças súbitas (se houver histórico suficiente)
        if len(self.historico_recente) > 1:
            fc_anterior = self.historico_recente[-1]['heart_rate_bpm']
            delta_fc = abs(fc - fc_anterior)
            if delta_fc > self.limiares['hr_delta_max']:
                return "Variação Súbita de FC", f"Variação súbita da frequência cardíaca detectada (de {fc_anterior} para {fc} bpm)."

        return None, None # Nenhuma anomalia

    def monitorar_ponto_de_dados(self, ponto_de_dados: dict) -> dict:
        """
        Recebe um único ponto de dados, o analisa e o adiciona ao histórico.
        Este é o principal ponto de entrada para integração.
        """
        tipo_alerta, mensagem = self._verificar_anomalia(ponto_de_dados)
        
        # Adiciona a leitura atual ao histórico para futuras análises de variação
        self.historico_recente.append(ponto_de_dados)
        
        if tipo_alerta:
            return {
                "status": "alerta",
                "paciente_id": self.paciente_id,
                "tipo_alerta": tipo_alerta,
                "motivo": mensagem,
                "dados_atuais": ponto_de_dados
            }
        else:
            return {
                "status": "normal",
                "paciente_id": self.paciente_id,
                "dados_atuais": ponto_de_dados
            }

# --- FUNÇÃO DE EXECUÇÃO E DEMONSTRAÇÃO ---

if __name__ == "__main__":
    print("="*60)
    print("DEMONSTRAÇÃO DO AGENTE IOT EM TEMPO REAL")
    print("="*60)
    
    # 1. Configurar o paciente e o agente
    id_paciente_demo = "Paciente-007"
    limiares_criticos = {
        'hr_max': 100,      # Batimentos por minuto (acima de)
        'hr_min': 50,       # Batimentos por minuto (abaixo de)
        'spo2_min': 92,     # Saturação de oxigênio (abaixo de)
        'hr_delta_max': 30  # Mudança máxima de FC entre duas leituras
    }
    
    agente_iot = AgenteIoT(id_paciente_demo, limiares_criticos)
    
    # 2. Iniciar o simulador de dispositivo
    # Vamos simular um paciente com maior risco para forçar anomalias
    fluxo_de_dados_iot = simular_dispositivo_iot(paciente_em_risco=True)
    
    print("\nIniciando monitoramento... (Pressione Ctrl+C para parar)\n")
    
    # 3. Loop de monitoramento em tempo real
    try:
        for i in range(50): # Monitora por 50 ciclos (aprox. 100 segundos)
            # Obter a leitura mais recente do dispositivo
            leitura_atual = next(fluxo_de_dados_iot)
            
            # O agente processa a leitura
            resultado = agente_iot.monitorar_ponto_de_dados(leitura_atual)
            
            # Exibir o status no console
            status = resultado['status'].upper()
            fc = resultado['dados_atuais']['heart_rate_bpm']
            spo2 = resultado['dados_atuais']['spo2_percent']
            timestamp = resultado['dados_atuais']['timestamp'].strftime('%H:%M:%S')
            
            print(f"[{timestamp}] Status: {status} | FC: {fc} bpm | SpO2: {spo2}%")
            
            # Se um alerta for gerado, tomar uma ação
            if resultado['status'] == 'alerta':
                print("-" * 60)
                print(f"🚨 ALERTA IMEDIATO GERADO PARA O PACIENTE: {resultado['paciente_id']} 🚨")
                print(f"   TIPO: {resultado['tipo_alerta']}")
                print(f"   MOTIVO: {resultado['motivo']}")
                print("-" * 60)
                
                # Em um sistema real, aqui você chamaria outro agente ou enviaria uma notificação.
                # Para a demonstração, vamos parar o monitoramento.
                break
                
    except KeyboardInterrupt:
        print("\nMonitoramento interrompido pelo usuário.")
    
    print("\nDemonstração concluída.")
