

# ========================================
# 5. main.py - ARQUIVO PRINCIPAL
# ========================================

"""
Arquivo: main.py
Ponto de entrada principal do projeto
"""

def main():
    """Função principal"""
    print("🤖 AGENTE ESTATÍSTICO - SISTEMA DE ANÁLISE CLÍNICA")
    print("=" * 60)
    
    # Verificar configurações
    from config import CONFIG
    
    if not CONFIG.openai_api_key:
        print("⚠️  ATENÇÃO: Configure sua OPENAI_API_KEY!")
        print("   export OPENAI_API_KEY='sua_chave_aqui'")
        return
    
    # Executar demos
    from examples import AgentDemo
    
    demo = AgentDemo()
    
    print("Escolha uma opção:")
    print("1. Demo Doença Cardíaca")
    print("2. Demo Epidemiológico Sintético") 
    print("3. Demo COVID-19")
    print("4. Executar todos os demos")
    print("5. Análise customizada")
    
    choice = input("\nDigite sua escolha (1-5): ").strip()
    
    if choice == "1":
        demo.demo_heart_disease()
    elif choice == "2":
        demo.demo_synthetic_epidemiological()
    elif choice == "3":
        demo.demo_covid_analysis()
    elif choice == "4":
        demo.run_all_demos()
    elif choice == "5":
        custom_analysis()
    else:
        print("❌ Opção inválida!")

def custom_analysis():
    """Análise customizada pelo usuário"""
    from datasets import DatasetLoader
    from statistical_agent import StatisticalAgent
    
    loader = DatasetLoader()
    agent = StatisticalAgent()
    
    print("\n🔧 ANÁLISE CUSTOMIZADA")
    print("=" * 40)
    
    # Escolher dataset
    print("Datasets disponíveis:")
    print("1. Doença Cardíaca (UCI)")
    print("2. Diabetes Pima")
    print("3. Câncer de Mama")
    print("4. Epidemiológico Sintético")
    print("5. COVID-19 Sintético")
    
    dataset_choice = input("Escolha o dataset (1-5): ").strip()
    
    data_loaders = {
        "1": loader.load_heart_disease_uci,
        "2": loader.load_diabetes_pima,
        "3": loader.load_breast_cancer_sklearn,
        "4": loader.create_synthetic_epidemiological_data,
        "5": loader.create_covid_synthetic_data
    }
    
    if dataset_choice in data_loaders:
        print("Carregando dataset...")
        data = data_loaders[dataset_choice]()
        
        if data is not None:
            print(f"✅ Dataset carregado: {data.shape}")
            
            # Solicitar análise
            request = input("\nDescreva sua análise desejada: ").strip()
            
            if request:
                print(f"\n🔍 Executando análise...")
                try:
                    results = agent.run_analysis(data, request)
                    
                    # Exibir resultados
                    for message in results.get("messages", []):
                        print(message)
                        
                except Exception as e:
                    print(f"❌ Erro na análise: {e}")
            else:
                print("❌ Solicitação de análise vazia!")
        else:
            print("❌ Falha ao carregar dataset!")
    else:
        print("❌ Opção de dataset inválida!")

if __name__ == "__main__":
    main()