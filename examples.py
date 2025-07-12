import os
import pandas as pd
from statistical_agent import StatisticalAgent  # Importa seu agente
from datasets import DatasetLoader  # Importa os carregadores

class AgentDemo:
    """Classe para demonstrar o uso do Agente Estatístico"""
    
    def __init__(self):
        # Configurar API key (se necessário)
        if not os.getenv('OPENAI_API_KEY'):
            print("⚠️  Configure sua OPENAI_API_KEY antes de usar o agente!")
        
        self.agent = StatisticalAgent()
        self.loader = DatasetLoader()
    
    def demo_heart_disease(self):
        """Demo com dataset de doença cardíaca"""
        print("🫀 DEMO: Análise de Doença Cardíaca")
        print("=" * 50)
        
        # Carregar dados
        data = self.loader.load_heart_disease_uci()
        if data is None:
            print("❌ Falha ao carregar dataset")
            return
        
        # Solicitações de exemplo
        requests = [
            "Realize análise descritiva dos fatores de risco cardiovascular por sexo",
            "Desenvolva modelo preditivo para diagnóstico de doença cardíaca com explicabilidade",
            "Identifique os 3 fatores mais importantes para o diagnóstico"
        ]
        
        for i, request in enumerate(requests, 1):
            print(f"\n📋 Análise {i}: {request}")
            print("-" * 40)
            
            try:
                results = self.agent.run_analysis(data, request)
                
                # Exibir resultados
                if results.get("results"):
                    print(f"✅ Análise concluída!")
                    print(f"Confiança: {results['results'].confidence:.2%}")
                    print(f"Tipo: {results['results'].analysis_type.value}")
                    
                # Salvar resultados
                self._save_results(f"heart_disease_analysis_{i}", results)
                
            except Exception as e:
                print(f"❌ Erro na análise: {e}")
    
    def demo_synthetic_epidemiological(self):
        """Demo com dataset epidemiológico sintético"""
        print("📊 DEMO: Análise Epidemiológica Completa")
        print("=" * 50)
        
        # Carregar dados sintéticos
        data = self.loader.create_synthetic_epidemiological_data(n_patients=1500)
        
        # Análises progressivas
        analyses = [
            {
                "title": "Análise Descritiva",
                "request": "Realize análise estatística descritiva completa dos dados epidemiológicos",
                "target": None
            },
            {
                "title": "Modelagem Preditiva", 
                "request": "Desenvolva modelo Random Forest para predizer eventos cardiovasculares usando todas as variáveis clínicas",
                "target": "cardiovascular_event"
            },
            {
                "title": "Inferência Causal",
                "request": "Estime o efeito causal do tratamento em eventos cardiovasculares controlando por confundidores",
                "target": "cardiovascular_event"
            },
            {
                "title": "Análise de Subgrupos",
                "request": "Compare fatores de risco entre homens e mulheres para estratificação",
                "target": "cardiovascular_event"
            }
        ]
        
        results_summary = {}
        
        for analysis in analyses:
            print(f"\n🔍 {analysis['title']}")
            print(f"Solicitação: {analysis['request']}")
            print("-" * 60)
            
            try:
                # Executar análise
                results = self.agent.run_analysis(data, analysis['request'])
                
                if results.get("results"):
                    print(f"✅ {analysis['title']} concluída!")
                    print(f"Confiança: {results['results'].confidence:.2%}")
                    
                    # Armazenar resumo
                    results_summary[analysis['title']] = {
                        'confidence': results['results'].confidence,
                        'type': results['results'].analysis_type.value,
                        'recommendations': len(results['results'].recommendations)
                    }
                
                # Salvar resultados detalhados
                self._save_results(f"synthetic_epi_{analysis['title'].lower().replace(' ', '_')}", results)
                
            except Exception as e:
                print(f"❌ Erro em {analysis['title']}: {e}")
                results_summary[analysis['title']] = {'error': str(e)}
        
        # Resumo final
        print(f"\n📈 RESUMO FINAL DAS ANÁLISES")
        print("=" * 50)
        for title, summary in results_summary.items():
            if 'error' not in summary:
                print(f"✅ {title}: Confiança {summary['confidence']:.1%}, Tipo: {summary['type']}")
            else:
                print(f"❌ {title}: {summary['error']}")
    
    def demo_covid_analysis(self):
        """Demo com dataset COVID-19"""
        print("🦠 DEMO: Análise COVID-19")
        print("=" * 50)
        
        data = self.loader.create_covid_synthetic_data(n_patients=1200)
        
        request = "Desenvolva modelo explicável para predizer casos severos de COVID-19 e analise o efeito protetor da vacinação"
        
        print(f"Solicitação: {request}")
        print("-" * 60)
        
        try:
            results = self.agent.run_analysis(data, request)
            
            if results.get("results"):
                print(f"✅ Análise COVID-19 concluída!")
                print(f"Confiança: {results['results'].confidence:.2%}")
                
                # Análise específica do efeito da vacinação
                if results.get("models"):
                    print(f"🏥 Modelos desenvolvidos: {list(results['models'].keys())}")
            
            self._save_results("covid_analysis", results)
            
        except Exception as e:
            print(f"❌ Erro na análise COVID: {e}")
    
    def _save_results(self, filename: str, results: dict):
        """Salva resultados em arquivo"""
        try:
            import json
            import os
            
            # Criar pasta de resultados se não existir
            os.makedirs("results", exist_ok=True)
            
            # Converter resultados para formato JSON serializável
            serializable_results = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "analysis_type": results.get("results").analysis_type.value if results.get("results") else None,
                "confidence": results.get("results").confidence if results.get("results") else None,
                "recommendations": results.get("results").recommendations if results.get("results") else [],
                "models_used": list(results.get("models", {}).keys()),
                "messages": results.get("messages", [])
            }
            
            with open(f"results/{filename}.json", "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Resultados salvos em: results/{filename}.json")
            
        except Exception as e:
            print(f"⚠️  Não foi possível salvar resultados: {e}")
    
    def run_all_demos(self):
        """Executa todas as demonstrações"""
        print("🚀 EXECUTANDO TODAS AS DEMONSTRAÇÕES")
        print("=" * 60)
        
        demos = [
            ("Doença Cardíaca", self.demo_heart_disease),
            ("Epidemiológico Sintético", self.demo_synthetic_epidemiological), 
            ("COVID-19", self.demo_covid_analysis)
        ]
        
        for name, demo_func in demos:
            print(f"\n🎯 Iniciando demo: {name}")
            try:
                demo_func()
                print(f"✅ Demo {name} concluída com sucesso!")
            except Exception as e:
                print(f"❌ Erro no demo {name}: {e}")
            print("\n" + "="*60)
