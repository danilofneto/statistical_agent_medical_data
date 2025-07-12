import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
import requests
import io
from datetime import datetime, timedelta

class DatasetLoader:
    """Classe para carregar diferentes datasets para testar o Agente Estatístico"""
    
    @staticmethod
    def load_heart_disease_uci():
        """
        Dataset clássico de doença cardíaca da UCI
        Ideal para: Análise preditiva, XAI, inferência causal
        """
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        
        column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        
        try:
            data = pd.read_csv(url, names=column_names, na_values='?')
            
            # Limpeza e preparação
            data = data.dropna()
            data['target'] = (data['target'] > 0).astype(int)  # Binarizar alvo
            
            # Adicionar labels descritivos
            data['sex'] = data['sex'].map({0: 'Feminino', 1: 'Masculino'})
            data['cp'] = data['cp'].map({
                1: 'Angina_Tipica', 2: 'Angina_Atipica', 
                3: 'Dor_Nao_Anginosa', 4: 'Assintomatico'
            })
            
            print("✅ Dataset de Doença Cardíaca carregado com sucesso!")
            print(f"📊 Dimensões: {data.shape}")
            print(f"🎯 Variável alvo: 'target' (0=Sem doença, 1=Com doença)")
            print(f"📈 Prevalência: {data['target'].mean():.2%}")
            
            return data
            
        except Exception as e:
            print(f"❌ Erro ao carregar dataset UCI: {e}")
            return None
    
    @staticmethod
    def load_diabetes_pima():
        """
        Dataset de Diabetes Pima Indians
        Ideal para: Análise epidemiológica, modelos preditivos
        """
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        
        column_names = [
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
            'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome'
        ]
        
        try:
            data = pd.read_csv(url, names=column_names)
            
            # Substituir zeros por NaN onde faz sentido
            zero_to_nan = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']
            for col in zero_to_nan:
                data[col] = data[col].replace(0, np.nan)
            
            # Preencher valores faltantes com mediana
            data = data.fillna(data.median())
            
            print("✅ Dataset de Diabetes Pima carregado com sucesso!")
            print(f"📊 Dimensões: {data.shape}")
            print(f"🎯 Variável alvo: 'outcome' (0=Sem diabetes, 1=Com diabetes)")
            print(f"📈 Prevalência: {data['outcome'].mean():.2%}")
            
            return data
            
        except Exception as e:
            print(f"❌ Erro ao carregar dataset Pima: {e}")
            return None
    
    @staticmethod
    def load_breast_cancer_sklearn():
        """
        Dataset de Câncer de Mama do scikit-learn
        Ideal para: Análise de biomarcadores, explicabilidade
        """
        try:
            cancer = load_breast_cancer()
            data = pd.DataFrame(cancer.data, columns=cancer.feature_names)
            data['target'] = cancer.target
            
            # Renomear alvo para interpretação clínica
            data['malignant'] = (data['target'] == 0).astype(int)  # 0=benigno, 1=maligno
            data = data.drop('target', axis=1)
            
            print("✅ Dataset de Câncer de Mama carregado com sucesso!")
            print(f"📊 Dimensões: {data.shape}")
            print(f"🎯 Variável alvo: 'malignant' (0=Benigno, 1=Maligno)")
            print(f"📈 Prevalência: {data['malignant'].mean():.2%}")
            
            return data
            
        except Exception as e:
            print(f"❌ Erro ao carregar dataset de câncer: {e}")
            return None
    
    @staticmethod
    def create_synthetic_epidemiological_data(n_patients=2000, random_state=42):
        """
        Cria dataset epidemiológico sintético realista
        Ideal para: Testar todas as funcionalidades, controle total dos dados
        """
        np.random.seed(random_state)
        
        # Demographics
        age = np.random.normal(55, 18, n_patients)
        age = np.clip(age, 18, 95)
        
        sex = np.random.binomial(1, 0.52, n_patients)  # 52% feminino
        
        # Socioeconomic factors
        education_years = np.random.poisson(12, n_patients)
        income_level = np.random.choice([1, 2, 3, 4, 5], n_patients, p=[0.2, 0.25, 0.3, 0.15, 0.1])
        
        # Lifestyle factors
        smoking = np.random.binomial(1, 0.15 + 0.1 * (sex == 1), n_patients)  # Homens fumam mais
        alcohol_consumption = np.random.exponential(2, n_patients)
        exercise_hours_week = np.random.gamma(2, 2, n_patients)
        
        # Clinical measurements
        bmi = np.random.normal(26, 5, n_patients)
        bmi = np.clip(bmi, 15, 50)
        
        systolic_bp = 120 + 0.3 * age + 2 * bmi + 5 * smoking + np.random.normal(0, 15, n_patients)
        diastolic_bp = 80 + 0.2 * age + 1 * bmi + 3 * smoking + np.random.normal(0, 10, n_patients)
        
        cholesterol = 180 + 0.5 * age + 1.5 * bmi + np.random.normal(0, 30, n_patients)
        glucose = 90 + 0.2 * age + 0.8 * bmi + np.random.normal(0, 15, n_patients)
        
        # Comorbidities (com correlações realistas)
        diabetes_risk = -8 + 0.05 * age + 0.15 * bmi + 0.02 * glucose + 0.5 * (income_level <= 2)
        diabetes = (diabetes_risk + np.random.logistic(0, 1, n_patients) > 0).astype(int)
        
        hypertension_risk = -6 + 0.04 * age + 0.12 * bmi + 0.01 * systolic_bp + 0.3 * smoking
        hypertension = (hypertension_risk + np.random.logistic(0, 1, n_patients) > 0).astype(int)
        
        # Treatment assignment (para análise causal)
        treatment_propensity = -2 + 0.02 * age + 0.3 * diabetes + 0.4 * hypertension + 0.2 * (income_level >= 4)
        treatment = (treatment_propensity + np.random.logistic(0, 1, n_patients) > 0).astype(int)
        
        # Outcome (cardiovascular events)
        cv_risk = (-10 + 
                  0.08 * age + 
                  0.5 * (sex == 1) +  # Homens maior risco
                  0.15 * bmi + 
                  0.02 * systolic_bp + 
                  0.01 * cholesterol +
                  0.8 * diabetes + 
                  0.6 * hypertension + 
                  1.2 * smoking + 
                  -0.3 * treatment +  # Tratamento reduz risco
                  -0.1 * exercise_hours_week +
                  0.2 * (income_level <= 2))  # Baixa renda aumenta risco
        
        cardiovascular_event = (cv_risk + np.random.logistic(0, 1, n_patients) > 0).astype(int)
        
        # Criar DataFrame
        data = pd.DataFrame({
            'patient_id': range(1, n_patients + 1),
            'age': age.round(0).astype(int),
            'sex': sex,
            'education_years': education_years,
            'income_level': income_level,
            'smoking': smoking,
            'alcohol_consumption': alcohol_consumption.round(1),
            'exercise_hours_week': exercise_hours_week.round(1),
            'bmi': bmi.round(1),
            'systolic_bp': systolic_bp.round(0).astype(int),
            'diastolic_bp': diastolic_bp.round(0).astype(int),
            'cholesterol': cholesterol.round(0).astype(int),
            'glucose': glucose.round(0).astype(int),
            'diabetes': diabetes,
            'hypertension': hypertension,
            'treatment': treatment,
            'cardiovascular_event': cardiovascular_event
        })
        
        # Adicionar algumas missings realistas
        missing_mask = np.random.random((n_patients, len(data.columns))) < 0.02  # 2% missing
        for i, col in enumerate(data.columns):
            if col not in ['patient_id', 'cardiovascular_event']:
                data.loc[missing_mask[:, i], col] = np.nan
        
        print("✅ Dataset epidemiológico sintético criado com sucesso!")
        print(f"📊 Dimensões: {data.shape}")
        print(f"🎯 Variável alvo: 'cardiovascular_event'")
        print(f"📈 Prevalência de eventos cardiovasculares: {data['cardiovascular_event'].mean():.2%}")
        print(f"📈 Prevalência de diabetes: {data['diabetes'].mean():.2%}")
        print(f"📈 Prevalência de hipertensão: {data['hypertension'].mean():.2%}")
        print(f"💊 Taxa de tratamento: {data['treatment'].mean():.2%}")
        
        return data
    
    @staticmethod
    def create_covid_synthetic_data(n_patients=1500, random_state=42):
        """
        Cria dataset sintético de COVID-19 para análise epidemiológica
        Ideal para: Análise de fatores de risco, modelos preditivos de severidade
        """
        np.random.seed(random_state)
        
        # Demographics
        age = np.random.gamma(4, 12, n_patients)  # Distribuição mais realista de idades
        age = np.clip(age, 0, 100).round(0).astype(int)
        
        sex = np.random.binomial(1, 0.51, n_patients)
        
        # Comorbidities (baseado em dados epidemiológicos reais)
        diabetes = np.random.binomial(1, 0.11 + 0.002 * age, n_patients)
        hypertension = np.random.binomial(1, 0.05 + 0.015 * (age > 40), n_patients)
        obesity = np.random.binomial(1, 0.36, n_patients)
        cardiovascular_disease = np.random.binomial(1, 0.03 + 0.01 * (age > 60), n_patients)
        chronic_kidney_disease = np.random.binomial(1, 0.02 + 0.005 * (age > 65), n_patients)
        immunocompromised = np.random.binomial(1, 0.05, n_patients)
        
        # Symptoms at presentation
        fever = np.random.binomial(1, 0.87, n_patients)
        cough = np.random.binomial(1, 0.67, n_patients)
        shortness_breath = np.random.binomial(1, 0.43, n_patients)
        fatigue = np.random.binomial(1, 0.38, n_patients)
        loss_taste_smell = np.random.binomial(1, 0.41, n_patients)
        
        # Laboratory values (with realistic correlations)
        wbc_count = np.random.lognormal(2.1, 0.4, n_patients)  # White blood cells
        lymphocyte_count = np.random.lognormal(1.8, 0.5, n_patients)
        d_dimer = np.random.lognormal(0.5, 0.8, n_patients)
        crp = np.random.lognormal(2.0, 1.2, n_patients)  # C-reactive protein
        ldh = np.random.normal(250, 100, n_patients)  # Lactate dehydrogenase
        
        # Vaccination status
        vaccination_prob = 0.2 + 0.01 * age  # Older people more likely vaccinated
        vaccinated = np.random.binomial(1, vaccination_prob, n_patients)
        
        # Severe outcome (ICU admission or death)
        severe_risk = (-6 + 
                      0.08 * age + 
                      0.3 * (sex == 1) +
                      1.2 * diabetes + 
                      0.8 * hypertension + 
                      0.6 * obesity +
                      1.5 * cardiovascular_disease +
                      1.0 * chronic_kidney_disease +
                      1.3 * immunocompromised +
                      0.5 * shortness_breath +
                      0.02 * d_dimer +
                      0.003 * crp +
                      -1.2 * vaccinated)  # Vaccination protective
        
        severe_outcome = (severe_risk + np.random.logistic(0, 1, n_patients) > 0).astype(int)
        
        # Length of stay
        los_mean = 7 + 3 * severe_outcome + 0.1 * age
        length_of_stay = np.random.gamma(2, los_mean/2, n_patients).round(0).astype(int)
        
        data = pd.DataFrame({
            'patient_id': range(1, n_patients + 1),
            'age': age,
            'sex': sex,
            'diabetes': diabetes,
            'hypertension': hypertension,
            'obesity': obesity,
            'cardiovascular_disease': cardiovascular_disease,
            'chronic_kidney_disease': chronic_kidney_disease,
            'immunocompromised': immunocompromised,
            'fever': fever,
            'cough': cough,
            'shortness_of_breath': shortness_breath,
            'fatigue': fatigue,
            'loss_taste_smell': loss_taste_smell,
            'wbc_count': wbc_count.round(2),
            'lymphocyte_count': lymphocyte_count.round(2),
            'd_dimer': d_dimer.round(2),
            'crp': crp.round(1),
            'ldh': ldh.round(0).astype(int),
            'vaccinated': vaccinated,
            'severe_outcome': severe_outcome,
            'length_of_stay': length_of_stay
        })
        
        print("✅ Dataset COVID-19 sintético criado com sucesso!")
        print(f"📊 Dimensões: {data.shape}")
        print(f"🎯 Variável alvo: 'severe_outcome' (ICU/Morte)")
        print(f"📈 Taxa de casos severos: {data['severe_outcome'].mean():.2%}")
        print(f"💉 Taxa de vacinação: {data['vaccinated'].mean():.2%}")
        
        return data

def test_all_datasets():
    """Testa todos os datasets disponíveis"""
    loader = DatasetLoader()
    
    print("=" * 60)
    print("🧪 TESTANDO TODOS OS DATASETS DISPONÍVEIS")
    print("=" * 60)
    
    datasets = {}
    
    # 1. Heart Disease UCI
    print("\n1. 🫀 DATASET DE DOENÇA CARDÍACA (UCI)")
    print("-" * 40)
    datasets['heart_disease'] = loader.load_heart_disease_uci()
    
    # 2. Diabetes Pima
    print("\n2. 🍯 DATASET DE DIABETES (PIMA)")
    print("-" * 40)
    datasets['diabetes'] = loader.load_diabetes_pima()
    
    # 3. Breast Cancer
    print("\n3. 🎗️ DATASET DE CÂNCER DE MAMA")
    print("-" * 40)
    datasets['breast_cancer'] = loader.load_breast_cancer_sklearn()
    
    # 4. Synthetic Epidemiological
    print("\n4. 📊 DATASET EPIDEMIOLÓGICO SINTÉTICO")
    print("-" * 40)
    datasets['synthetic_epi'] = loader.create_synthetic_epidemiological_data()
    
    # 5. COVID Synthetic
    print("\n5. 🦠 DATASET COVID-19 SINTÉTICO")
    print("-" * 40)
    datasets['covid'] = loader.create_covid_synthetic_data()
    
    return datasets

def get_analysis_examples():
    """Exemplos de solicitações para cada dataset"""
    
    examples = {
        'heart_disease': [
            "Desenvolva um modelo preditivo para diagnóstico de doença cardíaca usando todas as variáveis clínicas disponíveis",
            "Realize análise descritiva dos fatores de risco cardiovascular por sexo e idade",
            "Analise o efeito causal do tipo de dor torácica no diagnóstico de doença cardíaca"
        ],
        
        'diabetes': [
            "Crie modelo de machine learning para predizer diabetes tipo 2 em mulheres indígenas",
            "Identifique os principais biomarcadores preditivos de diabetes e explique sua importância clínica",
            "Analise a correlação entre IMC, idade e glicemia na predição de diabetes"
        ],
        
        'breast_cancer': [
            "Desenvolva modelo explicável para diagnóstico de malignidade em tumores de mama",
            "Identifique quais características dos núcleos celulares são mais preditivas de malignidade",
            "Realize análise de componentes principais dos biomarcadores tumorais"
        ],
        
        'synthetic_epi': [
            "Analise os determinantes sociais de saúde em eventos cardiovasculares",
            "Estime o efeito causal do tratamento na redução de eventos cardiovasculares",
            "Desenvolva modelo preditivo de risco cardiovascular considerando fatores socioeconômicos"
        ],
        
        'covid': [
            "Desenvolva modelo preditivo para casos severos de COVID-19 baseado em comorbidades",
            "Analise o efeito protetor da vacinação em desfechos severos",
            "Identifique biomarcadores laboratoriais mais preditivos de severidade"
        ]
    }
    
    return examples

# Exemplo completo de uso
def complete_example():
    """Exemplo completo de uso do sistema"""
    
    print("🚀 EXEMPLO COMPLETO DE USO DO AGENTE ESTATÍSTICO")
    print("=" * 60)
    
    # Carregar dataset
    loader = DatasetLoader()
    data = loader.create_synthetic_epidemiological_data(n_patients=1000)
    
    if data is not None:
        print(f"\n📋 RESUMO DOS DADOS:")
        print(f"Shape: {data.shape}")
        print(f"Colunas: {list(data.columns)}")
        print(f"\n📊 Primeiras linhas:")
        print(data.head())
        
        print(f"\n📈 ESTATÍSTICAS DESCRITIVAS:")
        print(data.describe())
        
        # Exemplos de solicitações
        examples = get_analysis_examples()
        print(f"\n💡 EXEMPLOS DE SOLICITAÇÕES PARA ESTE DATASET:")
        for i, example in enumerate(examples['synthetic_epi'], 1):
            print(f"{i}. {example}")
        
        # Preparar para uso com o agente
        print(f"\n🤖 CÓDIGO PARA USAR COM O AGENTE:")
        print("""
# Inicializar agente
from statistical_agent import StatisticalAgent
agent = StatisticalAgent()

# Executar análise
request = "Desenvolva modelo preditivo de risco cardiovascular considerando fatores socioeconômicos"
results = agent.run_analysis(data, request)

# Ver resultados
for message in results["messages"]:
    print(message)
        """)
    
    return data

if __name__ == "__main__":
    # Testar todos os datasets
    datasets = test_all_datasets()
    
    # Mostrar exemplos de uso
    print("\n" + "=" * 60)
    print("💡 EXEMPLOS DE SOLICITAÇÕES PARA CADA DATASET")
    print("=" * 60)
    
    examples = get_analysis_examples()
    for dataset_name, dataset_examples in examples.items():
        print(f"\n🔸 {dataset_name.upper()}:")
        for i, example in enumerate(dataset_examples, 1):
            print(f"   {i}. {example}")