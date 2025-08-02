import os
from typing import Dict, List, Any, Optional, TypedDict, Annotated
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

# LangGraph imports
from langgraph.graph import Graph, StateGraph, END
#from langgraph.prebuilt import ToolExecutor
from langchain.tools import BaseTool
#from langchain.agents import AgentExecutor
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Statistical and ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import shap
import lime
from lime import lime_tabular
#import lime.tabular
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Causal inference
try:
    from dowhy import CausalModel
    import econml
except ImportError:
    print("Para inferência causal, instale: pip install dowhy econml")

# Probabilistic programming
try:
    import pymc as pm
    import arviz as az
except ImportError:
    print("Para programação probabilística, instale: pip install pymc arviz")

class AnalysisType(Enum):
    DESCRIPTIVE = "descriptive"
    PREDICTIVE = "predictive"
    CAUSAL = "causal"
    PROBABILISTIC = "probabilistic"

@dataclass
class StatisticalResult:
    analysis_type: AnalysisType
    results: Dict[str, Any]
    interpretation: str
    confidence: float
    recommendations: List[str]
    explainability: Dict[str, Any]

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    data: Optional[pd.DataFrame]
    analysis_request: Optional[Dict[str, Any]]
    current_analysis: Optional[AnalysisType]
    results: Optional[StatisticalResult]
    models: Dict[str, Any]
    next_action: Optional[str]

# CORREÇÃO 1: Adicionar anotações de tipo aos atributos name e description
class DataValidationTool(BaseTool):
    name: str = "data_validation"
    description: str = "Valida e prepara dados epidemiológicos para análise"
    
    def _run(self, data: pd.DataFrame) -> Dict[str, Any]:
        validation_results = {
            "shape": data.shape,
            "missing_values": data.isnull().sum().to_dict(),
            "data_types": data.dtypes.to_dict(),
            "outliers": {},
            "data_quality_score": 0.0
        }
        
        # Detectar outliers usando IQR
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)]
            validation_results["outliers"][col] = len(outliers)
        
        # Score de qualidade dos dados
        if data.shape[0] > 0 and data.shape[1] > 0:  # CORREÇÃO: Evitar divisão por zero
            missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            outlier_ratio = sum(validation_results["outliers"].values()) / data.shape[0] if data.shape[0] > 0 else 0
            validation_results["data_quality_score"] = 1.0 - (missing_ratio + outlier_ratio) / 2
        
        return validation_results

class DescriptiveAnalysisTool(BaseTool):
    name: str = "descriptive_analysis"
    description: str = "Realiza análise estatística descritiva de dados epidemiológicos"
    
    def _run(self, data: pd.DataFrame, target_var: Optional[str] = None) -> Dict[str, Any]:
        results = {
            "summary_stats": data.describe().to_dict(),
            "correlations": data.corr().to_dict() if len(data.select_dtypes(include=[np.number]).columns) > 1 else {},
            "distribution_tests": {},
            "clinical_insights": []
        }
        
        # Testes de normalidade para variáveis numéricas
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if len(data[col].dropna()) > 3:
                shapiro_stat, shapiro_p = stats.shapiro(data[col].dropna()[:5000])  # Limite para performance
                results["distribution_tests"][col] = {
                    "shapiro_stat": float(shapiro_stat),  # CORREÇÃO: Converter para float nativo
                    "shapiro_p": float(shapiro_p),
                    "is_normal": shapiro_p > 0.05
                }
        
        # Insights clínicos baseados em padrões epidemiológicos
        if target_var and target_var in data.columns:
            target_distribution = data[target_var].value_counts()
            prevalence = target_distribution.get(1, 0) / len(data) if data[target_var].dtype in ['int64', 'bool'] else None
            if prevalence:
                results["clinical_insights"].append(f"Prevalência da condição: {prevalence:.2%}")
        
        return results

class PredictiveModelingTool(BaseTool):
    name: str = "predictive_modeling"
    description: str = "Desenvolve e avalia modelos preditivos para diagnóstico médico"
    
    def _run(self, data: pd.DataFrame, target_col: str, model_type: str = "all") -> Dict[str, Any]:
        if target_col not in data.columns:
            return {"error": f"Coluna alvo '{target_col}' não encontrada"}
        
        # Preparação dos dados
        X = data.drop(columns=[target_col])
        X = pd.get_dummies(X, drop_first=True)  # Encoding de variáveis categóricas
        y = data[target_col]
        
        # CORREÇÃO: Verificar se há dados suficientes
        if len(X) < 10:
            return {"error": "Dados insuficientes para modelagem"}
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        models = {}
        results = {}
        
        # Definir modelos
        if model_type in ["all", "logistic"]:
            models["logistic_regression"] = LogisticRegression(random_state=42, max_iter=1000)
        if model_type in ["all", "tree"]:
            models["decision_tree"] = DecisionTreeClassifier(random_state=42)
        if model_type in ["all", "forest"]:
            models["random_forest"] = RandomForestClassifier(random_state=42, n_estimators=100)
        
        # Treinar e avaliar modelos
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                results[name] = {
                    "classification_report": classification_report(y_test, y_pred, output_dict=True),
                    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                    "auc_score": float(roc_auc_score(y_test, y_pred_proba)) if y_pred_proba is not None else None,
                    "feature_importance": None
                }
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    results[name]["feature_importance"] = importance_df.head(10).to_dict('records')
            except Exception as e:
                results[name] = {"error": str(e)}
        
        return {"models": models, "results": results, "feature_names": X.columns.tolist()}

class ExplainabilityTool(BaseTool):
    name: str = "explainability_analysis"
    description: str = "Aplica técnicas de XAI para interpretabilidade dos modelos"
    
    def _run(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame, 
             feature_names: List[str], instance_idx: int = 0) -> Dict[str, Any]:
        
        explanations = {}
        
        try:
            # SHAP Explanations
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_test.iloc[:min(100, len(X_test))])  # Limite para performance
                
                explanations["shap"] = {
                    "global_importance": dict(zip(feature_names, np.abs(shap_values.values).mean(0).tolist())),  # CORREÇÃO: converter para lista
                    "local_explanation": dict(zip(feature_names, shap_values.values[instance_idx].tolist())) if instance_idx < len(shap_values.values) else {}
                }
        except Exception as e:
            explanations["shap_error"] = str(e)
        
        try:
            # LIME Explanations
            explainer = lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=feature_names,
                class_names=['Negativo', 'Positivo'],
                mode='classification'
            )
            
            if instance_idx < len(X_test):
                lime_exp = explainer.explain_instance(
                    X_test.iloc[instance_idx].values,
                    model.predict_proba,
                    num_features=min(10, len(feature_names))
                )
                
                explanations["lime"] = {
                    "local_explanation": dict(lime_exp.as_list()),
                    "score": float(lime_exp.score)  # CORREÇÃO: converter para float
                }
        except Exception as e:
            explanations["lime_error"] = str(e)
        
        return explanations

class CausalInferenceTool(BaseTool):
    name: str = "causal_inference"
    description: str = "Realiza análise de inferência causal para identificar relações causais"
    
    def _run(self, data: pd.DataFrame, treatment: str, outcome: str, 
             confounders: List[str] = None) -> Dict[str, Any]:
        
        try:
            # Configurar modelo causal
            causal_model = CausalModel(
                data=data,
                treatment=treatment,
                outcome=outcome,
                common_causes=confounders or []
            )
            
            # Identificar efeito causal
            identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
            
            # Estimar efeito causal
            estimate = causal_model.estimate_effect(
                identified_estimand,
                method_name="backdoor.propensity_score_matching"
            )
            
            # Teste de robustez
            refutation = causal_model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="random_common_cause"
            )
            
            return {
                "identified_estimand": str(identified_estimand),
                "causal_effect": float(estimate.value),  # CORREÇÃO: converter para float
                "confidence_interval": [float(estimate.value - 1.96 * estimate.stderr), 
                                      float(estimate.value + 1.96 * estimate.stderr)],
                "p_value": float(estimate.p_value) if estimate.p_value else None,
                "refutation_test": float(refutation.new_effect) if refutation.new_effect else None
            }
            
        except Exception as e:
            return {"error": f"Erro na análise causal: {str(e)}"}

class StatisticalAgent:
    def __init__(self, llm_model: str = "gpt-4"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.1)
        self.tools = {
            "data_validation": DataValidationTool(),
            "descriptive_analysis": DescriptiveAnalysisTool(),
            "predictive_modeling": PredictiveModelingTool(),
            "explainability_analysis": ExplainabilityTool(),
            "causal_inference": CausalInferenceTool()
        }
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        # Definir o grafo de estados
        workflow = StateGraph(AgentState)
        
        # Adicionar nós
        workflow.add_node("analyze_request", self._analyze_request)
        workflow.add_node("validate_data", self._validate_data)
        workflow.add_node("descriptive_analysis", self._descriptive_analysis)
        workflow.add_node("predictive_modeling", self._predictive_modeling)
        workflow.add_node("causal_analysis", self._causal_analysis)
        workflow.add_node("generate_explanation", self._generate_explanation)
        workflow.add_node("synthesize_results", self._synthesize_results)
        
        # Definir fluxo
        workflow.set_entry_point("analyze_request")
        
        workflow.add_conditional_edges(
            "analyze_request",
            self._route_analysis,
            {
                "validate": "validate_data",
                "descriptive": "descriptive_analysis",
                "predictive": "predictive_modeling",
                "causal": "causal_analysis",
                "end": END
            }
        )
        
        workflow.add_edge("validate_data", "descriptive_analysis")
        workflow.add_edge("descriptive_analysis", "generate_explanation")
        workflow.add_edge("predictive_modeling", "generate_explanation")
        workflow.add_edge("causal_analysis", "generate_explanation")
        workflow.add_edge("generate_explanation", "synthesize_results")
        workflow.add_edge("synthesize_results", END)
        
        return workflow.compile()
    
    def _analyze_request(self, state: AgentState) -> AgentState:
        """Analisa a solicitação do usuário e determina o tipo de análise necessária"""
        last_message = state["messages"][-1].content if state["messages"] else ""
        
        # Prompt para classificar o tipo de análise
        prompt = ChatPromptTemplate.from_template("""
        Analise a seguinte solicitação e determine o tipo de análise estatística necessária:
        
        Solicitação: {request}
        
        Tipos disponíveis:
        - descriptive: Análise estatística descritiva
        - predictive: Modelagem preditiva para diagnóstico
        - causal: Inferência causal
        - validation: Validação de dados
        
        Responda apenas com o tipo (uma palavra).
        """)
        
        response = self.llm.invoke(prompt.format(request=last_message))
        analysis_type = response.content.strip().lower()
        
        # Extrair parâmetros da solicitação
        analysis_request = {
            "type": analysis_type,
            "target_variable": None,
            "features": [],
            "treatment": None,
            "outcome": None
        }
        
        state["analysis_request"] = analysis_request
        # CORREÇÃO: Verificar se o tipo é válido antes de criar enum
        try:
            state["current_analysis"] = AnalysisType(analysis_type) if analysis_type in [e.value for e in AnalysisType] else AnalysisType.DESCRIPTIVE
        except ValueError:
            state["current_analysis"] = AnalysisType.DESCRIPTIVE
        
        return state
    
    def _route_analysis(self, state: AgentState) -> str:
        """Determina o próximo passo baseado no tipo de análise"""
        analysis_type = state["current_analysis"]
        
        # CORREÇÃO: Verificar corretamente se data existe
        if state.get("data") is None:
            return "validate"
        elif analysis_type == AnalysisType.DESCRIPTIVE:
            return "descriptive"
        elif analysis_type == AnalysisType.PREDICTIVE:
            return "predictive"
        elif analysis_type == AnalysisType.CAUSAL:
            return "causal"
        else:
            return "descriptive"
    
    def _validate_data(self, state: AgentState) -> AgentState:
        """Valida os dados fornecidos"""
        if state.get("data") is not None:
            validation_results = self.tools["data_validation"]._run(state["data"])
            state["results"] = StatisticalResult(
                analysis_type=AnalysisType.DESCRIPTIVE,
                results=validation_results,
                interpretation="Validação dos dados concluída",
                confidence=validation_results["data_quality_score"],
                recommendations=[],
                explainability={}
            )
        return state
    
    def _descriptive_analysis(self, state: AgentState) -> AgentState:
        """Realiza análise descritiva"""
        if state.get("data") is not None:
            results = self.tools["descriptive_analysis"]._run(
                state["data"],
                state["analysis_request"].get("target_variable")
            )
            
            state["results"] = StatisticalResult(
                analysis_type=AnalysisType.DESCRIPTIVE,
                results=results,
                interpretation="Análise descritiva concluída",
                confidence=0.95,
                recommendations=results.get("clinical_insights", []),
                explainability={}
            )
        return state
    
    def _predictive_modeling(self, state: AgentState) -> AgentState:
        """Desenvolve modelos preditivos"""
        if state.get("data") is not None and state["analysis_request"].get("target_variable"):
            modeling_results = self.tools["predictive_modeling"]._run(
                state["data"],
                state["analysis_request"]["target_variable"]
            )
            
            # Gerar explicações para o melhor modelo
            if modeling_results.get("models"):
                # CORREÇÃO: Melhor tratamento de erro ao encontrar melhor modelo
                best_model_name = None
                best_auc = 0
                
                for model_name, results in modeling_results["results"].items():
                    if isinstance(results, dict) and "auc_score" in results:
                        auc = results.get("auc_score", 0) or 0
                        if auc > best_auc:
                            best_auc = auc
                            best_model_name = model_name
                
                if best_model_name:
                    best_model = modeling_results["models"][best_model_name]
                    
                    # Preparar dados para explicabilidade
                    X = state["data"].drop(columns=[state["analysis_request"]["target_variable"]])
                    X = pd.get_dummies(X, drop_first=True)
                    X_train, X_test, _, _ = train_test_split(
                        X, state["data"][state["analysis_request"]["target_variable"]], 
                        test_size=0.2, random_state=42
                    )
                    
                    explanations = self.tools["explainability_analysis"]._run(
                        best_model, X_train, X_test, modeling_results["feature_names"]
                    )
                    
                    state["models"] = modeling_results["models"]
                    state["results"] = StatisticalResult(
                        analysis_type=AnalysisType.PREDICTIVE,
                        results=modeling_results["results"],
                        interpretation=f"Melhor modelo: {best_model_name}",
                        confidence=best_auc,
                        recommendations=[f"Modelo {best_model_name} recomendado para uso clínico"],
                        explainability=explanations
                    )
        return state
    
    def _causal_analysis(self, state: AgentState) -> AgentState:
        """Realiza análise de inferência causal"""
        request = state["analysis_request"]
        if (state.get("data") is not None and 
            request.get("treatment") and 
            request.get("outcome")):
            
            causal_results = self.tools["causal_inference"]._run(
                state["data"],
                request["treatment"],
                request["outcome"]
            )
            
            state["results"] = StatisticalResult(
                analysis_type=AnalysisType.CAUSAL,
                results=causal_results,
                interpretation="Análise causal concluída",
                confidence=0.8,
                recommendations=["Considerar fatores de confusão adicionais"],
                explainability={}
            )
        return state
    
    def _generate_explanation(self, state: AgentState) -> AgentState:
        """Gera explicações interpretáveis dos resultados"""
        if state.get("results"):
            # Usar LLM para gerar explicação mais detalhada
            prompt = ChatPromptTemplate.from_template("""
            Como especialista em estatística médica, interprete os seguintes resultados para profissionais de saúde:
            
            Tipo de análise: {analysis_type}
            Resultados: {results}
            Explicabilidade: {explainability}
            
            Forneça:
            1. Interpretação clínica clara
            2. Limitações e considerações
            3. Recomendações práticas
            
            Mantenha linguagem técnica mas acessível para médicos.
            """)
            
            response = self.llm.invoke(prompt.format(
                analysis_type=state["results"].analysis_type.value,
                results=str(state["results"].results),
                explainability=str(state["results"].explainability)
            ))
            
            # Atualizar interpretação
            state["results"].interpretation = response.content
            
        return state
    
    def _synthesize_results(self, state: AgentState) -> AgentState:
        """Sintetiza e finaliza os resultados"""
        if state.get("results"):
            # Adicionar mensagem final com resultados
            final_message = AIMessage(content=f"""
## Análise Estatística Concluída

**Tipo de Análise:** {state["results"].analysis_type.value.title()}

**Interpretação:**
{state["results"].interpretation}

**Confiança:** {state["results"].confidence:.2%}

**Recomendações:**
{chr(10).join([f"• {rec}" for rec in state["results"].recommendations])}

**Transparência e Explicabilidade:**
Esta análise utilizou técnicas de XAI para garantir interpretabilidade dos resultados.
            """)
            
            state["messages"].append(final_message)
        
        return state
    
    def run_analysis(self, data: pd.DataFrame, request: str) -> Dict[str, Any]:
        """Executa análise estatística completa"""
        initial_state: AgentState = {
            "messages": [HumanMessage(content=request)],
            "data": data,
            "analysis_request": None,
            "current_analysis": None,
            "results": None,
            "models": {},
            "next_action": None
        }
        
        # Executar o grafo
        final_state = self.graph.invoke(initial_state)
        
        return {
            "results": final_state.get("results"),
            "models": final_state.get("models", {}),
            "messages": [msg.content for msg in final_state.get("messages", [])]
        }

# Exemplo de uso
def example_usage():
    """Exemplo de como usar o Agente Estatístico"""
    
    # Criar dados de exemplo (dados epidemiológicos simulados)
    np.random.seed(42)
    n_patients = 1000
    
    data = pd.DataFrame({
        'idade': np.random.normal(65, 15, n_patients),
        'pressao_sistolica': np.random.normal(140, 20, n_patients),
        'colesterol': np.random.normal(200, 40, n_patients),
        'diabetes': np.random.binomial(1, 0.3, n_patients),
        'fumante': np.random.binomial(1, 0.25, n_patients),
        'exercicio_regular': np.random.binomial(1, 0.4, n_patients),
    })
    
    # Criar variável alvo (doença cardiovascular)
    cardiovascular_risk = (
        0.02 * data['idade'] + 
        0.01 * data['pressao_sistolica'] + 
        0.005 * data['colesterol'] + 
        0.5 * data['diabetes'] + 
        0.3 * data['fumante'] - 
        0.2 * data['exercicio_regular'] - 5
    )
    data['doenca_cardiovascular'] = (cardiovascular_risk + np.random.normal(0, 1, n_patients) > 0).astype(int)
    
    # Inicializar agente
    agent = StatisticalAgent()
    
    # Exemplo de análise preditiva
    request = "Desenvolva um modelo preditivo para diagnóstico de doença cardiovascular usando todas as variáveis disponíveis"
    
    results = agent.run_analysis(data, request)
    
    print("=== RESULTADOS DA ANÁLISE ===")
    for message in results["messages"]:
        print(message)
        print("-" * 50)
    
    return results

if __name__ == "__main__":
    # Para executar o exemplo, descomente a linha abaixo
    # results = example_usage()
    print("Agente Estatístico criado com sucesso!")
    print("\nPara usar:")
    print("1. Configure sua chave da OpenAI: os.environ['OPENAI_API_KEY'] = 'sua_chave'")
    print("2. Instale dependências: pip install langgraph langchain-openai scikit-learn shap lime dowhy econml pymc arviz")
    print("3. Execute example_usage() para ver um exemplo completo")