from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class SequenciaParaVetor(BaseEstimator, TransformerMixin):
    """
    Converte cada sequência temporal em um único vetor.

    Exemplo:
    (20 frames, 126 features) -> 2520 features

    Mantém a posição de cada feature de cada frame,
    permitindo o uso da sequência pelo XGBoost em formato 2D.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(X.shape[0], -1)


def criar_pipeline_xgboost():
    return Pipeline([
        ("flatten", SequenciaParaVetor()),
        ("xgb", XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=1,
        )),
    ])

def criar_pipeline_rf():
    return Pipeline([
        ("flatten", SequenciaParaVetor()),
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(max_depth=None,class_weight="balanced",random_state=42)),
    ])