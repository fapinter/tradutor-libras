from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

class SequenciaParaHistograma(BaseEstimator, TransformerMixin):
    """
    Agrupa os frames em poses (K-Means) e resume cada sequência num histograma de
    ocupação + transição entre poses consecutivas.
    """

    def __init__(self, n_clusters=110):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        frames_treino = X.reshape(-1, X.shape[-1])
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.kmeans_.fit(frames_treino, y)
        return self

    def transform(self, X):
        n_clusters = self.n_clusters
        vetores = np.zeros(
            (len(X), n_clusters + n_clusters * n_clusters))

        for i, seq in enumerate(X):
            clusters_seq = self.kmeans_.predict(seq)

            for c in clusters_seq:
                vetores[i, c] += 1

            vetores[i, :n_clusters] /= len(clusters_seq)

            for c_atual, c_seguinte in zip(clusters_seq[:-1],clusters_seq[1:]):
                vetores[i, n_clusters + c_atual * n_clusters +c_seguinte] += 1
            n_transicoes = max(len(clusters_seq) - 1, 1)
            vetores[i, n_clusters:] /= n_transicoes

        return vetores

def criar_pipeline_kmeans():
    return Pipeline([
        ("compactar", SequenciaParaHistograma()),
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(max_depth=None,class_weight="balanced",random_state=42)),
    ])


def criar_pipeline_kmeans_knn():
    return Pipeline([
        ("compactar", SequenciaParaHistograma()),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(weights="distance")),
    ])