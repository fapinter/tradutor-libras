from sklearn.base import BaseEstimator, TransformerMixin
from tslearn.clustering import TimeSeriesKMeans
from utils.constants import SEED
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# Time Series KMeans
class AgrupadorSeriesTemporais(BaseEstimator, TransformerMixin):
    """
    Agrupa as séries temporais completas usando TimeSeriesKMeans com DTW.
    Gera como características as distâncias de cada vídeo para os K centroides.
    """
    def __init__(self, n_clusters=20, metric="dtw"):
        self.n_clusters = n_clusters
        self.metric = metric

    def fit(self, X, y=None):
        # O tslearn já espera o formato 3D (n_amostras, n_frames, n_features)
        # que é o formato original do seu X, não precisando do reshape(-1, ...)
        self.kmeans_ = TimeSeriesKMeans(
            n_clusters=self.n_clusters, 
            metric=self.metric, 
            random_state=SEED,
            n_init=3 # Reduzido para evitar lentidão extrema com DTW
        )
        self.kmeans_.fit(X)
        return self

    def transform(self, X):
        # transform() no TimeSeriesKMeans retorna uma matriz (n_amostras, n_clusters)
        # contendo as distâncias da amostra para cada um dos centroides.
        # Estas distâncias serão as novas features para o Random Forest ou KNN.
        distancias = self.kmeans_.transform(X)
        return distancias


def criar_time_kmeans_rf():
    return Pipeline([
        ("compactar", AgrupadorSeriesTemporais(n_clusters=30, metric="dtw")), # softdtw ou euclidean tbm
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(max_depth=None, class_weight="balanced", random_state=SEED)),
    ])

def criar_time_kmeans_knn():
    return Pipeline([
        ("compactar", AgrupadorSeriesTemporais(n_clusters=30, metric="dtw")), # softdtw ou euclidean tbm
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(weights="distance")),
    ])


# KMeans RF e KNN
class SequenciaParaHistograma(BaseEstimator, TransformerMixin):
    """
    Agrupa os frames em poses (K-Means) e resume cada sequência num histograma de
    ocupação + transição entre poses consecutivas.
    """

    def __init__(self, n_clusters=110):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        frames_treino = X.reshape(-1, X.shape[-1])
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=SEED, n_init=10)
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
        ("rf", RandomForestClassifier(max_depth=None,class_weight="balanced",random_state=SEED)),
    ])


def criar_pipeline_kmeans_knn():
    return Pipeline([
        ("compactar", SequenciaParaHistograma()),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(weights="distance")),
    ])