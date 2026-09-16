import numpy as np

def ensemble_predict(lstm_probs, kmeans_class, kmeans_confidence=0.5):
    """
    Combina a previsão da LSTM com a previsão do K-Means.
    """

    lstm_class = int(np.argmax(lstm_probs))
    lstm_confidence = float(np.max(lstm_probs))

    if lstm_class == kmeans_class:
        return lstm_class

    if lstm_confidence >= kmeans_confidence:
        return lstm_class

    return kmeans_class