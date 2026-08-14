import pandas as pd
import numpy as np

def import_from_csv(filepath: str, mode: str = 'lstm'):
    """
    Carrega o dataset a partir de um CSV no formato 3D para o modelo LSTM: (amostras, frames, features).
    """
    df = pd.read_csv(filepath)

    feature_cols = [col for col in df.columns if col not in ['target', 'frame_idx', 'sample_idx']]
    unique_samples = df['sample_idx'].unique()
    num_samples = len(unique_samples)
    num_frames = df['frame_idx'].nunique()
    num_features = len(feature_cols)

    features = np.zeros((num_samples, num_frames, num_features))
    labels = []

    for i, (sample_id, df_sample) in enumerate(df.groupby('sample_idx', sort=False)):
        df_sample = df_sample.sort_values(by='frame_idx')

        landmarks_matrix = df_sample[feature_cols].values
        t_real = landmarks_matrix.shape[0]
        features[i, : t_real, :] = landmarks_matrix

        sample_label = df_sample['target'].iloc[0]
        labels.append(sample_label)

    labels = np.array(labels)
    return features, labels


if __name__ == "__main__":
    csv_path = 'dataset/dataset_completo_lstm.csv'
    if pd.os.path.exists(csv_path):
        features_lstm, labels_lstm = import_from_csv(filepath=csv_path)
        print(f'Shape das Features de LSTM: {features_lstm.shape}')
        print(f'Shape das Labels de LSTM: {labels_lstm.shape}')