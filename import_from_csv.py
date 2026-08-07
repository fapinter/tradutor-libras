import pandas as pd
import numpy as np

def import_from_csv(filepath : str, mode : str ='rf'):
    df = pd.read_csv(filepath)
    
    if mode == 'rf':
        # Converter explicitamente para numpy para garantir tipo correto
        labels = df['target'].values
        features = df.drop('target', axis=1).values
        return features, labels
    
    elif mode == 'lstm':
        feature_cols = [col for col in df.columns if col not in ['target', 'frame_idx', 'sample_idx']]
        unique_samples = df['sample_idx'].unique()
        num_samples = len(unique_samples)
        num_frames = df['frame_idx'].nunique()
        num_features = len(feature_cols)

        features = np.zeros((num_samples, num_frames, num_features))
        labels = []
        
        # Processo de redimensionização vetorizado para Array 3D (O(N))
        for i, (sample_id, df_sample) in enumerate(df.groupby('sample_idx', sort=False)):
            df_sample = df_sample.sort_values(by='frame_idx')

            # Captura das features por frame da amostra
            landmarks_matrix = df_sample[feature_cols].values
            t_real = landmarks_matrix.shape[0]
            features[i, : t_real, :] = landmarks_matrix

            # Coleta da label da amostra
            sample_label = df_sample['target'].iloc[0]
            labels.append(sample_label)

        labels = np.array(labels)
        return features, labels
    
    else:
        print('Please enter a valid mode')
        return None, None
    
if __name__ == "__main__":
    features_lstm, labels_lstm = import_from_csv(filepath='dataset/dataset_agarrar_lstm.csv', mode='lstm')
    print(f'Shape das Features de LSTM: {features_lstm.shape}')
    print(f'Shape das Labels de LSTM: {labels_lstm.shape}')

    features_rf, labels_rf = import_from_csv(filepath='dataset/dataset_agarrar_rf.csv', mode='rf')
    print(f'Shape das Features de RF: {features_rf.shape}')
    print(f'Shape das Labels de RF: {labels_rf.shape}')