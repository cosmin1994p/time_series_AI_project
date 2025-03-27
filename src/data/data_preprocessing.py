import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Day'], format="%d/%m/%Y") + pd.to_timedelta(df['Hour'] - 1, unit='h')
    df = df.sort_values('Timestamp').reset_index(drop=True)

    scaler = MinMaxScaler()
    df['MWh_scaled'] = scaler.fit_transform(df[['MWh']])

    return df, scaler
