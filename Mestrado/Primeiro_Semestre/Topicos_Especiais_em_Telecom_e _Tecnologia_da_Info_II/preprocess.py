"""
Preprocessing script for the InRete Lab Hard Failure optical dataset,
replicating the feature matrix described in Silva et al. 2022 (TNSM).

Features (13): 
  Ampli1 InputPower, Ampli1 OutputPower,
  Ampli2 InputPower, Ampli2 OutputPower,
  Ampli3 InputPower, Ampli3 OutputPower,
  Ampli4 InputPower, Ampli4 OutputPower,
  Card1 OSNR (SPO1), Card1 BER (SPO1),
  Link amp1->amp2 (Ampli1 Out - Ampli2 In),
  Link amp2->amp3 (Ampli2 Out - Ampli3 In),
  Link amp3->amp4 (Ampli3 Out - Ampli4 In)

Target for detection: Failure flag (1 during simulated hard failures, else 0).
"""
import pandas as pd
import numpy as np

RAW = "/Trabalho_de_fim_de_curso/project_fault_management/data/HardFailure_dataset.csv"
OUT = "/Trabalho_de_fim_de_curso/project_fault_management/optical_features.csv"

def load_and_pivot():
    df = pd.read_csv(RAW)
    df['ts'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.sort_values('ts')

    def series_for(id_, cols, prefix):
        sub = df[df['ID'] == id_][['ts'] + cols].dropna(how='all', subset=cols).copy()
        sub = sub.sort_values('ts')
        sub.columns = ['ts'] + [f"{prefix}_{c}" for c in cols]
        return sub

    ampli1 = series_for('Ampli1', ['InputPower', 'OutputPower'], 'Ampli1')
    ampli2 = series_for('Ampli2', ['InputPower', 'OutputPower'], 'Ampli2')
    ampli3 = series_for('Ampli3', ['InputPower', 'OutputPower'], 'Ampli3')
    ampli4 = series_for('Ampli4', ['InputPower', 'OutputPower'], 'Ampli4')
    spo1 = series_for('SPO1/18/11', ['BER', 'OSNR'], 'Card1')
    failure = df[df['ID'] == 'SPO1/18/11'][['ts', 'Failure']].copy()  # failure flag alignment
    # Failure flag actually applies globally; use max over any row at that ts window
    fail_all = df.groupby('ts')['Failure'].max().reset_index()

    # base timeline = Ampli1 timestamps (most complete series)
    base = ampli1[['ts']].copy()

    merged = base.copy()
    for piece in [ampli1, ampli2, ampli3, ampli4, spo1]:
        merged = pd.merge_asof(merged.sort_values('ts'), piece.sort_values('ts'),
                                on='ts', direction='nearest', tolerance=pd.Timedelta('5s'))
    merged = pd.merge_asof(merged.sort_values('ts'), fail_all.sort_values('ts'),
                            on='ts', direction='nearest', tolerance=pd.Timedelta('5s'))

    merged['Failure'] = merged['Failure'].fillna(0).astype(int)

    # link difference features
    merged['Link_amp1_amp2'] = merged['Ampli1_OutputPower'] - merged['Ampli2_InputPower']
    merged['Link_amp2_amp3'] = merged['Ampli2_OutputPower'] - merged['Ampli3_InputPower']
    merged['Link_amp3_amp4'] = merged['Ampli3_OutputPower'] - merged['Ampli4_InputPower']

    feature_cols = [
        'Ampli1_InputPower', 'Ampli1_OutputPower',
        'Ampli2_InputPower', 'Ampli2_OutputPower',
        'Ampli3_InputPower', 'Ampli3_OutputPower',
        'Ampli4_InputPower', 'Ampli4_OutputPower',
        'Card1_OSNR', 'Card1_BER',
        'Link_amp1_amp2', 'Link_amp2_amp3', 'Link_amp3_amp4',
    ]

    merged = merged.dropna(subset=feature_cols).reset_index(drop=True)
    final = merged[['ts'] + feature_cols + ['Failure']]
    return final


if __name__ == "__main__":
    import os
    os.makedirs("/home/claude/data", exist_ok=True)
    final = load_and_pivot()
    final.to_csv(OUT, index=False)
    print("Shape:", final.shape)
    print("Failure samples:", final['Failure'].sum())
    print("Total samples:", len(final))
    print(final.describe())
