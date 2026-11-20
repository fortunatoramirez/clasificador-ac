# probar_segmentacion_calidad_pediatrico.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

DATASET_PATH = "dataset_circor.xlsx"
PACIENTES_PRUEBA_PATH = "pacientes_prueba_final.csv"

def main():
    df = pd.read_excel(DATASET_PATH)
    df['paciente_id'] = df['paciente_id'].astype(str)
    feature_cols = [c for c in df.columns if c not in ('Etiqueta', 'archivo', 'paciente_id')]

    pacientes_prueba = pd.read_csv(PACIENTES_PRUEBA_PATH)['paciente_id'].astype(str)
    es_prueba = df['paciente_id'].isin(pacientes_prueba)
    df_dev = df[~es_prueba]
    df_prueba = df[es_prueba].copy()

    print(f"dev: {len(df_dev)} filas, {df_dev['paciente_id'].nunique()} pacientes")
    print(f"prueba: {len(df_prueba)} filas, {df_prueba['paciente_id'].nunique()} pacientes")

    # reentrenamos SOLO con dev, para que la prueba sobre los 117 siga siendo honesta
    modelo = Pipeline([('escalador', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=0))])
    modelo.fit(df_dev[feature_cols].values, df_dev['Etiqueta'].values)

    proba = modelo.predict_proba(df_prueba[feature_cols].values)[:, 1]
    df_prueba['prob_soplo'] = proba

    # densidad de ciclos por archivo = proxy de que tan bien segmento esa grabacion
    ciclos_por_archivo = df_prueba.groupby('archivo').size().rename('n_ciclos')
    resumen = df_prueba.groupby('archivo').agg(real=('Etiqueta', 'first'), prob_promedio=('prob_soplo', 'mean')).join(ciclos_por_archivo)

    mediana_ciclos = resumen['n_ciclos'].median()
    resumen['grupo'] = np.where(resumen['n_ciclos'] >= mediana_ciclos,
                                  f'bien segmentado (>= {mediana_ciclos:.0f} ciclos)',
                                  f'mal segmentado (< {mediana_ciclos:.0f} ciclos)')

    sanos = resumen[resumen['real'] == 0]
    print(f"\nsujetos REALMENTE Sanos con dato de segmentacion: {len(sanos)}")
    print(sanos.groupby('grupo')['prob_promedio'].describe())

if __name__ == "__main__":
    main()