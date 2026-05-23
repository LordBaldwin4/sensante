import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

print('Chargement des donnees...')
df = pd.read_csv('/app/data/patients_dakar.csv')

print('Colonnes:', list(df.columns))

le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_enc'] = le_sexe.fit_transform(df['sexe'])
df['region_enc'] = le_region.fit_transform(df['region'])

feature_cols = ['age', 'sexe_enc', 'temperature', 'tension_sys', 'toux', 'fatigue', 'maux_tete', 'frissons', 'nausee', 'region_enc']

X = df[feature_cols]
y = df['diagnostic']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, '/app/model.pkl')
joblib.dump(le_sexe, '/app/encoder_sexe.pkl')
joblib.dump(le_region, '/app/encoder_region.pkl')
joblib.dump(feature_cols, '/app/feature_cols.pkl')

print('Done! Classes:', list(model.classes_))
