import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("breastcancerdataset.csv")
if 'id' in df.columns:
    df = df.drop(columns=['id'])
df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})
X = df.drop(columns='diagnosis')
Y = df['diagnosis']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, Y_train)
input_data = (14.53,13.98,93.86,644.2,0.1099,0.09242,0.06895,0.06495,0.165,0.06121,0.306,0.7213,2.143,25.7,0.006133,0.01251,0.01615,0.01136,0.02207,0.003563,15.8,16.93,103.1,749.9,0.1347,0.1478,0.1373,0.1069,0.2606,0.0781)
input_array = np.asarray(input_data).reshape(1, -1)
input_df = pd.DataFrame(input_array, columns=X.columns)
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
print("\nPrediction:", "the tumor is Benign" if prediction[0] == 1 else "the tumor is Malignant")