#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import logging
from joblib import Parallel, delayed

pd.set_option('display.max_colwidth', None)  # Не переносить довгі колонки
pd.set_option('display.expand_frame_repr', False)  # Відключає авто-перенесення

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Завантаження даних
logging.info("Завантаження даних...")
df = pd.read_csv("~/Downloads/construction_data2.csv")

# Видалення рядків з порожніми значеннями
logging.info("Видалення рядків з порожніми значеннями...")
df.dropna(inplace=True)

# Вибір 10% даних для подальшої обробки
logging.info("Вибір 10% даних для обробки...")
df = df.sample(frac=0.1, random_state=42)

# Конвертація часу
logging.info("Конвертація часу...")
df["Start_Time"] = pd.to_datetime(df["Start_Time"], format='mixed', errors='coerce')
df["End_Time"] = pd.to_datetime(df["End_Time"], format='mixed', errors='coerce')
df["Duration"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 3600  # у годинах
# Закодуємо категоріальні змінні
logging.info("Закодування категоріальних змінних...")
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
#%%
# Завдання 1: Прогнозування рівня впливу (Severity)
logging.info("Початок тренування моделей для прогнозування рівня впливу...")
X = df.drop(columns=["Severity", "ID", "Start_Time", "End_Time"])
y = df["Severity"]
y = y - 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабування даних для кращої конвергенції
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),  # Increased max_iter
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),  # Use all cores
    "XGBoost": XGBClassifier(eval_metric="mlogloss", n_jobs=-1)
}

def train_and_evaluate(name, model):
    logging.info(f"Тренування моделі: {name}...")
    model.fit(X_train_scaled, y_train)  # Fit with scaled data
    y_pred = model.predict(X_test_scaled)  # Predict with scaled data
    print(y_pred)
    result = {
        'model': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=1),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=1),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=1),
        "y_pred": y_pred
    }
    return result

# Parallelize model training
results = Parallel(n_jobs=-1)(delayed(train_and_evaluate)(name, model) for name, model in models.items())
# Print results
for result in results:
    print(f"  Model: {result['model']}")
    print(f"  Accuracy: {result['accuracy']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall: {result['recall']:.4f}")
    print(f"  F1-score: {result['f1']:.4f}")
    print(result['y_pred'])
    print()
#%%
# Завдання 2: Кластеризація дорожніх робіт з іншими ознаками за допомогою K-Means
logging.info("Початок кластеризації дорожніх робіт за допомогою K-Means з ознаками 'Distance(mi)', 'Temperature(F)', 'Wind_Speed(mph)'...")

# Вибір ознак для кластеризації без Severity
X_clustering = df[["Distance(mi)", "Temperature(F)", "Wind_Speed(mph)"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clustering)

# K-Means кластеризація
kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)

# Візуалізація кластерів
logging.info("Візуалізація кластерів K-Means...")
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df["Cluster_KMeans"], palette="viridis", alpha=0.5)
plt.title("K-Means Clustering with Distance, Temperature, and Wind Speed")
plt.xlabel("Feature 1 (Distance)")
plt.ylabel("Feature 2 (Temperature)")
plt.legend(title="Cluster")
plt.show()
#%%
# Завдання 3: Прогнозування тривалості будівництва
logging.info("Початок тренування моделей для прогнозування тривалості будівництва...")
X = df.drop(columns=["Duration", "ID", "Start_Time", "End_Time"])
y = df["Duration"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, n_jobs=-1),  # Use all cores
    "XGBoost Regressor": XGBRegressor(n_jobs=-1)
}

# Define the train and evaluate function for regression
def train_and_evaluate_regressor(name, model):
    logging.info(f"Тренування моделі: {name}...")
    model.fit(X_train, y_train)  # Fit the model on training data
    y_pred = model.predict(X_test)  # Predict on test data
    result = {
        'model': name,
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        "y_test": y_test,
        "y_pred": y_pred
    }
    return result

# Parallelize model training
results = Parallel(n_jobs=-1)(delayed(train_and_evaluate_regressor)(name, model) for name, model in models.items())

# Print the results
for result in results:
    print(f"\n{result['model']}:")
    print(f"  MAE: {result['MAE']:.4f}")
    print(f"  RMSE: {result['RMSE']:.4f}")
    print(result["y_test"][:5])
    print(result['y_pred'][:5])


#%%
logging.info("Початок виявлення асоціативних правил...")


# Вибираємо більше колонок для аналізу
features = ["Severity", "Distance(mi)", "Temperature(F)", "Humidity(%)",
            "Precipitation(in)", "Wind_Speed(mph)", "Visibility(mi)"]
df_bin = df[features].copy()

# Бінаризація числових значень (розбиття на 5 категорій)
for col in df_bin.columns:
    df_bin[col] = pd.qcut(df_bin[col], q=5, labels=False, duplicates='drop')

# Перетворення у булевий формат
df_bin = df_bin.astype("bool")

# Використання алгоритму Apriori
frequent_itemsets = apriori(df_bin, min_support=0.02, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)

# Виведення знайдених закономірностей
print("\n🔹 Виявлені асоціативні правила:")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])