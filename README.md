# 🧠 Diabetes Prediction using Machine Learning (Google Colab + Sklearn)

This project demonstrates a full data science pipeline using Python in Google Colab to predict diabetes using various machine learning models. The dataset is preprocessed, visualized, and evaluated through multiple classifiers such as Logistic Regression, Decision Tree, Random Forest, and SVM.

---

## 📁 1. Mount Google Drive

We mount Google Drive to access the dataset stored in it.

```python
from google.colab import drive
drive.mount('/content/drive')

```

## 📦 2. Load Dataset
Load the dataset (in this case, a .zip or .csv) from Google Drive into a Pandas DataFrame.
```python
import pandas as pd

file_path = '/content/drive/MyDrive/archive (1).zip'
df = pd.read_csv(file_path)
print(df.head())
```

## 🔧 3. Handle Missing Values
Fill missing numeric columns with mean and categorical columns with mode.
from sklearn.preprocessing import LabelEncoder
```python
label_encoder = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = label_encoder.fit_transform(df[col])
```

## 🏷️ 4. Encode Categorical Features
Convert non-numeric (categorical) features into numeric using Label Encoding.

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = label_encoder.fit_transform(df[col])

```

## 🔢 5. Normalize Numerical Features
Normalize numerical values to the 0–1 range using MinMaxScaler.

```python

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

```

## 📊 6. Data Exploration and Visualization
Explore data distributions and relationships between features.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Histograms
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Box plots
df.plot(kind='box', subplots=True, layout=(3, 4), figsize=(12, 8), sharex=False, sharey=False)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

```

## 🎯 7. Define Features and Target Variable
Split the dataset into independent features (X) and target labels (y).
```python
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
```

## ⚖️ 8. Show Class Distribution
Check balance between diabetic and non-diabetic samples.
```python

print(y.value_counts())

sns.countplot(x=y)
plt.title("Target Class Distribution")
plt.xlabel("Diabetes Status")
plt.ylabel("Count")
plt.show()

```
## 🤖 9. Train and Evaluate ML Models
Use multiple classifiers to find the best-performing model.

```python

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(kernel='linear')
}

# Train, Predict & Evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"\n{name} Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Non-Diabetic", "Diabetic"],
                yticklabels=["Non-Diabetic", "Diabetic"])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

```
## 📈 10. Compare Model Accuracies
Visual comparison of model performances.

```python
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette='viridis')
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## ✅ Conclusion
This notebook covers a full pipeline:

✅ Load and preprocess data

✅ Explore distributions & feature correlations

✅ Encode & normalize

✅ Train four models

✅ Visualize results and compare accuracies

## 📚 Requirements
Python 3.x

Google Colab

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
