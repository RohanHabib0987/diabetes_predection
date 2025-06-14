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

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = label_encoder.fit_transform(df[col])
```
