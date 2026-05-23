<div style="background: linear-gradient(135deg, #1a1a2e, #0f3460); border-radius: 12px; padding: 2rem; color: white; font-family: monospace;">

<h1 style="color:#fff; margin:0 0 6px 0;">🏠 California Housing</h1>
<h3 style="color:#9FE1CB; margin:0 0 12px 0; font-weight:400;">Linear Regression — Price Prediction Pipeline</h3>
<p style="color:rgba(255,255,255,0.55); margin: 0 0 18px 0;">Machine Learning · Task 3 &nbsp;|&nbsp; Tayyabah Rehman &nbsp;|&nbsp; MPhil Artificial Intelligence &nbsp;|&nbsp; May 2026</p>

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2ea44f?style=for-the-badge)

</div>

---

## 📋 Project Summary

| | |
|---|---|
| **Dataset** | California Housing (20,640 samples, 8 numerical features) |
| **Goal** | Predict median house values using Linear Regression |
| **Input** | Clean numerical dataset from `sklearn.datasets` |
| **Output** | Trained model + evaluation metrics + scatter plot visualization |
| **Split** | 80% Train (16,512) / 20% Test (4,128) |

---

## 📊 Model Results

| Metric | Score | Meaning |
|--------|-------|---------|
| 📉 **RMSE** | `0.7456` | Average error of ~$74,560 per prediction |
| 📏 **MAE** | `0.5332` | Typical prediction error of ~$53,320 |
| 📊 **R²** | `0.5758` | Model explains 57.6% of variance in house prices |

---

## 🗺️ Pipeline Overview

```
Load Dataset  →  Train/Test Split (80/20)  →  Train LinearRegression
             →  Predict on Test Set        →  Evaluate (RMSE, MAE, R²)
             →  Visualize Actual vs Predicted
```

---

## 📦 Dataset Source

> Loaded directly from Scikit-learn — no manual download needed.

```python
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

print(df.shape)              # (20640, 9)
print(df.isnull().sum().sum())  # 0 — no missing values
```

| Feature | Description |
|---------|-------------|
| `MedInc` | Median income in block group |
| `HouseAge` | Median house age in block group |
| `AveRooms` | Average rooms per household |
| `AveBedrms` | Average bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |

---

## 🔧 Methodology

### 1️⃣ Train / Test Split
```python
from sklearn.model_selection import train_test_split

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Train: 16,512 samples | Test: 4,128 samples
```

### 2️⃣ Model Training
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
# Intercept: -37.0233
```

### 3️⃣ Evaluation Metrics
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # 0.7456
mae  = mean_absolute_error(y_test, y_pred)           # 0.5332
r2   = r2_score(y_test, y_pred)                      # 0.5758
```

### 4️⃣ Visualization
```python
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='white', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', linewidth=1.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}')
plt.savefig('predicted_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 🔍 Interpretation

- **R² = 0.5758** — A moderate score; the model captures meaningful patterns but is limited by the linearity assumption
- **MAE = 0.5332** — On average the model is off by ~$53,000 per house — solid baseline, room for improvement
- **RMSE = 0.7456** — Slightly higher than MAE, indicating some larger errors from outliers pulling the score up
- **Plot** — Points cluster reasonably around the diagonal; higher-value properties (above 4.0) tend to be underpredicted

---

## ⚠️ Limitations & Future Work

| Limitation | Potential Fix |
|------------|---------------|
| Linear assumption may not hold for Lat/Long | Add polynomial features or use tree-based models |
| No feature scaling before training | Apply `StandardScaler` to improve convergence |
| Outliers in high-value properties | Apply log transformation to target variable |
| R² of 0.58 leaves 42% unexplained | Try Random Forest, Gradient Boosting, or Ridge Regression |

---

## 📁 Project Structure

```
linear-regression-california-housing/
│
├── Task3_ML_Tayyabah_Rehman.ipynb   # Main Jupyter Notebook
├── predicted_vs_actual.png          # Output scatter plot (300 DPI)
├── Task3_Description.docx           # Written task description
└── README.md                        # This file
```

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/linear-regression-california-housing.git
cd linear-regression-california-housing
```

### 2. Install Required Libraries
```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

### `requirements.txt`
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
jupyter>=1.0.0
```

---

## ▶️ How to Run

**Option A — Jupyter Notebook**
```bash
jupyter notebook Task3_ML_Tayyabah_Rehman.ipynb
```
Run all cells: **Kernel → Restart & Run All**

**Option B — Google Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lPowg0NVOucV_45TtO85QCs6I02PZ9uE)

> Dataset loads automatically from Scikit-learn — no local files needed.

---

## 🧰 Tools & Libraries

| Tool | Version | Purpose |
|------|---------|---------|
| `pandas` | ≥1.3 | Data handling and DataFrames |
| `numpy` | ≥1.21 | Numerical operations (RMSE calc) |
| `scikit-learn` | ≥0.24 | Model, splitting, metrics |
| `matplotlib` | ≥3.4 | Scatter plot visualization |
| `jupyter` | ≥1.0 | Interactive notebook environment |

---

## 👩‍💻 Author

**Tayyabah Rehman**
MPhil Artificial Intelligence · Machine Learning Task 3 · May 2026

---

<div align="center">⭐ If you found this useful, give it a star!</div>
