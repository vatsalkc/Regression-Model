# 🧠 Regression Model Collection

This repository contains multiple **machine learning regression projects**, each focusing on predicting real-world outcomes using Python and Scikit-learn.  
Each model demonstrates the use of regression algorithms for prediction, evaluation, and data analysis.

---

## 📂 Project Overview

### 🎓 1. **Marks Prediction Model**
**File:** `marks_predict_model.ipynb`  
**Dataset:** `student_performance.csv`

#### 🧮 Features:
| Feature | Description |
|----------|--------------|
| `Hours_Studied` | Number of study hours per day |
| `Lectures_Attended` | Number of lectures attended |
| `Sleep_Hours` | Average sleep hours per day |

#### 🎯 Target:
- `Exam_Score` → Predicted student performance score based on habits and effort.

#### 📈 Objective:
To determine how studying habits and sleep affect students’ exam performance using **Linear Regression**.

---

### 🧬 2. **Diabetes Prediction Model**
**File:** `predict_diabetes.ipynb`  
**Dataset:** `diabetes_data.csv`

#### 🧮 Features:
| Feature | Description |
|----------|--------------|
| `Age` | Patient age in years |
| `BMI` | Body Mass Index (health indicator) |
| `Blood_Sugar` | Blood sugar level (mg/dL) |

#### 🎯 Target:
- `Diabetes` → Binary variable (0 = No diabetes, 1 = Diabetes risk).

#### 📈 Objective:
To predict whether a person is at risk of diabetes based on health indicators using **Regression & Classification techniques**.

---

### 🏙️ 3. **City Price Prediction Model**
**File:** `city_price_predict.ipynb`  
**Dataset:** `city_data.csv`

#### 🧮 Features:
| Feature | Description |
|----------|--------------|
| `City` | City name (categorical, encoded) |
| `Tier` | Tier classification of the city (1/2/3) |
| `Year` | Data year (2015, 2020, 2025) |
| `Cost_of_Living_Index` | Index representing cost of living |
| `Average_Annual_Salary_INR_Lakhs` | Average salary in lakhs |
| `Dominating_Sector` | Main economic sector of the city |

#### 🎯 Target:
- `Avg_House_Price_per_Sq_Ft_INR` → Average house price per square foot.

#### 📈 Objective:
To predict housing prices using cost of living, salary, and economic data via **Multiple Linear Regression**.

---

## 🧰 Tech Stack

- **Python 3.8+**
- **Pandas** – Data manipulation  
- **NumPy** – Numerical operations  
- **Matplotlib / Seaborn** – Visualization  
- **Scikit-learn** – Regression models & evaluation  

---

## 🧪 Model Evaluation

Each notebook includes:
- **Train/Test Split**
- **Linear Regression Model**
- **Performance Metrics:**
  - R² Score → Model accuracy  
  - MAE (Mean Absolute Error) → Average prediction error  
  - RMSE (Root Mean Squared Error) → Penalizes larger errors  

---

## 📊 Example Workflow

```python
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
df = pd.read_csv("city_data.csv")

# Select features & target
X = df[["Tier", "Year", "Cost_of_Living_Index", "Average_Annual_Salary_INR_Lakhs"]]
y = df["Avg_House_Price_per_Sq_Ft_INR"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
