
---

## 📊 Datasets

### 1. `student_performance.csv`
- **Columns**: `Hours_Studied`, `Lectures_Attended`, `Sleep_Hours`, `Exam_Score`
- **Task**: Predict exam score (continuous) using **Linear Regression**.

### 2. `diabetes_data.csv`
- **Columns**: `Age`, `BMI`, `Blood_Sugar`, `Diabetes`
- **Task**: Predict diabetes (0 = No, 1 = Yes) using **Logistic Regression**.

---

## 🧑‍💻 Notebooks

### 1. `marks_predict_model.ipynb`
- Loads **student dataset**
- Splits into training & testing sets
- Trains a **Linear Regression model**
- Evaluates with metrics: **MAE, MSE, RMSE, R² Score**

### 2. `predict_diabetes.ipynb`
- Loads **diabetes dataset**
- Splits into training & testing sets
- Trains a **Logistic Regression model**
- Evaluates with metrics: **Accuracy, Precision, Recall, F1-Score**

---

## 📌 Difference Between Linear and Logistic Regression

Linear Regression and Logistic Regression are two popular machine learning algorithms, but they are used for different types of problems.  

- **Linear Regression** is used when we want to predict a **number** (a continuous value). For example, in this project we used it to predict a student's **exam score** based on study hours, lectures attended, and sleep hours. The output can be any real number, like 72.5 or 85.3.  

- **Logistic Regression** is used when we want to predict a **category** (a classification). Instead of giving a number directly, it gives a **probability** between 0 and 1, which is then converted into a class (like 0 or 1). In this project we used it to predict whether a person has **diabetes (Yes = 1, No = 0)** based on age, BMI, and blood sugar level.  

👉 In short:  
- Linear Regression = predicting "how much" (e.g., marks).  
- Logistic Regression = predicting "which class" (e.g., diabetes yes/no).  

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/vatsalkc/Regression-Model.git
   cd Regression-Model
