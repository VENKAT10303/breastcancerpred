# 🩺 Breast Cancer Prediction using Machine Learning

This project is a **Breast Cancer Prediction System** developed using **Python and Machine Learning**.  
It predicts whether a breast tumor is **Benign** or **Malignant** based on medical diagnostic features using a **Random Forest Classifier**.

---

## 📌 Project Overview

Early detection of breast cancer is crucial for effective treatment and improved survival rates.  
This project uses the **Breast Cancer Wisconsin Dataset** and applies machine learning techniques to classify tumors.

The model processes diagnostic measurements extracted from breast mass images and predicts the tumor diagnosis.

---

## ⚙️ Technologies Used

- Python 🐍  
- NumPy  
- Pandas  
- Scikit-learn  
- Machine Learning (Random Forest Classifier)

---

## 📂 Dataset Details

- Dataset: Breast Cancer Wisconsin Dataset  
- Target column:
  - `diagnosis`
    - `M` → Malignant
    - `B` → Benign
- The `id` column is removed during preprocessing.

---

## 🔄 Project Workflow

1. Load the dataset  
2. Remove unnecessary columns  
3. Encode diagnosis labels  
4. Split data into training and testing sets  
5. Apply feature scaling using `StandardScaler`  
6. Train a **Random Forest Classifier**  
7. Predict tumor diagnosis for given input values  

---

## 🧠 Machine Learning Model

- **Random Forest Classifier**
  - Number of estimators: 100
  - Random state: 42
- Chosen for its robustness and good performance on medical datasets.

---

## ▶️ How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
