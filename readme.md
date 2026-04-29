# 🍌 Banana Leaf Disease Prediction

A Django-based web application that predicts whether a banana leaf is **Healthy or Diseased** based on leaf features using Machine Learning.

---

## 🖥️ Screenshots

### 🏠 Home Page – Input Form
![Home Page](assets/screenshots/Banana%20leaf%20desease-1.png)

### ✅ Prediction Result Page
![Prediction Result](assets/screenshots/Banana%20leaf%20desease-2.png)

---

## 🛠️ Tech Stack

- **Backend:** Django (Python)
- **ML Model:** Scikit-learn (`.pkl` models)
- **Frontend:** HTML, CSS (Django Templates)
- **Dataset:** banana.csv

---

## 📋 Input Features

The model takes the following inputs to predict the disease:

| Feature | Type |
|---|---|
| Leaf Length | Number |
| Leaf Width | Number |
| Color Intensity | Dropdown (e.g., Light) |
| Spots Present | Dropdown (Yes / No) |
| Moisture Level | Number |
| Texture | Dropdown (e.g., Smooth) |
| Humidity | Number |
| Temperature | Number |
| Soil Type | Dropdown (e.g., Clay) |

---

## 🤖 ML Model Performance

| Algorithm | Accuracy |
|---|---|
| Random Forest | 54% |
| Decision Tree | 47% |
| Logistic Regression | 54% |
| KNN | 54% |

> ✅ **Best Model: Random Forest**

---

## 📁 Project Structure

```
Banana_leaf_disease_prediction/
├── assets/
│   └── screenshots/
│       ├── Banana leaf desease-1.png
│       └── Banana leaf desease-2.png
├── dataset/
│   └── banana.csv
├── django_app/
│   ├── config/
│   └── predictor/
│       ├── __pycache__/
│       ├── migrations/
│       ├── templates/
│       ├── __init__.py
│       ├── admin.py
│       ├── apps.py
│       ├── models.py
│       ├── tests.py
│       ├── urls.py
│       └── views.py
├── ml_model/
├── saved_model/
│   ├── accuracy.pkl
│   ├── le_color.pkl
│   ├── le_label.pkl
│   ├── le_soil.pkl
│   ├── le_spots.pkl
│   ├── le_texture.pkl
│   └── model.pkl
├── static/
│   ├── healthy.jpg
│   └── unhealthy.jpg
├── .gitignore
├── manage.py
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Harshinigb/Banana_leaf_disease_prediction.git
cd Banana_leaf_disease_prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Django Server
```bash
python manage.py runserver
```

### 4. Open in Browser
```
http://127.0.0.1:8000/
```

---

## 📊 Dataset

- `banana.csv` contains the training data used for banana leaf disease prediction

---

## 👩‍💻 Author

**Harshini GB**  
GitHub: [@Harshinigb](https://github.com/Harshinigb)