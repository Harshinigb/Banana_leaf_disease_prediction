import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("../dataset/banana.csv")

# Create encoders
le_spots = LabelEncoder()
le_texture = LabelEncoder()
le_soil = LabelEncoder()
le_label = LabelEncoder()
le_color = LabelEncoder()

# Encode categorical columns
data['SpotsPresent'] = le_spots.fit_transform(data['SpotsPresent'])
data['Texture'] = le_texture.fit_transform(data['Texture'])
data['SoilType'] = le_soil.fit_transform(data['SoilType'])
data['ColorIntensity'] = le_color.fit_transform(data['ColorIntensity'])
data['DiseaseLabel'] = le_label.fit_transform(data['DiseaseLabel'])

# Split features and target
X = data.drop("DiseaseLabel", axis=1)
y = data["DiseaseLabel"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize models
rf = RandomForestClassifier()
dt = DecisionTreeClassifier()
lr = LogisticRegression(max_iter=200)
knn = KNeighborsClassifier()

# Train models
rf.fit(X_train, y_train)
dt.fit(X_train, y_train)
lr.fit(X_train, y_train)
knn.fit(X_train, y_train)

# Calculate accuracy
rf_acc = accuracy_score(y_test, rf.predict(X_test))
dt_acc = accuracy_score(y_test, dt.predict(X_test))
lr_acc = accuracy_score(y_test, lr.predict(X_test))
knn_acc = accuracy_score(y_test, knn.predict(X_test))

print("Random Forest:", rf_acc)
print("Decision Tree:", dt_acc)
print("Logistic Regression:", lr_acc)
print("KNN:", knn_acc)

# Select best model
best_model = rf
best_accuracy = rf_acc
best_name = "Random Forest"

if dt_acc > best_accuracy:
    best_model = dt
    best_accuracy = dt_acc
    best_name = "Decision Tree"

if lr_acc > best_accuracy:
    best_model = lr
    best_accuracy = lr_acc
    best_name = "Logistic Regression"

if knn_acc > best_accuracy:
    best_model = knn
    best_accuracy = knn_acc
    best_name = "KNN"

print("Best Model:", best_name)

# Save best model
pickle.dump(best_model, open("../saved_model/model.pkl", "wb"))

# Save accuracy data
accuracy_data = {
    "Random Forest": rf_acc,
    "Decision Tree": dt_acc,
    "Logistic Regression": lr_acc,
    "KNN": knn_acc,
    "Best Model": best_name
}

pickle.dump(accuracy_data, open("../saved_model/accuracy.pkl", "wb"))

# Save encoders
pickle.dump(le_spots, open("../saved_model/le_spots.pkl", "wb"))
pickle.dump(le_texture, open("../saved_model/le_texture.pkl", "wb"))
pickle.dump(le_soil, open("../saved_model/le_soil.pkl", "wb"))
pickle.dump(le_label, open("../saved_model/le_label.pkl", "wb"))
pickle.dump(le_color, open("../saved_model/le_color.pkl", "wb"))