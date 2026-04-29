import pickle
import numpy as np
from django.shortcuts import render

# Load model and encoders
model = pickle.load(open("../saved_model/model.pkl", "rb"))
le_spots = pickle.load(open("../saved_model/le_spots.pkl", "rb"))
le_texture = pickle.load(open("../saved_model/le_texture.pkl", "rb"))
le_soil = pickle.load(open("../saved_model/le_soil.pkl", "rb"))
le_label = pickle.load(open("../saved_model/le_label.pkl", "rb"))
le_color = pickle.load(open("../saved_model/le_color.pkl", "rb"))

# 👉 Load raw accuracy data
raw_accuracy = pickle.load(open("../saved_model/accuracy.pkl", "rb"))


def index(request):

    if request.method == "POST":
        try:
            # Numeric inputs
            length = float(request.POST.get('length'))
            width = float(request.POST.get('width'))
            moisture = float(request.POST.get('moisture'))
            humidity = float(request.POST.get('humidity'))
            temp = float(request.POST.get('temp'))

            # Categorical inputs
            color = le_color.transform([request.POST.get('color')])[0]
            texture = le_texture.transform([request.POST.get('texture')])[0]
            soil = le_soil.transform([request.POST.get('soil')])[0]

            # Spots encoding
            spots_input = request.POST.get('spots')
            spots = le_spots.transform([spots_input])[0]

            # Arrange input
            input_data = np.array([[length, width, color, spots, moisture, texture, humidity, temp, soil]])

            # Prediction
            prediction = model.predict(input_data)
            result = le_label.inverse_transform(prediction)[0]

            # 👉 FORMAT accuracy (remove decimals → %)
            accuracy_data = {}
            best_model = ""

            for key, value in raw_accuracy.items():
                if key == "Best Model":
                    best_model = value
                else:
                    accuracy_data[key] = f"{int(value * 100)}%"

            # 👉 Send to template
            return render(request, "result.html", {
                "result": result,
                "accuracy": accuracy_data,
                "best_model": best_model
            })

        except Exception as e:
            return render(request, "index.html", {"result": "Error: " + str(e)})

    return render(request, "index.html")