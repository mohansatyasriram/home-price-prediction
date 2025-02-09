import pickle
import numpy as np
from flask import Flask, request, render_template
import json

# Create a Flask app
app = Flask(__name__)

# Load the pickle model and the columns data
model = pickle.load(open("benglore_home_price.pickle", "rb"))
with open("columns.json", "r") as f:
    __data_columns = json.load(f)["data_columns"]
__locations = __data_columns[3:]

# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Take the input values from the form and convert them to the right type
        total_sqft = float(request.form['total_sqft'])
        location = request.form['location']
        bhk = int(request.form['bhk'])
        bath = int(request.form['bath'])

        # Handle location and features preparation
        try:
            loc_index = __data_columns.index(location.lower())
        except:
            loc_index = -1
        
        x = np.zeros(len(__data_columns))
        x[0] = total_sqft
        x[1] = bath
        x[2] = bhk
        if loc_index >= 0:
            x[loc_index] = 1

        # Make prediction using the loaded model
        prediction = model.predict([x])[0]

        # Render the prediction result on the webpage
        return render_template("index.html", prediction_text="The predicted price is: ₹{}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
