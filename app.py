from flask import Flask, render_template, request
import pickle
import json


app = Flask(__name__)

# Load the model using pickle
with open('rfcmodel.pkl', 'rb') as file:
    model = pickle.load(file)
    
with open('day_mapping.json', 'r') as file:
    day_mapping = json.load(file)
    days = list(day_mapping.keys())
    
with open('places_mapping.json', 'r') as file:
    places_mapping = json.load(file)
    places = list(places_mapping.keys())
    
with open('routes_mapping.json', 'r') as file:
    routes_mapping = json.load(file)
    routes = list(routes_mapping.keys())
    
    
with open('weather_mapping.json', 'r') as file:
    weather_mapping = json.load(file)
    weathers = list(weather_mapping.keys())
    
with open('prediction_mapping.json', 'r') as file:
    prediction_mapping = json.load(file)
    prediction = list(prediction_mapping.keys())

@app.route('/')
def home():
    return render_template('index.html',days=days,places=places,routes=routes,weathers=weathers,prediction=prediction)

@app.route('/predict', methods=['POST'])

def predict():
    # Get input values from the form
    day = day_mapping[request.form['day']]
    dest = places_mapping[request.form['destination_location']]
    route_dist = float(request.form['fastest_route_distance'])
    route = routes_mapping[request.form['fastest_route_name']]
    route_time = float(request.form['fastest_route_time'])
    holiday = float(request.form['holiday'])
    special = float(request.form['special_condition'])
    start = places_mapping[request.form['starting_location']]
    weather = weather_mapping[request.form['weather']]
    # Add more features as needed

    # Create a feature vector based on the input values
    input_data = [[day,dest,route_dist,route,route_time,holiday,special,start,weather]]  # Update with your feature names

    # Make the prediction using the loaded model
    traffic = model.predict(input_data)[0]
    result = prediction[traffic]

    return render_template('index.html', result=result, days=days,places=places,routes=routes,weathers=weathers,prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
