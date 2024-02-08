# -*- coding: utf-8 -*-

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
# Assume the dataset has features (X) and target labels (y)
df = pd.read_csv('TrafficFlowRandomForest.csv')

"""Preprocessing steps"""

# Not significant nor related to data set
df.drop('Date',inplace=True, axis=1)
df.drop('Sys_Time',inplace=True,axis=1)

day_mapping = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6}
df['Day']= df['Day'].map(day_mapping)

# Combine unique values from both columns
unique_places = pd.concat([df['Destination_Location'], df['Starting_Location']]).unique()

# Create a dictionary to map places to their indices
places_mapping = {place: index for index, place in enumerate(unique_places)}

# Map places to indices in the dataframe
df['Destination_Location'] = df['Destination_Location'].map(places_mapping)
df['Starting_Location'] = df['Starting_Location'].map(places_mapping)

# Combine unique values from both columns
unique_routes = df['Fastest_Route_Name'].unique()

# Create a dictionary to map places to their indices
routes_mapping = {place: index for index, place in enumerate(unique_routes)}

# Map places to indices in the dataframe
df['Fastest_Route_Name'] = df['Fastest_Route_Name'].map(routes_mapping)

# Combine unique values from both columns
unique_weather = df['Weather'].unique()

# Create a dictionary to map places to their indices
weather_mapping = {place: index for index, place in enumerate(unique_weather)}

# Map places to indices in the dataframe
df['Weather'] = df['Weather'].map(weather_mapping)

def binary_encoding(value):
    return 1 if value.lower() == 'yes' else 0

# convert yes and no to binary for holiday and special_condition

df['Holiday'] = df['Holiday'].apply(lambda x: binary_encoding(x))
df['Special_Condition'] = df['Special_Condition'].apply(lambda x: binary_encoding(x))

# Combine unique values from both columns
unique_prediction = df['Data_prediction'].unique()

# Create a dictionary to map places to their indices
prediction_mapping = {place: index for index, place in enumerate(unique_prediction)}

# Map places to indices in the dataframe
df['Data_prediction'] = df['Data_prediction'].map(prediction_mapping)

# Split the data into features (X) and target labels (y)
X = df.drop('Data_prediction', axis=1)  # Replace 'target_column' with the actual target column name
y = df['Data_prediction']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=256)

# Create and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=256)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')


import pickle

# Save the model using pickle
with open('rfcmodel.pkl', 'wb') as file:
    pickle.dump(model, file)

# save the mapping for processing inputs
import json

# Create a list of dictionaries
mappings = [day_mapping, places_mapping,routes_mapping,weather_mapping,prediction_mapping]

# Loop through the list and save each dictionary as a JSON file
for mapping in mappings:
    # Get the variable name dynamically
    var_name = [name for name, obj in globals().items() if obj is mapping][0]

    # Convert the dictionary to a JSON-formatted string
    json_str = json.dumps(mapping, indent=2)

    # Save the JSON string to a file with the variable name
    with open(f'{var_name}.json', 'w') as file:
        file.write(json_str)

# Display the list of file names
file_names = [f'{var_name}.json' for var_name in [name for name, obj in globals().items() if isinstance(obj, dict)]]
print(f'Saved files: {file_names}')