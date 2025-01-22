import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

#Defining variables
df = None
model = None
scaler = None
X_train_std = None
X_test_std = None
X_train = None
X_test = None
y_train = None
y_test = None

#uploading the csv file
@app.route('/upload', methods=['POST'])
def upload():
    global df
    file = request.files['file']
    if file:
        df = pd.read_csv(file)
        return jsonify({"message": "File uploaded successfully!"}), 200
    return jsonify({"message": "No file uploaded!"}), 400

#training - decision tree
@app.route('/train', methods=['POST'])
def train():
    global df, model, scaler, X_train_std, X_test_std, X_train, X_test, y_train, y_test
    
    if df is None:
        return jsonify({"message": "No data uploaded!"}), 400
    
    # Preprocessing Dataset
    encoder = LabelEncoder()
    df["Type"] = encoder.fit_transform(df["Type"])

    #sampling the data, as there are more datapoints for non failure compared to failure, for better accuracy
    non_failure = df[df['Machine failure'] == 0]
    failure = df[df["Machine failure"] == 1]    
    non_failure_sampled = non_failure.sample(n=339)
    sampled_df = pd.concat([non_failure_sampled, failure], axis=0)
    
    #dropping unneccessary columns
    X = sampled_df.drop(["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"], axis=1)
    y = sampled_df["Machine failure"]

    #out of the remaining cols, we need to decided which cols have strong relationship with the machine failure, hence we will use Mutual Class Information
    mutual_info = mutual_info_classif(X, y)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = X.columns
    print(mutual_info.sort_values(ascending=False))

    #then we will drops those columns which are not strongly realted to machine failure, in this case or data set, Rotational Speed and Torque have the most effect on the machine failure
    X = X.drop(["Air temperature [K]", "Type", "Process temperature [K]", "Tool wear [min]"], axis=1)

    #splitting into training and testing, training 70% testing 30%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #standardizing the data
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    
    #training our decision tree model, with max depth as 3 and class_weight as balanced to ensure less overfitting of the model
    model = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')
    model.fit(X_train_std, y_train)

    #evaluating the model
    y_pred = model.predict(X_test_std)
    accuracy = accuracy_score(y_test, y_pred)
    #report = classification_report(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    
      
    return jsonify({
        "accuracy": accuracy,
        "classification_report": report_dict,
        "confusion_matrix": conf_matrix,
        #"tree_plot": tree_plot
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler
    if model is None:
        return jsonify({"message": "Model not trained yet!"}), 400

    input_data = request.get_json()

    #users data
    input_df = pd.DataFrame([input_data])

    #standardizing the input data
    input_scaled = scaler.transform(input_df)

    #prediction based on the input and get the confidence
    prediction = model.predict(input_scaled)
    probalities = model.predict_proba(input_scaled)
    
    confidence = max(probalities[0]) * 100

    result = 'Yes' if prediction[0] == 1 else 'No'
    return jsonify({
        "Confidence": f"{confidence:.2f}%",  
        "Machine Failure": result,
                               
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
