# Machine Failure Prediction API

This project implements a RESTful API for predicting machine failure based on input parameters such as `Rotational speed [rpm]` and `Torque [Nm]`.

## Features
- **Python with Flask**: The API is built using Flask for simplicity and efficiency.
- **scikit-learn for ML**: A Decision Tree Classifier is used for predictions.
- **Test Locally**: Endpoints can be tested using tools like Postman or `curl`.

---

## Project Structure
```Machine Failure Prediction
├── app.py                   # Flask application with API endpoints
├── requirements.txt         # Dependencies required for the project
├── dataset/                 # Folder containing the sample dataset
│   └── ai4i2020.csv         # Example dataset for training the model
```

## Dataset Information

This project requires a dataset to train the machine learning model. A sample dataset is provided in the project folder under `dataset/ai4i2020.csv`.

### Steps to Use the Dataset
1. Ensure the `dataset/` folder exists in your project directory.
2. Place your dataset CSV file (e.g., `ai4i2020.csv`) inside the `dataset/` folder.
3. Use the `/upload` API endpoint to upload the dataset for training.

## Setup Instructions
### Prerequisites
- Python 3.7 or above
- pip (Python package manager)

### Steps
 1. Clone the repository and navigate to the project folder:
   ```bash
   git clone https://github.com/nileshapsingkar/Predicting-Machine-Failure
   cd project/
```
2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```
4. Run the Flask Application:
```bash
python app.py
```
5. The API will be available at http://127.0.0.1:5000.

## API Endpoints

### 1. Upload Dataset
- **Endpoint**: `/upload`
- **Method**: POST
- **Description**: Uploads a CSV dataset to train the model.

**Request**:
```bash
curl -X POST -F "file=@dataset/ai4i2020.csv" http://127.0.0.1:5000/upload
```
**Response**:
```bash
{
  "message": "File uploaded successfully!"
}
```
### 2. Train Model
- **Endpoint**: `/train`
- **Method**: POST
- **Description**: Trains the decision tree model using the uploaded dataset.
**Request**:
```bash
curl -X POST http://127.0.0.1:5000/train
```
**Response**:
```bash
{
  "accuracy": 0.85,
  "classification_report": "precision recall f1-score support...",
  "confusion_matrix": [[94, 16], [15, 79]]
}
```
### 2. Predict Machine Failure
- **Endpoint**: `/predict`
- **Method**: POST
- **Description**: Trains the decision tree model using the uploaded dataset.
**Request**:
```bash
curl -X POST -H "Content-Type: application/json" -d "{\"Rotational speed [rpm]\": 200, \"Torque [Nm]\": 40.0}" http://127.0.0.1:5000/predict
```
**Response**:
```bash
{
  "Machine Failure": "Yes",
  "Confidence": 85.0
}
```
## Conclusion

This project provides a functional API for predicting machine failures using a decision tree model built with Python, Flask, and scikit-learn. It is designed for ease of use, extensibility, and integration with other systems.
