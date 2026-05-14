# Disease Prediction System

A machine learning application that predicts diseases based on patient age, gender, and symptoms. It combines four classifiers into an ensemble model and provides a Tkinter desktop GUI for interactive predictions.

## Dataset

`Data/Healthcare.csv` — contains patient records with the following columns:

- `Patient_ID` — dropped during preprocessing
- `Age` — numeric
- `Gender` — Male / Female / Other
- `Symptoms` — comma-separated symptom strings
- `Disease` — target label

## How It Works

### 1. Data Loading
Reads the CSV file and reports the number of rows and columns loaded.

### 2. Preprocessing
- Drops `Patient_ID`
- Fills missing `Age` with the median, missing `Gender` with "Other"
- Parses the `Symptoms` column into lists and encodes them using `MultiLabelBinarizer`
- Encodes `Gender` as an integer (Male=0, Female=1, Other=2)
- Encodes the `Disease` target with `LabelEncoder`
- Final feature matrix: `Age` + `Gender` + binary symptom columns

### 3. Model Training
Trains four individual classifiers and combines them into a soft-voting ensemble:

| Model | Notes |
|---|---|
| Random Forest | 200 trees, balanced class weights |
| Logistic Regression | max 1000 iterations, balanced weights |
| Naive Bayes | GaussianNB |
| SVM | RBF kernel, probability=True, balanced weights |
| **Ensemble** | VotingClassifier (soft voting) |

### 4. Evaluation
Prints accuracy for each individual model and the ensemble, plus the ensemble confusion matrix, to the console.

### 5. Prediction
Takes age, gender, and a list of symptoms, encodes them the same way as training data, and returns:
- Predicted disease name
- Confidence score (max class probability %)
- Risk level: **High**, **Medium**, or **Low**

### 6. GUI
A Tkinter window where the user can:
- Enter age and select gender
- Tick symptom checkboxes or type symptoms manually
- Click **Predict Disease** to see the result
- Get a warning popup for high-risk predictions

## Requirements

```
numpy
pandas
scikit-learn
tkinter (standard library)
```

Install dependencies:

```bash
pip install numpy pandas scikit-learn
```

## Usage

```bash
python disease_prediction.py
```

The script trains the models on startup (roughly 30 seconds), prints evaluation results to the console, then opens the GUI.
