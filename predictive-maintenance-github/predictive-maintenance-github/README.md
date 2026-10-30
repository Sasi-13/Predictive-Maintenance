# Smart Predictive Maintenance System

A machine-learning project that estimates the likelihood of machine failure from sensor measurements using the AI4I 2020 Predictive Maintenance dataset.

## Features

- Random Forest classification pipeline
- Categorical feature encoding inside the saved pipeline
- Failure probability estimation
- Streamlit dashboard for individual predictions
- CSV batch prediction and downloadable results
- Evaluation metrics: accuracy, precision, recall, F1-score, ROC-AUC and confusion matrix

## Dataset

1. Download the AI4I 2020 Predictive Maintenance Dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).
2. Put the CSV file in `dataset/`.
3. Rename it to `ai4i2020.csv`.

The dataset is not included in this GitHub repository. Review the dataset's license and terms before redistribution.

## Requirements

- Python 3.9 or later
- pip

## Setup

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY

python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

This creates:

- `models/random_forest_pipeline.joblib`
- `reports/metrics.json`

These generated files are ignored by Git. Run the training command after cloning.

## Run the dashboard

```bash
streamlit run app.py
```

Streamlit opens the app in your browser. Enter sensor readings to get a prediction, or upload a CSV with these exact input columns:

- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

## Project structure

```text
predictive-maintenance/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── dataset/
│   └── README.md
├── models/
│   └── README.md
└── reports/
    └── README.md
```

## Important notes

- The AI4I 2020 dataset is a synthetic benchmark dataset; it is not a live industrial sensor feed.
- The model output is an estimate, not a guarantee that a machine will or will not fail.
- The project uses a classification approach. It does not currently forecast exact time-to-failure or Remaining Useful Life (RUL).
- The failure-mode columns are excluded from the input features to avoid target leakage.

## Future improvements

- Compare Random Forest with XGBoost.
- Add time-series data and LSTM/RUL estimation.
- Connect real IoT sensors using MQTT.
- Add user authentication and a database for machine history.
- Deploy the Streamlit app.

## License

Choose a license before publishing this project. If you do not want others to reuse the code, do not add an open-source license.
