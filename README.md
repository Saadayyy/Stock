S&P 500 Stock Prediction using Random Forest Classifier

This repository contains a Python script that utilizes historical S&P 500 data to predict stock movements using a Random Forest Classifier. The script employs the yfinance library to fetch historical stock data and scikit-learn for building and evaluating the predictive model.
Table of Contents

    Installation
    Usage
    Script Overview
    Functions
    Results
    Contributing
    License

Installation

To run the script, you need to have Python 3.x installed along with the following libraries:

    yfinance
    pandas
    scikit-learn
    matplotlib

You can install the required libraries using pip:

sh

pip install yfinance pandas scikit-learn matplotlib

Usage

Clone the repository and navigate to the directory:

sh

git clone https://github.com/yourusername/sp500-stock-prediction.git
cd sp500-stock-prediction

Run the script:

sh

python sp500_prediction.py

Script Overview

The script performs the following steps:

    Fetch Historical Data: Downloads the historical data for the S&P 500 index using yfinance.
    Data Preprocessing: Processes the data by adding target labels and feature columns.
    Model Training: Trains a Random Forest Classifier on the historical data.
    Model Evaluation: Evaluates the model using precision score and visualizes the predictions.

Functions
predict(train, test, predictors, model)

Trains the model on the training set and makes predictions on the test set.

Parameters:

    train (DataFrame): The training dataset.
    test (DataFrame): The testing dataset.
    predictors (list): List of predictor column names.
    model (RandomForestClassifier): The machine learning model to be trained.

Returns:

    combined (DataFrame): DataFrame containing actual targets and predictions.

backtest(data, model, predictors, start=2500, step=250)

Performs backtesting on the dataset by making predictions in a rolling window fashion.

Parameters:

    data (DataFrame): The dataset to be used for backtesting.
    model (RandomForestClassifier): The machine learning model to be used.
    predictors (list): List of predictor column names.
    start (int): The starting index for backtesting.
    step (int): The step size for rolling window.

Returns:

    all_predictions (DataFrame): DataFrame containing all predictions.

Results

The script outputs the precision score of the model and plots the actual vs predicted values. Below are some key results:

    Precision Score: 0.543
    Predictions Count:
        Class 0: 5166
        Class 1: 1018

Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
License

This project is licensed under the MIT License. See the LICENSE file for details.
