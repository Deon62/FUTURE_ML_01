

# Spotify Mood Classification Model

## Overview
This project focuses on classifying Spotify songs into three distinct mood categories: **Sad**, **Happy**, and **Energetic** using machine learning. The model is trained using various song features such as **danceability**, **energy**, **loudness**, and more, with the goal of accurately predicting the mood of songs from Spotify's dataset.

## Features:
- **Mood Classification**: Classifies songs into **Sad**, **Happy**, and **Energetic** categories.
- **Model**: Built using **XGBoost Classifier** to classify moods.
- **Cross-Validation**: The model is evaluated using cross-validation to ensure robustness and accuracy.
- **Visualizations**: Includes various visualizations such as confusion matrix, feature importance, and classification reports to evaluate the model's performance.

## Project Structure
```
.
├── data/
│   └── processed_data.csv     # Processed data used for training and testing the model
├── notebooks/
│   └── spotify_mood_classification.ipynb  # Jupyter notebook for model training and evaluation
├── models/
│   └── xgboost_model.pkl      # Saved model
│   └── feature_columns.pkl    # Saved feature columns for inference
├── README.md                 # This readme file
└── requirements.txt           # List of required packages for the project
```

## Requirements

To run this project, make sure to install the necessary dependencies. You can do this by using the following command:

```
pip install -r requirements.txt
```

### Dependencies:
- pandas
- numpy
- xgboost
- scikit-learn
- matplotlib
- seaborn

## Getting Started

### 1. Data Preprocessing:
- The data is cleaned and features like **danceability**, **energy**, **loudness**, etc., are selected for model training.
- The mood column is created to represent the three target categories: **Sad**, **Happy**, and **Energetic**.

### 2. Model Training:
- The model is trained using **XGBoost** on the selected features.
- Cross-validation is used to evaluate the model and avoid overfitting.

### 3. Inference:
- Once the model is trained, it is saved along with the selected feature columns.
- The model can be used to predict the mood of new Spotify songs by providing the required features.

### 4. Visualization:
- Various charts and reports such as **Confusion Matrix**, **Feature Importance**, and **Classification Reports** are generated for model evaluation.

## How to Use the Model
1. **Load the Model:**
   ```python
   import pickle
   
   model = pickle.load(open('models/xgboost_model.pkl', 'rb'))
   feature_columns = pickle.load(open('models/feature_columns.pkl', 'rb'))
   ```

2. **Prepare Data:**
   - Ensure that the song data you wish to classify has the same features as used for training.
   
3. **Predict Mood:**
   ```python
   def classify_song(song_data, model, feature_columns):
       # Ensure the input features are in the same order as trained features
       song_data = song_data[feature_columns]
       return model.predict([song_data])[0]
   ```

4. **Sample Input:**
   - A sample input for a song might look like this:
     ```python
     song = {
         'popularity': 65,
         'duration_ms': 220000,
         'explicit': 0,
         'danceability': 0.8,
         'energy': 0.7,
         'key': 5,
         'loudness': -5,
         'mode': 1,
         'speechiness': 0.05,
         'acousticness': 0.1,
         'instrumentalness': 0.02,
         'liveness': 0.1,
         'valence': 0.6,
         'tempo': 120,
         'time_signature': 4
     }
     
     predicted_mood = classify_song(song, model, feature_columns)
     print(predicted_mood)  # This will print the predicted mood (Sad, Happy, Energetic)
     ```

## Results
- The model achieves **99.99%** accuracy with cross-validation, demonstrating its effectiveness at classifying moods.
- Performance metrics like **Precision**, **Recall**, and **F1-Score** for each class (Sad, Happy, Energetic) are used to evaluate the model.

## Conclusion
The **Spotify Mood Classification Model** effectively predicts the mood of songs from Spotify using a variety of features. With the **XGBoost** model and careful data preprocessing, this solution provides high accuracy and reliability.

## Future Work
- Fine-tuning the model for further improvement.
- Incorporating additional features such as lyrics or user data.
- Deploying the model in a real-time application for mood-based song recommendations.
