
# Planetary Defense Analytics

A machine learning project to classify, predict, and cluster asteroids using NASA's Small-Body Database.

## Project Overview

This project applies ML to astronomical data for:
- **Case 1:** PHA (Potentially Hazardous Asteroid) Classification
- **Case 2:** Diameter Regression
- **Case 3:** Asteroid Clustering (K-Means + DBSCAN)

## Dataset

- Source: NASA Small-Body Database
- Size: 958,524 rows × 45 columns
- Features: Orbital elements, physical properties, proximity metrics

## Project Structure
├── data/
│ ├── raw/ # Original dataset
│ └── processed/ # Cleaned + preprocessed data
├── models/ # Saved ML models (.pkl)
├── notebook/ # Jupyter notebooks
├── app.py # Streamlit app
├── requirements.txt
└── README.md


## Models Used

### Case 1: PHA Classification
- Logistic Regression
- Random Forest (Best — F1: 0.3072)
- XGBoost

### Case 2: Diameter Regression
- Linear Regression
- Random Forest
- XGBoost

### Case 3: Clustering
- K-Means
- DBSCAN

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

Live Demo
https://planetary-defense-analytics.streamlit.app/

Author : Sureka
GitHub : sureka2006
