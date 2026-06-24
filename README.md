# 🏠 Singapore HDB Resale Price Predictor

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning project designed to predict the resale prices of Housing & Development Board (HDB) flats in Singapore. By analyzing historical transaction data and integrating geospatial features, this model helps buyers, sellers, and agents estimate fair market valuations.

---

## 📌 Project Overview
Predicting HDB resale prices is a complex regression problem. Property values are heavily influenced not just by the flat's inherent physical attributes (size, age, floor level), but also by macroeconomic trends and proximity to critical amenities like MRT stations, school zones, and commercial centers. 

This project handles the complete lifecycle: **Data Extraction ➔ Feature Engineering ➔ Geospatial Mapping ➔ Model Training ➔ Web App Deployment**.

---

## 📊 Dataset & Features
The baseline transactional data is sourced from [Data.gov.sg](https://data.gov.sg/) (HDB Resale Flat Prices).

### Engineered Features
To significantly improve model accuracy, raw data was augmented with auxiliary location-based data (via **OneMap API**):
* **Lease Decay:** Calculated remaining lease years based on the transaction date and lease commencement year.
* **Storey Elevation:** Converted the `storey_range` string (e.g., "04 TO 06") into a numerical median value (`5`).
* **Proximity to MRT:** Walking distance (in meters) to the nearest MRT/LRT station.
* **Proximity to CBD:** Distance to Raffles Place.
* **Amenities:** Distances to the nearest shopping mall, primary school, and hawker center.

---

## 🛠️ Tech Stack
* **Data Wrangling:** `pandas`, `numpy`
* **Geospatial Processing:** `geopy`, `requests` (OneMap API integration)
* **Machine Learning:** `scikit-learn`, `LightGBM`
* **Web Application:** `fastapi`


