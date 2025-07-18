# 🇨🇦 Canada Housing Price Prediction Pipeline

This project develops a multi-layered pipeline to predict housing prices in Canada by combining machine learning, spatial (GIS) features, and macroeconomic indicators.

---

## Project Motivation

In complex real estate markets like Vancouver or Toronto, housing prices are not only affected by property attributes but also by geographic and economic context. This project aims to build an extendable, modular pipeline for:

- Predicting property prices based on structural features
- Enhancing accuracy using spatial micro-level features (e.g. transit, school zones)
- Improving market forecasting by integrating macroeconomic trends (e.g. interest rates, CPI)

---

## Project Architecture
[Raw Housing Data] → [Base Model]
                    ↓
+ [Micro Features: GIS, Environment, Accessibility]
                    ↓
+ [Macro Features: Interest Rates, CPI, Policy]
                    ↓
[Final Prediction Engine]
                    ↓
[Automated Reports / Region Estimations]


---

##  3-Phase Model Structure

### 1. Base Model
- Input: Structural housing features (e.g. area, year built, type)
- Method: Regression models (e.g. Linear, XGBoost)
- Output: Baseline price prediction

### 2️. Micro Feature Enhancement
- Input: GIS-based proximity data (e.g. distance to subway, green space, schools)
- Tools: GeoPandas, QGIS, spatial joins, buffer analysis
- Output: Enhanced local sensitivity & neighborhood profiling

### 3️. Macro Feature Integration
- Input: Monthly/quarterly national indicators (e.g. BoC interest rate, CPI, HPI)
- Tools: Time-series merge & forecasting models
- Output: Market-aware prediction ready for long-term planning

---

##  Tech Stack

| Layer        | Tools Used                         |
|--------------|------------------------------------|
| Data Prep    | Pandas, GeoPandas, Jupyter         |
| GIS Analysis | QGIS, OSM, Shapefiles, Geocoding   |
| ML Modeling  | Scikit-learn, XGBoost, LightGBM    |
| Reporting    | Matplotlib, Seaborn, Jinja2 (HTML reports) |
| Versioning   | Git, GitHub                        |

---

## Folder Structure
realestate_sim/

├── data/ # Raw CSVs, shapefiles, geojsons

├── scripts/ # Python modules for cleaning, modeling

├── notebooks/ # EDA & model development

├── output/ # Predictions, graphs, and reports

├── templates/ # (Optional) Report templates

├── figures/ # Static images for README or dashboard

├── main.py # Entry point for CLI pipeline

└── README.md # You are here


---

## Goals & What’s Next

-   Build base model using real housing data
-   Integrate GIS-derived micro-features
-   Add macroeconomic forecasting layer
-   Visualize prediction accuracy across regions
-   Enable potential region recommendation system for developers

---

## Author

**Chih-Wei Chuang**  
_M.Sc. Data Science | Based in Vancouver, BC_  
🔗 GitHub: [johnwoody-pn](https://github.com/johnwoody-pn)

---

##  License

This project is for educational and demonstration purposes. Commercial use requires permission.

