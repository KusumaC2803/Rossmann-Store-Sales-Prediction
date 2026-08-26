# 📈 Rossmann Store Sales Prediction

An end-to-end **time-series machine learning project** for forecasting daily sales across Rossmann retail stores using historical sales, promotions, holidays, store characteristics, competition information, and calendar-based features.

The project demonstrates how incorporating **lag and rolling-window time-series features** can significantly improve retail sales forecasting performance.

---

## 🎯 Project Objective

Retail businesses need reliable sales forecasts to support:

* Inventory planning
* Workforce allocation
* Promotional planning
* Demand forecasting
* Store-level operational decisions
* Business strategy

The objective of this project is to build a machine learning pipeline that predicts **daily sales for Rossmann stores** using historical and business-related information.

### Problem Type

**Multivariate Time-Series Forecasting + Regression**

### Target Variable

`Sales`

---

## 📊 Dataset

This project uses the **Rossmann Store Sales** dataset from Kaggle.

The dataset contains historical sales information for **1,115 Rossmann stores**.

### Dataset Files

```text
data/
├── train.csv
├── test.csv
└── store.csv
```

### `train.csv`

Contains historical store-level observations including:

* Store
* DayOfWeek
* Date
* Sales
* Customers
* Open
* Promo
* StateHoliday
* SchoolHoliday

### `test.csv`

Contains future store/date observations for which `Sales` must be predicted.

### `store.csv`

Contains additional store information:

* StoreType
* Assortment
* CompetitionDistance
* CompetitionOpenSinceMonth
* CompetitionOpenSinceYear
* Promo2
* Promo2SinceWeek
* Promo2SinceYear
* PromoInterval

---

## 🔎 Exploratory Data Analysis

The project performs exploratory analysis to understand the major factors affecting sales.

### Analyses performed

* Daily sales trends
* Sales by day of week
* Promotion impact
* Store-type sales comparison
* Overall sales statistics

### Dataset Summary

| Metric               |    Value |
| -------------------- | -------: |
| Usable sales records |  844,338 |
| Number of stores     |    1,115 |
| Total recorded sales |   ~5.87B |
| Average sales        | 6,955.96 |
| Median sales         |    6,369 |
| Maximum sales        |   41,551 |

EDA outputs are stored in:

```text
outputs/eda/
├── daily_sales_trend.png
├── sales_by_weekday.png
├── promotion_impact.png
├── store_type_sales.png
└── eda_summary.csv
```

---

## ⚙️ Data Preprocessing

The preprocessing pipeline performs:

1. Dataset loading
2. Train/store data merging
3. Date conversion
4. Chronological sorting
5. Missing-value handling
6. Categorical-value cleaning
7. Mixed datatype correction
8. Removal of zero-sales observations for regression modeling

The store information is merged using:

```text
Store
```

This enriches the historical sales records with store-level characteristics.

---

## 🧠 Feature Engineering

The project combines standard business features with time-series features.

### Calendar Features

* Year
* Month
* Day
* Week of Year
* Quarter
* Day of Week
* Weekend indicator

### Promotion Features

* Promo
* Promo/competition interaction
* Promo2 activity

### Competition Features

* Competition distance
* Competition opening duration

### Time-Series Features

Historical sales behavior is represented using:

* `Sales_Lag_1`
* `Sales_Lag_7`
* `Sales_Lag_14`
* `Sales_Lag_28`
* `Sales_Rolling_7`
* `Sales_Rolling_14`
* `Sales_Rolling_28`

Lag features allow the model to learn relationships between current demand and previous sales observations.

Rolling averages capture short- and medium-term sales trends.

---

## 🤖 Machine Learning Approach

### Baseline Model

A Random Forest regression model was initially trained using standard business and calendar features.

### Improved Model

The second model incorporates historical sales behavior through lag and rolling-window features.

```text
Historical Data
       ↓
Data Preprocessing
       ↓
Store Data Merge
       ↓
Calendar Features
       ↓
Promotion Features
       ↓
Competition Features
       ↓
Lag Features
       ↓
Rolling Features
       ↓
Chronological Validation
       ↓
Random Forest Regressor
       ↓
Sales Forecast
```

---

## ⏱️ Time-Series Validation

Because sales are time-dependent, the project avoids random shuffling during validation.

The data is split chronologically:

```text
Earlier observations
        ↓
     Training
        ↓
Later observations
        ↓
    Validation
```

This provides a more realistic estimate of how the model performs on future observations.

---

## 📈 Model Performance

### Baseline Random Forest

| Metric |    Score |
| ------ | -------: |
| MAE    | 2,312.52 |
| RMSE   | 3,232.12 |
| R²     |  -0.2231 |
| RMSPE  |   0.5959 |

### Time-Series Random Forest

| Metric |      Score |
| ------ | ---------: |
| MAE    | **654.60** |
| RMSE   | **968.79** |
| R²     | **0.8906** |
| RMSPE  | **0.2870** |

### Improvement

The addition of time-series features produced a substantial improvement.

| Metric |                     Improvement |
| ------ | ------------------------------: |
| MAE    |                ~71.7% reduction |
| RMSE   |                ~70.0% reduction |
| RMSPE  |                ~51.8% reduction |
| R²     | Improved from -0.2231 to 0.8906 |

The results demonstrate that **historical sales patterns are highly valuable for retail demand forecasting**.

---

## 📊 Model Evaluation

The project generates:

* Actual vs Predicted Sales
* Prediction Error Distribution
* Feature Importance
* Model Comparison

Evaluation outputs:

```text
outputs/evaluation/
├── actual_vs_predicted.png
├── error_distribution.png
├── feature_importance.png
├── feature_importance.csv
└── evaluation_summary.csv
```

Model comparison outputs:

```text
outputs/comparison/
├── model_comparison.csv
├── model_improvement.csv
└── model_comparison.png
```

---

## 🔮 Future Sales Prediction

After training the model, the project uses the trained model to generate predictions for the official `test.csv`.

Run:

```bash
python -m src.predict
```

The generated prediction file is:

```text
outputs/submission.csv
```

with the structure:

```text
Id,Sales
1,3848.639717
2,4656.266881
3,5823.318325
...
```

The prediction pipeline ensures that future sales are not directly used as input features.

---

## 📁 Project Structure

```text
Rossman-Store-Sales-Prediction/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── store.csv
│
├── models/
│   ├── random_forest_model.pkl
│   └── random_forest_timeseries.pkl
│
├── notebooks/
│
├── outputs/
│   ├── eda/
│   │   ├── daily_sales_trend.png
│   │   ├── sales_by_weekday.png
│   │   ├── promotion_impact.png
│   │   ├── store_type_sales.png
│   │   └── eda_summary.csv
│   │
│   ├── evaluation/
│   │   ├── actual_vs_predicted.png
│   │   ├── error_distribution.png
│   │   ├── feature_importance.png
│   │   ├── feature_importance.csv
│   │   └── evaluation_summary.csv
│   │
│   ├── comparison/
│   │   ├── model_comparison.csv
│   │   ├── model_improvement.csv
│   │   └── model_comparison.png
│   │
│   ├── timeseries_metrics.csv
│   ├── timeseries_predictions.csv
│   └── submission.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── predict.py
│   ├── eda.py
│   ├── evaluate.py
│   └── model_comparison.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Rossman-Store-Sales-Prediction
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

### 3. Activate the environment

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train and evaluate the model

```bash
python main.py
```

This performs:

1. Data loading
2. Data preprocessing
3. Feature engineering
4. Time-series feature creation
5. Chronological train/validation split
6. Model training
7. Model evaluation
8. Model serialization
9. Validation prediction generation

---

### Run Exploratory Data Analysis

```bash
python -m src.eda
```

Outputs are saved to:

```text
outputs/eda/
```

---

### Run Model Evaluation

```bash
python -m src.evaluate
```

Outputs are saved to:

```text
outputs/evaluation/
```

---

### Compare Models

```bash
python -m src.model_comparison
```

Outputs are saved to:

```text
outputs/comparison/
```

---

### Generate Future Predictions

```bash
python -m src.predict
```

Output:

```text
outputs/submission.csv
```

---

## 💾 Saved Model

The trained time-series model is saved using Joblib:

```text
models/random_forest_timeseries.pkl
```

This allows the trained model to be reused without retraining.

---

## 💡 Key Insights

### 1. Time matters

Sales are strongly dependent on previous sales patterns. Adding lag and rolling features significantly improved model performance.

### 2. Promotions matter

Promotional periods provide important information for predicting demand.

### 3. Store characteristics matter

Store type, assortment, competition distance, and promotion programs provide useful context for sales prediction.

### 4. Chronological validation is important

Random train/test splitting can produce unrealistic results for time-dependent data. Chronological validation better represents real-world forecasting.

### 5. Feature engineering had a major impact

The baseline Random Forest produced an R² of **-0.2231**, while the time-series version achieved **0.8906**.

This demonstrates that appropriate feature engineering can be more important than simply changing the machine learning algorithm.

---

## ⚠️ Limitations

* Random Forest does not explicitly model long-term temporal dependencies.
* Future lag features require careful construction because future sales are unknown.
* RMSPE can be sensitive to very small actual sales values.
* Hyperparameter optimization has not been exhaustively performed.
* More advanced gradient boosting approaches may provide further improvements.
* The current validation score should not be interpreted as the official Kaggle leaderboard score.

---

## 🔮 Future Improvements

Possible improvements include:

* XGBoost or LightGBM
* Hyperparameter optimization
* Store-specific forecasting models
* Advanced rolling statistics
* Seasonal decomposition
* Ensemble forecasting
* SHAP-based explainability
* Forecast uncertainty estimation
* REST API deployment
* Dockerization
* Automated model monitoring

---

## 🛠️ Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Core programming language |
| Pandas       | Data processing           |
| NumPy        | Numerical operations      |
| Scikit-learn | Machine learning          |
| Matplotlib   | Visualization             |
| Seaborn      | Statistical visualization |
| Joblib       | Model serialization       |

---

## 📌 Results Summary

The final time-series Random Forest model achieved:

```text
MAE   = 654.60
RMSE  = 968.79
R²    = 0.8906
RMSPE = 0.2870
```

Compared with the baseline, the improved model achieved a substantial reduction in prediction error by incorporating historical sales patterns.

---

## 👩‍💻 Author

**Kusuma C**

AI / Machine Learning Project

---

## ⭐ Project Highlights

* End-to-end ML pipeline
* Multivariate time-series forecasting
* Retail sales prediction
* Chronological validation
* Lag and rolling features
* Model comparison
* Feature importance analysis
* EDA visualization
* Future sales prediction
* Reusable serialized model
* Kaggle-style submission generation
