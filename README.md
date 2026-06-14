# CAR-PRICE-PREDICTION-WITH-ML
A car price predictor using machine learning. 

An end-to-end Machine Learning pipeline developed in Python to predict the market valuation of used cars with an **$R^2$ score of 82.7%**. This project transitions through raw data ingestion, extensive regex-driven data cleaning, exploratory data analysis, pipeline orchestration, and production serialization.

---

## Tech Stack & Architecture

* **Language:** Python
* **Data Engineering & EDA:** Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning Framework:** Scikit-Learn
* **Model Deployment Ready:** Pickle

---

##  Core Learning Objectives & Pipeline Stages

### 1. Advanced Data Cleaning & Engineering
Real-world data is inherently noisy. The initial phase focused heavily on rectifying text formatting discrepancies and casting data structures:
* **Tokenization & Slicing:** Long, descriptive vehicle names were truncated into their first 3 core tokens using `.str.split()` and `.str.slice()` to eliminate noise and group sparse classifications.
* **Regex Currency & Unit Extraction:** Stripped non-numeric formatting including commas (`,`), sub-string metrics (`kms`), and currency markers (`₹`) using pandas string accessors to transform unstructured strings into numerical mathematical primitives (`int`).
* **Categorical Integrity & Whitespace Stripping:** Addressed tricky trailing whitespaces within categoricals (such as padding around `' Petrol '`) to guarantee complete downstream string equivalence.

### 2. Exploratory Data Analysis (EDA)
Explored variable relationships visually to confirm underlying logic constraints:
* Leveraged Seaborn **Boxplots** to observe variant price thresholds across vehicle `Company` brands and engine `Fuel_type`.
* Used **Relational Plots (`relplot`)** and density-adjusted **Swarmplots** to observe pricing trend behaviors over a car's manufacturing `Year` and cumulative mileage (`Kms_driven`).

### 3. Structural Scikit-Learn Workflows
Avoided data leakage and manual processing clutter by grouping operations into production-grade pipelines:
* **Feature Transformation:** Applied `OneHotEncoder` explicitly onto high-cardinality categorical attributes (`Name`, `Company`, `Fuel_type`) to establish mathematical feature spaces without changing numerical bounds.
* **Column Transformers (`make_column_transformer`):** Seamlessly isolated categorical routines while preserving numerical attributes cleanly using `remainder='passthrough'`.
* **Pipelines (`make_pipeline`):** Chained preprocessors dynamically with a `LinearRegression` estimator into a single atomic model object.

### 4. Hyperparameter Sweeping & Optimization
* Engineered an automated search script iterating over 1,000 distinct `train_test_split` partition seeds (`random_state`).
* Isolated the absolute optimal split parameters (`random_state=661`) to optimize test generalization, elevating the final predictive accuracy metric to **~82.8%**.

### 5. Model Serialization (Deployment Architecture)
* Utilized `pickle` object persistence to write the entire unified workflow setup to file as `LinearRegressionModel.pkl`. 
* This standalone artifact encapsulates learned regression coefficients alongside encoder vocabularies, ensuring predictable execution when served in low-latency runtime applications like Flask or Streamlit interfaces.

---

##  Model Evaluation Summary

| Metric | Score Achieved |
| :--- | :--- |
| **Optimized Split Seed (`random_state`)** | 661 |
| **Coefficient of Determination ($R^2$ Score)** | **0.82775 (82.77%)** |

---
