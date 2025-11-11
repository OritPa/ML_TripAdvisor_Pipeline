# TripAdvisor ML Pipeline

Predicting Restaurant Success on Tripadvisor (Italy). The dataset was retreived from Kaggle: https://www.kaggle.com/datasets/stefanoleone992/tripadvisor-european-restaurants

A reproducible ML pipeline that predicts whether an Italian restaurant will be a high performer on Tripadvisor (defined as average rating ≥ 4.5 with review volume above the dataset mean). The project walks from raw, messy data to a tuned model with explainable features and deployment-ready artifacts.

Highlights

End-to-end: data cleaning → EDA → imputation & outlier control → feature engineering → feature selection → imbalance handling → model selection & tuning

Leakage-safe: removes rating subcomponents and popularity surrogates used in target definition

Human-context features: spatial density, distance to city center, dietary inclusiveness, operational cadence, prestige × price

Reproducible artifacts: saved datasets, selected features, tuned pipeline, metrics, and a summary deck

Environment
python>=3.10
pip install -r requirements.txt


requirements.txt (minimal):

numpy
pandas
scikit-learn>=1.2
imblearn
matplotlib
seaborn
statsmodels
xgboost
missingno
folium
pygeohash

Quickstart (local)
 1) Data prep  → produces data/interim/df_main.pkl and logs
python src/data_preparation.py

 2) EDA (optional save of figures to artifacts/eda/)
python src/eda.py

 3) Missing values & outliers → data/interim/df_after_outliers.pkl
python src/missings_outliers.py

 4) Feature engineering → data/processed/df_after_feature_engineering.pkl
python src/feature_engineering.py

 5) Modeling: split, preprocess (OHE/median), FS (VT + MI k=30), SMOTE, train candidates,
    leaderboard, grid-search winner, export artifacts/
python src/modeling.py


Outputs you’ll get in artifacts/:

selected_features.csv / .json – top 30 features by mutual information

chosen_params.json – tuned hyperparameters for the winner

selected_model_metrics.csv – test metrics table

best_<Model>_pipeline.joblib – drop-in pipeline: preprocess + FS + SMOTE + model

Project Logic (1–2 lines per phase)

Data Preparation

Scope to Italy; normalize text; rebuild location hierarchy (Region → Province → City); reduce cardinality (top-1 per region + “other_*”).

Define target: 1 if avg_rating ≥ 4.5 and total_reviews_count > mean; else 0.

Remove leakage columns (ratings subcomponents, direct popularity surrogates); persist clean df_main.pkl.

EDA

Confirm leakage via monotonic target relationships with experience scores (food/service/value/atmosphere).

Geography & language matter: Center region outperforms; English default language correlates with success.

Flag redundant pairs (e.g., cuisines vs. tags) and skewed numerics.

Missing & Outliers

KNN impute operational metrics; city/province medians for coords; group-median for max_price then re-bin price_range.

IQR capping (1.5×) on continuous features; correlation stability check; export capping reports.

Feature Engineering

Spatial density within 800 m (BallTree), 0–100 popularity index, tourist-zone flag; distance to city center.

Inclusiveness: vegan/vegetarian/gluten-free triple flag; kosher/halal.

Operational: open all week, shift length; prestige × price (michelin_x_price); foot-traffic address keywords; city size buckets; popularity rank parsing.

Save df_after_feature_engineering.pkl.

Modeling & Tuning

Pipeline: Preprocess (median + OHE) → VarianceThreshold → SelectKBest(MI,k=30) → SMOTE → Classifier.

Candidates: Logistic Regression, Random Forest, XGBoost, Decision Tree, GBM, Linear SVM, AdaBoost.

Leaderboard by ROC-AUC; grid-search winner; save tuned pipeline + metrics and selected features.

Interpreting the Model

Common top signals:

Location & density (distance to center, area_popularity_0_100, region effects)

Visibility (default_language_English)

Inclusiveness & operations (is_full_diet_friendly, is_full_week, shift_length_hours)

Economics & prestige (price_level, michelin_x_price)

Reusing the Model (inference)
import joblib, pandas as pd

pipe = joblib.load("artifacts/best_<Model>_pipeline.joblib")
X_new = pd.read_parquet("data/production/new_restaurants.parquet")
proba = pipe.predict_proba(X_new)[:, 1]  # success probability
pred  = (proba >= 0.5).astype(int)


The exported pipeline already contains preprocessing, feature selection, SMOTE handling (only used during training), and the trained classifier.

Slides

A 10-slide summary deck is included:

presentation/Tripadvisor_Restaurant_Success_Analysis.pptx


Import to Canva or present directly.

Data & Ethics

Input is user-generated, noisy, and may carry biases (tourist hubs, language visibility).

Leakage-prone fields are excluded from training.

Do not commit raw data with private or license-restricted content.


