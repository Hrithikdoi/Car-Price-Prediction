# Car Price Prediction

A regression project that predicts car prices from nine features. It covers data cleaning, exploratory analysis, preprocessing, a comparison of six models, hyperparameter tuning, and a Flask web app that returns a predicted price from a form.

## Dataset

`raw_data.xlsx` has 10,000 rows and no missing values or duplicates.

| Column | Description |
|---|---|
| Brand, Model | 10 brands and 30 models |
| Year | 2000 to 2023 |
| EngineSize | Engine size (1.0 to 5.0) |
| Fuel | Petrol, Diesel, Hybrid, Electric |
| Transmission | Manual, Automatic, Semi-Automatic |
| Mileage, Doors, OwnerCount | Numeric |
| Price | Target, from 2,000 to 18,301 (mean about 8,850) |

## Workflow

Run the notebooks in this order. They pass data to each other with `%store`.

1. **`Cleaningprocess.ipynb`:** holds out 20% of the data as a test set (2,000 rows), then checks data types, missing values, duplicates and category values on the remaining 8,000.
2. **`EDA.ipynb`:** distribution plots, correlation with price, and ANOVA for each categorical column.
3. **`PreProcessing.ipynb`:** mean target encoding for Brand and Model, one-hot encoding for Fuel and Transmission. The encoders are saved as pickle files.
4. **`DataSplitting.ipynb`:** splits the 8,000 rows into 6,400 for training and 1,600 for validation.
5. **`Trying_Algorithms.ipynb`:** trains and compares six models with 5-fold cross-validation, tunes the best one, and scores it on the test set.

Final split: 6,400 train, 1,600 validation, 2,000 test.

## Exploratory Findings

- **Numeric features:** Year has a positive correlation with price (0.66), Mileage a negative one (-0.55) and EngineSize a weaker positive one (0.36). Doors and OwnerCount have almost none.
- **Fuel:** Electric cars have the highest average price (10,061), then Hybrid (9,094), Diesel (8,092) and Petrol (8,018).
- **Transmission:** Automatic averages 9,955, against 8,338 for Manual and 8,205 for Semi-Automatic.
- **Brand and Model:** average price barely changes between brands (the top five brands sit within about 130 of each other). ANOVA F-statistics were 1.40 for Brand and 0.91 for Model, against 207.6 for Fuel and 274.2 for Transmission.

## Model Comparison

Validation set (1,600 rows). StandardScaler was used for Linear Regression, SVR and XGBoost.

| Model | Validation R² | Validation RMSE |
|---|---|---|
| Linear Regression | 0.9996 | 63.30 |
| Decision Tree | 0.8941 | 1011.53 |
| Random Forest | 0.9630 | 597.52 |
| SVR (RBF) | 0.0855 | 2971.90 |
| XGBoost | 0.9928 | 264.41 |
| CatBoost (default) | 0.9997 | 49.63 |
| CatBoost (tuned) | 0.9994 | 78.35 |

The tuned CatBoost model was saved as the final model. CatBoost was tuned with `GridSearchCV` (5-fold, 108 parameter combinations, 540 fits). The best parameters were `depth=6`, `iterations=200`, `learning_rate=0.1`, `l2_leaf_reg=1` and `border_count=64`.

**Test set result (2,000 rows): R² 0.9994, RMSE 75.21.**

## Notes on the Results

- The scores are unusually high. The data appears to be synthetic: price follows a near-linear pattern in year, engine size, fuel type and transmission, and Brand and Model add almost nothing. This is why plain Linear Regression scores as well as CatBoost.
- The results show how the pipeline performs on this dataset. They should not be read as accuracy on real car listings.
- Brand and Model encodings were calculated on the 8,000 rows before the train/validation split, so the validation score is slightly optimistic. The test set was held out before any encoding, so the test score is not affected.

## Web App

`app.py` is a Flask app. It loads the saved model and encoders from `models/`, takes Brand, Model, Fuel, Transmission, Year, EngineSize, Mileage, Doors and OwnerCount from a form, applies the same encoding as training, and returns the predicted price. It handles both normal form posts and AJAX requests.

## How to Run

```bash
pip install flask pandas numpy scikit-learn catboost
python app.py
```

Then open http://127.0.0.1:5000 in a browser. Flask looks for `index.html` inside a `templates/` folder.

## Repository Structure

```text
Car-Price-Prediction/
├── models/          # Model.pkl, Brand_Encoder.pkl, Model_Encoder.pkl, OneHot_Encoder.pkl
├── src/             # notebooks
├── templates/
│   └── index.html
├── app.py
├── raw_data.xlsx
└── README.md
```

## Tools Used

Python, Pandas, NumPy, Scikit-learn, XGBoost, CatBoost, Statsmodels, Matplotlib, Seaborn, Flask

## Author

**Hrithik Doiphode**
GitHub: https://github.com/Hrithikdoi