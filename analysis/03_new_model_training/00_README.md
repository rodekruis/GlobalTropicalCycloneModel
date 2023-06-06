## Report of analysis: code and steps

All the codes from number 01 to 15 are related to the analysis of the
input data, features, and the three main implemented models (regression, classification, and hybrid one).

Code 01 <br />
[Input data collection](01_collate_data.ipynb)

In this code we collect and combine the data needed to train the model including windspeed, rainfall and etc.

Code 02 <br />
[Naive baseline model](02_model_training-baselines.ipynb)

In this code, we implemented a Naive baseline model as a simple algorithm to provide predictions without complex computations.

Code 03 <br />
[XGBoost regression model (on building counts)](03_model_training.ipynb)

In this code, we trained the XGBoost regression model obtained from the baseline model analysis, on the collected dataset (data in grid format). 

Code 03 <br />
[XGBoost regression model (on houses counts)](03_model_training_updated_with_houses.ipynb)

In this code, we trained the XGBoost regression model obtained from the baseline model analysis, on the collected dataset (data in grid format). The code 02 was updated to train the model on input data using the total_houses instead of total_building since there are distinct concepts for buildings and houses. Consequently, the target variable percent_damage_building was replaced with percent_damage_houses to accurately represent the damage percentage for houses.

Code 04 <br />
[Correlation Matrix](04_Correlation_Matrix.ipynb)

In this code, we utilized a correlation matrix to estimate the Pearson correlation coefficient values of features in the dataset and figure out the highly correlated features.

Code 05 <br />
[Running Mean](05_Moving_Average.ipynb)

In this code, we used this code to simply check how the two certain variables are related to each other.

Code 07 <br />
[XGBoost Regression Model - Feature Importance](07_Feature_Importance.ipynb)

Feature Importance based on XGBoost regression model.
In this code, we applied feature importance to input data using SHAP values and built-in XGBoost feature importance plot. 

Code 08 <br />
[new variable: "percent_houses_damaged_5years"](08_historical_variable.ipynb)

In this code, we added a new variable to the dataset. To estimate this new feature, an average of damages caused by typhoons over the past 5 years will be calculated for every data point and recorded as the value for this new variable.

Code 09 <br />
[Binary Model - Random Forest](09_binary_model-RandomForest.ipynb)

In this code, 10% is considered a reasonable house damage threshold to convert the continuous target into a binary one. Afterward, we implemented Random Forest Classification algorithm on the input dataset with this converted binary target.

Code 09 <br />
[Binary Model - XGBoost](09_binary_model-Xgboost.ipynb)

In this code, 10% is considered a reasonable house damage threshold to convert the continuous target into a binary one. Afterward, we implemented XGBoost Classification algorithm on the input dataset with this converted binary target.

Code 09 <br />
[Binary Logistic Regression Model](09_binary_model_LogisticRegr.ipynb)

In this code, 10% is considered a reasonable house damage threshold to convert the continuous target into a binary one. Afterward, we implemented Logistic Regression algorithm on the input dataset with this converted binary target.

Code 09 <br />
[XGBoost (different n_estimators)](09_binary_Xgboost-different_n_estimators.ipynb)

In this code, we track and compare the performance of the XGBoost binary model by plotting the f1_score macro average VS different n_estimators.

Code 10 <br />
[XGBoost Classification Model - Feature Importance](10_Feature_Importance(SHAP)_Xgboost_binary_model.ipynb)

Feature Importance based on XGBoost classification model.
In this code, we applied feature importance to input data using SHAP values and built-in XGBoost feature importance plot. 

Code 11 <br />
[ROC CURVE for Binary Models (Binary Logistic Regression, Random Forest, and XGBoost)](11_ROC_CURVE_xgb_rf_lregr.ipynb)

In this code, We used the ROC curve to have a graphical representation of the performance of the three binary classification models and compare their performance.

Code 12 <br />
[Regression Model (Data Resampling)](12_XGBoost_Regression_resampling.ipynb)

In this code, we utilized the SMOTE technique to reduce the class imbalance in the continuous target data by oversampling the minority class.

Code 13 <br />
[RMSE estimation for region(adm1)](13_RMSE_for_region_lastVersion.ipynb)

In this code, we decided to check how the model can perform for a wide area.
For this reason, we imported two CSV files, one including grid_id and municipality_code (ADM3) and the other one including region name and code (ADM1). We joined these pieces of information with the typhoon-based prediction data frame to estimate the difference between real and predicted damage per region with respect to each typhoon.

Code 14 <br />
[Combined Model (XGBoost Undersampling + XGBoost Regression)](14_Combined_model_LastVersion.ipynb)

In this code, we decided to improve the performance of model for high bins (high damaged values).
Therefore, we developed a hybrid model using both XGBoost regression and XGBoost classification with undersampling technique.

Code 14 <br />
[Combined Model in a loop](14_Combined_model_in_Loop.ipynb)

In this code, we defined a loop of m iteration.
Since we faced some variation in the result of hybrid model in multiple runs, we inserted the whole code of hybrid model in a loop to have an m average of RMSE estimation.

Code 14 <br />
[Combined Model typhoon split (eave-one-out cross-validation)](14_Combined_model_train_test_split_typhoon.ipynb)

In this code, we defined a loop with the length of number of typhoons.
We evaluated the performance of the combined model by conducting multiple iterations of a for loop, where each iteration used a different typhoon as the test set, while the remaining typhoons were included in the training set. This approach allowed us to estimate the model's performance when the training and test data were split based on all typhoons.

Code 15 <br />
[Combined Model typhoonTime split (walk forward evaluation)](15_Combined_model_train_test_split_typhoonTime(undersampling).ipynb)

The idea of this code is to determine how well the model performs in learning from older typhoons' characteristics to make predictions on the target value of the most recent ones.
In this code, we defined a loop with the length of 12 since the training/test ratio is considered 70:30.
We run the hybrid model to estimate the RMSE while the split of training and test data is done based on the typhoon's time. Therefore, 12 of most recent typhoons wrt times are considered as the test set and the rest as the training set. In each iteration a new typhoon is added to the training set, and the model is tested on the next one.
