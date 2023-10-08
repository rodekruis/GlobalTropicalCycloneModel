# Report of analysis: code and steps

All the codes from number 01 to 15 are related to the analysis of the
input data, features, and the three main implemented models
(regression, classification, and hybrid one).

Code 01
[Input data collection](01_collate_data.ipynb)

In this code, we collect and combine the input data including windspeed,
rainfall and etc. However, some data like Topography or building data were
converted from municipality to grid-based format.

Achievements: The input data for the model is ready
and it is used to train the model.

Code 02
[new variable: "percent_houses_damaged_5years"](02_historical_variable.ipynb)

In this code, we added a new variable to the dataset.
To estimate this new feature, an average of damages caused by typhoons
over the past 5 years will be calculated for every data point and recorded as
the value for this new variable.

Achievements: This newly generated variable is added to the dataset and could
slightly improve the model performance.

Code 03
[Naive baseline model](03_model_training-baselines.ipynb)

In this code, we implemented a Naive baseline model as a simple algorithm to
provide predictions without complex computations.
The Baseline returns the average of the target from training data.
It should be mentioned that the Naive baseline assumes independence
among features.

Achievements: Since this algorithm is easy to implement and understand and
even efficient for large datasets,
it was a good option to give us fast basic results.

Code 04
[XGBoost regression model (on building counts)](04_model_training.ipynb)

In this code, we trained the XGBoost regression model obtained from the
baseline model analysis, on the collected dataset (data in grid format).

Achievements: Since the model is the same regression model we implemented
for the municipality dataset, so there is a possibility to compare the
performance of the model on grid data with municipality data for the same
set of features to check how well the model performs wrt provided dataset.

Code 04
[XGBoost regression model (on houses counts)](04_model_training_updated_with_houses.ipynb)

In this code, we trained the XGBoost regression model obtained from the baseline
model analysis, on the collected dataset (data in grid format).
Code number 02 was updated to train the model on input data using the total_houses
instead of total_building since there are distinct concepts for buildings and houses.
Consequently, the target variable percent_damage_building was replaced with
percent_damage_houses to accurately represent the damage percentage for houses.

Achievements: The obtained results illustrate that the performance of the model
improved when we replaced house data with building data.

Code 05
[Correlation Matrix](05_Correlation_Matrix.ipynb)

In this code, we utilized a correlation matrix to estimate the
Pearson correlation coefficient values of features in the dataset and
figure out the highly correlated features.

Achievements: To find the highly correlated features we used the same threshold
that we had for the municipality dataset (0.8), and the obtained result
indicates that some features in the same category are highly correlated
e.g. mean_slope and mean_tri or total_houses and total_pop.

Code 06
[Running Mean](06_Moving_Average.ipynb)

Since we want to better understand the relationship between some of the
most important features together and/or with the target variable,
we used the moving average which is a statistical technique.
Applying a moving average to non-time series data can improve data visualization.
By reducing noise and highlighting trends, it can provide a clearer and more
intuitive representation of the data, making it easier to communicate and interpret.

Achievements: This code simply checks how the two certain variables are
related to each other e.g. relationship between wind_speed or rainfall_max_6h
with the target variable.

Code 07
[XGBoost Regression Model - Feature Importance](07_Feature_Importance.ipynb)

Feature Importance based on XGBoost regression model.
We applied feature importance to input data using SHAP values and a built-in
XGBoost feature importance plot to find which features are the most effective
ones in model prediction. Since these techniques may produce different rankings
of feature importance in multiple executions so it is useful to compare their
results to get a more comprehensive understanding of the importance of
different features.

Achievements: One of the strong results represents that wind speed always has
a positive impact on the prediction of the model.

Code 08
[Binary Model - Random Forest](08_binary_model-RandomForest.ipynb)

Since so far we applied a regression model to the grid-based data, we want to
also check the performance of a classification model on this dataset.
Therefore, 10% is considered a reasonable house damage threshold to convert the
continuous target into a binary one. Afterward, we implemented
Random Forest Classification algorithm on the input dataset with this
converted binary target.

Achievements: The confusion matrix and classification report are used to
display how the model performs. We can say that the model is not very stable
in different executions and is a bit prone to overfitting.

Code 08
[Binary Model - XGBoost](08_binary_model-Xgboost.ipynb)

Since so far we applied a regression model to the grid-based data,
we want to also check the performance of a classification model on this dataset.
Therefore, 10% is considered a reasonable house damage threshold to convert the
continuous target into a binary one. Afterward, we implemented the
XGBoost Classification algorithm on the input dataset with this
converted binary target.

Achievements: The confusion matrix and classification report are used to
display how the model performs. We can say that the model is not very stable
in different executions and is a bit prone to overfitting.
However, this binary model has the best performance among all three
classification models.

Code 08
[Binary Logistic Regression Model](08_binary_model_LogisticRegr.ipynb)

Since so far we applied a regression model to the grid-based data,
we want to also check the performance of a classification model on this dataset.
Therefore, 10% is considered a reasonable house damage threshold to convert the
continuous target into a binary one. Afterward, we implemented the
Logistic Regression algorithm on the input dataset with this
converted binary target.

Achievements: The confusion matrix and classification report are used to
display how the model performs. We can say that this model is very sensitive to
highly correlated features so we need to remove those features before using
data to train the model.

Code 08
[XGBoost (different n_estimators)](08_binary_Xgboost-different_n_estimators.ipynb)

Since the XGBoost binary model has the best performance, we track and compare
the performance of the XGBoost binary model by plotting the f1_score macro
average VS different n_estimators.

Achievements: Among the n_estimators from 10 to 100 there is not a
slight difference in estimated F1_score.

Code 09
[XGBoost Classification Model - Feature Importance](09_Feature_Importance(SHAP)_Xgboost_binary_model.ipynb)

Feature Importance based on XGBoost classification model.
The classification model can help in selecting important features by
identifying the most informative and relevant features for the classification task.
In this code, we applied feature importance to input data using SHAP values
and a built-in XGBoost feature importance plot.

Achievements: The result represents that the most effective features for
prediction in the binary model are very similar to the regression model
e.g. wind_speed.

Code 10
[ROC CURVE for Binary Models (Binary Logistic Regression, Random Forest, and XGBoost)](10_ROC_CURVE_xgb_rf_lregr.ipynb)

In this code, We used the ROC curve to have a graphical representation of the
performance of the three binary classification models and compare their performance.

Achievements: A visual comparison of the performance of three binary models in
one plot.

Code 11
[Regression Model (Data Resampling)](11_XGBoost_Regression_resampling.ipynb)

In this code, we utilized the SMOTE technique to reduce the class imbalance
in the continuous target data by oversampling the minority class, and to achieve
this goal we create a binary target variable to serve as an auxiliary variable
for resampling the training data.

Achievements: Estimating the accuracy of a resampled regression model and
comparing it with the performance of the regression model before resampling.

Code 12
[RMSE estimation for region(adm1)](12_RMSE_for_region_lastVersion.ipynb)

In this code, we decide to check how the model can perform for a wide area.
For this reason, we imported two CSV files, one including grid_id and
municipality_code (ADM3) and the other one including region name and code (ADM1).
We joined these pieces of information with the typhoon-based prediction data frame
to estimate the difference between real and predicted damage per region with respect
to each typhoon.

Achievements: The obtained result of this code is the prediction error for each
region according to a group of typhoons in the test set and it illustrates that the
model can not perform well for a wider area.

Code 13
[Combined Model (XGBoost Undersampling + XGBoost Regression)]
(13_Combined_model_LastVersion.ipynb)

In this code, we decide to improve the performance of the model for high bins
(high damaged values).
Therefore, we developed a hybrid model using both XGBoost regression and XGBoost
classification with the undersampling technique.

Achievements: The attainment indicates that we could improve the performance of
the model for the higher bins by implementing this hybrid model while we will
obtain the worst result for low-damaged bins.

Code 13
[Combined Model in a loop](13_Combined_model_in_Loop.ipynb)

In this code, we define a loop of m iteration.
Since we faced some variation in the result of the hybrid model in multiple runs,
we inserted the whole code of the hybrid model in a loop to have an m average of
RMSE estimation.

Achievements: Since the result of the combined model is not stable,
a loop can give us the average result which is more trustable.

Code 14
[Combined Model typhoon split (leave-one-out cross-validation)]
(14_Combined_model_train_test_split_typhoon.ipynb)

In this code, we define a loop with the length of the number of typhoons.
We evaluated the performance of the combined model by conducting multiple
iterations of a for loop, where each iteration used a different typhoon as
the test set, while the remaining typhoons were included in the training set.
This approach allowed us to estimate the model's performance when the training
and test data were split based on all typhoons.

Achievements: The total RMSE displays that we have better performance of
the combined model in typhoon split(leave-one-out cross-validation) than
train-test split while worst performance for highly damaged bins.

Code 14
[Combined Model (using oversampling of minority class)
typhoonTime split (walk forward evaluation)](14_Combined_model_train_test_split_typhoonTime(oversampling).ipynb)

The idea of this code is to determine how well the model performs in
learning from older typhoons' characteristics to make predictions on the
target value of the most recent ones.
In this code, we defined a loop with a length of 12 since the training/test ratio
is considered 70:30.
We run the hybrid model to estimate the RMSE while the split of training and
test data is done based on the typhoon's time. Therefore, 12 of the most recent
typhoons with respect to times are considered as the test set and the rest as
the training set. In each iteration, a new typhoon is added to the training set,
and the model is tested on the next one.

Note: It is necessary to mention that in this code, instead of undersampling in
the classification section of the combined model we use the oversampling technique.

Achievements: The total RMSE displays that we have slightly better performance of
the combined model in typhoon split of (walk forward evaluation) than
(leave-one-out cross-validation) with better performance for highly damaged bins.

Code 14
[Combined Model (using undersampling of majority class) typhoonTime split
(walk forward evaluation)](14_Combined_model_train_test_split_typhoonTime(undersampling).ipynb)

The idea of this code is to determine how well the model performs in
learning from older typhoons' characteristics to make predictions on the
target value of the most recent ones.
In this code, we defined a loop with the length of 12 since the training/test ratio
is considered 70:30.
We run the hybrid model to estimate the RMSE while the split of training and
test data is done based on the typhoon's time. Therefore, 12 of most recent
typhoons wrt times are considered as the test set and the rest as the training set.
In each iteration a new typhoon is added to the training set,
and the model is tested on the next one.

Achievements: The total RMSE displays that we have better performance of
combined model with undersampling than oversampling in typhoon split
(walk forward evaluation) with even better performance for high damaged bins.
