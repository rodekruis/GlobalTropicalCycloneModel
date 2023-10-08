## Documentation of all codes in this branch (baseline model)


Code 01: [Main correlation matrix](01_Main_Correlation_Matrix.ipynb)

Goal: This code is utilized to compute the correlation among various features present in the input dataset and let us show a square matrix with dimensions equal to the number of features.

After importing the required libraries, we also import a function named `get_clean_dataset()` from module utils, then we call the function so we have the input data.

Using the following code we create a folder in the specific direction that allows us to store the results of this code.

`output_dir = (Path(os.getenv("STORM_DATA_DIR"))/"analysis/01_baseline_model/output")`

Then we get the Pearson correlation coefficient of features `corrMatrix = df.corr()` and plot a heat map and save it in `output_dir`.

Since the absolute correlation is an easier way to find highly correlated pair of features, we also check that and make a plot.

Therefore, we define a threshold equal to 0.8 and list all those pairs that include a correlation value greater than this threshold.

Finally, we drop the highly correlated features from the data frame and list the remaining features (24 features remain over 37), and make a new heatmap with these remaining 24 features.

In the next step, we decide to compute the Variance inflation factor (VIF) which is another method for finding highly correlated features if there is still in existence.

Brief description about VIF: VIF method, picks each feature and regresses it against all of the other features so the VIF value for a feature demonstrates the correlation of that feature in total with all the other ones, and not only with one specific feature. Normally if the estimated VIF value for a feature is greater than 7 so it can be considered a highly correlated feature.

To calculate the VIF of features we define an empty data frame with a column named "feature" and insert all the remaining features' names in this data frame and estimate the VIF values. Then we store features based on their VIF values in ascending order so it seems we have three features (21, 22, and 23) with values greater than the VIF threshold (7).



Code 02.1: [Feature Importance (Linear Regression Model)](02.1_Feature_Importance-LinearRegression.ipynb)

Goal: This code is utilized to estimate the most important features using the Linear Regression model.

After importing the required libraries, we also import a function named `get_clean_dataset()` from module utils, then we call the function so we have the input data. We make a hist plot of target values and it is obvious that we face an imbalanced dataset.

Since we have a low number of samples for high damage values we stratify data to make sure we will have a range of all samples for both training and test sets `bins2=[0, 1, 60, 101]`. Indeed, the bins categorize target values into different groups called bins.

The NumPy Digitize() function is used to get the indices of bins to which each of these values belongs in the input array.

We list all features after highly correlated ones based on the VIF result and then split features into X and y sets (all features were considered for X and the target variable for y). Since different features have various ranges we standardize X data. The method of standard scalar standardize features by removing the mean and scaling to unit variance.

We define the train_test_split() function to randomly separate training and test data with the ratio of 80:20, while for the option stratify we consider the defined bins. We apply the Linear Regression model to train data.

Then, we get the coefficient value of each feature using `importance = regressor.coef_` and insert the name of features and their coefficient values into a created data frame and put it in descending order.

In the end, using matplotlib.pyplot we make a plot of these features and coefficients.



Code 02.2: [Feature Importance (Random Forest regression model)](02.2_Feature_Importance-RandomForest.ipynb)

Goal: This code is utilized to estimate the most important features using the Random Forest regression model according to two different approaches: 1. SHAP values, 2.Random Forest Built-in Feature Importance.

After importing the required libraries, we also import a function named `get_clean_dataset()` from module utils, then we call the function so we have the input data. We make a hist plot of target values and it is obvious that we face an imbalanced dataset.

Since we have a low number of samples for high damage values we stratify data to make sure we will have a range of all samples for both training and test sets `bins2=[0, 1, 60, 101]`. Indeed, the bins categorize target values into different groups called bins.

The `NumPy.Digitize()` function is used to get the indices of bins to which each of these values belongs in the input array.

We list features and decide to drop highly correlated features where correlation value > 0.99 from X data (so we drop only 5 features) and then split features into X and y sets (all features were considered for X and the target variable for y).

Since different features have various ranges we standardize X data. The method of standard scalar standardize features by removing the mean and scaling to unit variance. We define the `train_test_split()` function to randomly separate training and test data with the ratio of 80:20, while for the option stratify we consider the defined bins. We apply the Random Forest regression model to train data.

The following line of code creates a DataFrame called X_train4shapely using the training data X_train and assigns column names from the features list.

`X_train4shapely = pd.DataFrame(data=X_train, columns=features)`

In the next line, we create an explainer object using the SHAP library's Explainer class. It takes two arguments: the trained Random Forest model rf_model and the training data DataFrame (X_train4shapely). The explainer_rf object will be used to explain the model's predictions.

`explainer_rf = shap.Explainer(rf_model, X_train4shapely)`

The last line generates the SHAP values for the training data using the explainer object. It calculates the contributions of each feature to the predictions made by the Random Forest model. The resulting shap_values_rf object contains the SHAP values for each data point in X_train4shapely.

`shap_values_rf = explainer_rf(X_train4shapely,check_additivity=False)`

To have a visualization of the results we show the bar, beeswarm, and heatmap plots based on the estimated SHAP values.

 In the end, we also use the Random Forest Built-in Feature Importance `rf.feature_importances_` to illustrate the relative importance of each feature in making predictions in a bar plot and compare it with the results according to SHAP values.



Code 02.3: [Feature Importance (XGBoost regression model)](02.3_Feature_Importance-XGBoost.ipynb)

Goal: This code is utilized to estimate the most important features using the XGBoost regression model according to two different approaches: 1. SHAP values, 2. XGBoost Built-in Feature Importance.

After importing the required libraries, we also import a function named get_clean_dataset() from module utils, then we call the function so we have the input data. We make a hist plot of target values and it is obvious that we face an imbalanced dataset.

Since we have a low number of samples for high damage values we stratify data to make sure we will have a range of all samples for both training and test sets `bins2=[0, 1, 60, 101]`. Indeed, the bins categorize target values into different groups called bins.

The `NumPy.Digitize()` function is used to get the indices of bins to which each of these values belongs in the input array.

We list features and decide to drop highly correlated features where correlation value > 0.99 from X data (so we drop only 5 features) and then split features into X and y sets (all features were considered for X and the target variable for y).

Since different features have various ranges we standardize X data. The method of standard scalar standardize features by removing the mean and scaling to unit variance. We define the train_test_split() function to randomly separate training and test data with the ratio of 80:20, while for the option stratify we consider the defined bins. We apply the Random Forest regression model to train data.

The following line of code creates a DataFrame called X_train4shapely using the training data X_train and assigns column names from the features list.

`X_train4shapely = pd.DataFrame(data=X_train, columns=features)`

In the next line, we create an explainer object using the SHAP library's Explainer class. It takes two arguments: the trained Random Forest model xgb_model and the training data DataFrame (X_train4shapely). The explainer_xgb object will be used to explain the model's predictions.

`explainer_xgb = shap.Explainer(xgb_model, X_train4shapely)`

The last line generates the SHAP values for the training data using the explainer object. It calculates the contributions of each feature to the predictions made by the Random Forest model. The resulting shap_values_xgb object contains the SHAP values for each data point in X_train4shapely.

`shap_values_xgb = explainer_xgb(X_train4shapely)`

To have a visualization of the results we show the bar, beeswarm, and heatmap plots based on the estimated SHAP values.

 In the end, we also use the Random Forest Built-in Feature Importance `xgb.feature_importances_.argsort()` to illustrate the relative importance of each feature in making predictions in a bar plot and compare it with the results according to SHAP values.



Codes <br />
03.1: [Linear Regression](03.1_Stratify_proportion_damage-LinearRegression.ipynb) <br />
03.2: [Random Forest Regression](03.2_Stratify_proportion_damage-RandomForest.ipynb)<br />
03.3: [XGBoost Regression](03.3_Stratify_proportion_damage-XGBoost.ipynb)

Goal: These codes are implemented to estimate the performance of the built models for the average of 20 random shuffle splits using different error metrics (RMSE, MSE, MAE, Average Error).

After importing the required libraries, we also import a function named get_clean_dataset() from module utils, then we call the function so we have the input data. We make a hist plot of target values and it is obvious that we face an imbalanced dataset.

Since we have a low number of samples for high damage values we stratify data to make sure we will have a range of all samples for both training and test sets `bins2=[0, 0.00009, 1, 10, 50, 101]`. Indeed, the bins categorize target values into different groups called bins.

The `NumPy.Digitize()` function is used to get the indices of bins to which each of these values belongs in the input array.

We list features and decide to drop highly correlated features where correlation value > 0.99 from X data (so we drop only 5 features) and then split features into X and y sets (all features were considered for X and the target variable for y). Since different features have various ranges we standardize X data. The method of standard scalar standardize features by removing the mean and scaling to unit variance.

We determine RMSE, MSE, MAE, and AVE (average error) lists for both train and test data to use later for saving the results.

We define a for loop with a range of 20 iterations and specify the `train_test_split()` function in the loop to randomly separate training and test data with the ratio of 80:20 in each iteration, while for the option stratify we consider the defined bins. We implement the model and fit it to train data.

Subsequently, we perform ordinary least squares (OLS) regression analysis.

`X2 = sm.add_constant(X_train)`<br />
    `est = sm.OLS(y_train, X2)`<br />
    `est2 = est.fit()`<br />
    `print(est2.summary())`

The above-mentioned code performs OLS regression by fitting a model to the training data and then printing a summary of the regression results.

Note: The `add_constant()` function from the sm module is used to add a column of ones to the beginning of X_train and later X_test `(X2_test = sm.add_constant(X_test))`, ensuring that the model estimates a constant term. The constant term ensures that the model considers the baseline value captured by the intercept term when predicting the values of the dependent variable based on the independent variables in X_train and X_test.

In the final section of the loop, we make predictions on both training and test data to calculate and print the RMSE, MSE, MAE, and Average Error calculated in each run, and append them to the lists.

Now we have a list of 20 numbers for each evaluation metric and we can simply estimate the mean and standard deviation of each list and show their plots.



Codes <br />
04.1: [Performance of Random Forest per bin](04.1_RandomForest-bins.ipynb)<br />
04.2: [Performance of XGBoost per bin](04.2_XGBoost-bins.ipynb)

Goal: These two codes are implemented to estimate the performance of the built models per bin for the average of 20 random shuffle splits using RMSE.

After importing the required libraries, we also import a function named `get_clean_dataset()` from module utils, then we call the function so we have the input data. We make a hist plot of target values and it is obvious that we face an imbalanced dataset.

Since we have a low number of samples for high damage values we stratify data to make sure we will have a range of all samples for both training and test sets `bins2=[0, 1, 60, 101]`. Indeed, the bins categorize target values into different groups called bins.

The `NumPy.Digitize()` function is used to get the indices of bins to which each of these values belongs in the input array.

Here we define some empty lists for both test and train data to later save results per bin. The note is that if you use `bins2=[0, 1, 60, 101]` then because we will categorize data into three groups, we need to define three lists while if we use `bins2 =[0, 0.00009, 1, 10, 50, 101]` we will need five lists.

We list features and decide to drop highly correlated features where correlation value > 0.99 from X data (so we drop only 5 features) and then split features into X and y sets (all features were considered for X and the target variable for y). Since different features have various ranges we standardize X data. The method of standard scalar standardize features by removing the mean and scaling to unit variance.

We define a for loop with a range of 20 iterations and specify the `train_test_split()` function in the loop to randomly separate training and test data with the ratio of 80:20 in each iteration, while for the option stratify we consider the defined bins. We implement the model and fit it to train data.

Subsequently, we perform ordinary least squares (OLS) regression analysis.

`X2 = sm.add_constant(X_train)`<br />
    `est = sm.OLS(y_train, X2)`<br />
    `est2 = est.fit()`<br />
    `print(est2.summary())`

The above-mentioned code performs OLS regression by fitting a model to the training data and then printing a summary of the regression results.

Note: The `add_constant()` function from the sm module is used to add a column of ones to the beginning of X_train and later X_test `(X2_test = sm.add_constant(X_test))`, ensuring that the model estimates a constant term. The constant term ensures that the model considers the baseline value captured by the intercept term when predicting the values of the dependent variable based on the independent variables in X_train and X_test.

In the final section of the loop, we make predictions on both training and test data to calculate the RMSE per bin in each run and append them to the lists.

Now we have a list of 20 numbers (RMSEs) for each bin and we can simply estimate the mean and standard deviation of each list and show their plots.



Codes <br />
05.1: [Random Forest performance per bin(damage>10)](05.1_RandomForest-percent-damage.ipynb)
05.2: [XGBoost performance per bin(damage>10)](05.2_XGBoost-percent-damage.ipynb)

Goal: These codes are implemented to allow the built models trained only with damage >10%, and check the model’s performance only on high bins (per bin).

After importing the required libraries, we also import a function named `get_clean_dataset()` from module utils, then we call the function so we have the input data. We filter the input data frame df to retain only the rows where the 'damage' value is greater than 10, discarding the rest.

`df = df[df['DAM_perc_dmg'] > 10]`

Since we have a low number of samples for high damage values we stratify data to make sure we will have a range of all samples for both training and test sets `bins2=[10.0009, 20, 50, 101]`. However, as you can see bins2 will start from the values>10, due to the fact that we aim to categorize high values of the target column into different groups.

The `NumPy.Digitize()` function is used to get the indices of bins to which each of these values belongs in the input array.  Then we define three empty lists for training and subsequently test data to later save results per bin.

We list features and decide to drop highly correlated features where correlation value > 0.99 from X data (so we drop only 5 features) and then split features into X and y sets (all features were considered for X and the target variable for y). Since different features have various ranges we standardize X data. The method of standard scalar standardize features by removing the mean and scaling to unit variance.

We define a for loop with a range of 20 iterations and specify the train_test_split() function in the loop to randomly separate training and test data with the ratio of 80:20 in each iteration, while for the option stratify we consider the defined bins. We implement the model and fit it to train data.

Note: As you can see we add some new hyper-parameters or change the values of previous ones to build a reduced overfitted model which could decrease the test RMSE and the difference between test and train RMSEs.

Subsequently, we perform ordinary least squares (OLS) regression analysis.

`X2 = sm.add_constant(X_train)`<br />
    `est = sm.OLS(y_train, X2)`<br />
    `est2 = est.fit()`<br />
    `print(est2.summary())`

The above-mentioned code performs OLS regression by fitting a model to the training data and then printing a summary of the regression results.

Note: The `add_constant()` function from the sm module is used to add a column of ones to the beginning of X_train and later X_test `(X2_test = sm.add_constant(X_test))`, ensuring that the model estimates a constant term. The constant term ensures that the model considers the baseline value captured by the intercept term when predicting the values of the dependent variable based on the independent variables in X_train and X_test.

In the final section of the loop, we make predictions on both training and test data to calculate and print the RMSE per bin in each run and append them to the lists.

Now we have a list of 20 numbers (RMSEs) for each bin and we can simply estimate the mean and standard deviation of each list and show their plots.



Codes <br />
06.1: [Random Forest performance in the average of all bins(damage>10)](06.1_RandomForest-wholedataset.ipynb)<br />
06.2: [XGBoost performance in the average of all bins (damage>10)](06.2_XGBoost-wholedataset.ipynb)

Goal: These codes are implemented to allow the built models trained only with damage >10%, and check the model’s performance only on high bins (average of all bins).

After importing the required libraries, we also import a function named get_clean_dataset() from module utils, then we call the function so we have the input data. We filter the input data frame df to retain only the rows where the 'damage' value is greater than 10, discarding the rest.

`df = df[df['DAM_perc_dmg'] > 10]`

Since we have a low number of samples for high damage values we stratify data to make sure we will have a range of all samples for both training and test sets `bins2=[10.0009, 20, 50, 101]`. However, as you can see bins2 will start from the values>10, due to the fact that we aim to categorize high values of the target column into different groups.

The `NumPy.Digitize()` function is used to get the indices of bins to which each of these values belongs in the input array.  Then we define two empty lists for training and subsequently test data to later save the results of total RMSEs.

We list features and decide to drop highly correlated features where correlation value > 0.99 from X data (so we drop only 5 features) and then split features into X and y sets (all features were considered for X and the target variable for y). Since different features have various ranges we standardize X data. The method of standard scalar standardize features by removing the mean and scaling to unit variance.

We define a for loop with a range of 20 iterations and specify the `train_test_split()` function in the loop to randomly separate training and test data with the ratio of 80:20 in each iteration, while for the option stratify we consider the defined bins. We implement the model and fit it to train data.

Note: As you can see we add some new hyper-parameters or change the values of previous ones to build a reduced overfitted model which could decrease the test RMSE and the difference between test and train RMSEs.

Subsequently, we perform ordinary least squares (OLS) regression analysis.

`X2 = sm.add_constant(X_train)`<br />
    `est = sm.OLS(y_train, X2)`<br />
    `est2 = est.fit()`<br />
    `print(est2.summary())`

The above-mentioned code performs OLS regression by fitting a model to the training data and then printing a summary of the regression results.

Note: The `add_constant()` function from the sm module is used to add a column of ones to the beginning of X_train and later X_test `(X2_test = sm.add_constant(X_test))`, ensuring that the model estimates a constant term. The constant term ensures that the model considers the baseline value captured by the intercept term when predicting the values of the dependent variable based on the independent variables in X_train and X_test.

In the final section of the loop, we make predictions on both training and test data to calculate and print the total RMSEs of test and train in each run and append them to the lists.

Now we have two lists each one includes 20 numbers (RMSEs) and we can simply estimate the mean and standard deviation for each list and show the results in a bar plot.



Codes <br />
07.1: [(True vs Prediction Error) for Random Forest](07.1_RandomForest-predicted-and-true.ipynb)<br />
07.2: [(True vs Prediction Error) for XGBoost](07.2_XGBoost-predicted-and-true.ipynb)

Goal: These codes are implemented to estimate the real vs prediction Error for a single run in Random Forest and XGBoost Regression models and visualize their difference using a scatter plot.

After importing the required libraries, we also import a function named `get_clean_dataset()` from module utils, then we call the function so we have the input data. We can even simply filter the input data frame df to retain only the rows where the 'damage' value is greater than 10, discarding the rest.

`df = df[df['DAM_perc_dmg'] > 10]`

Since we have a low number of samples for high damage values we stratify data to make sure we will have a range of all samples for both training and test sets. We can choose one of the following bin sets:

`bins2= [0, 1, 60, 101]`<br />
`bins2 = [0, 0.00009, 1, 10, 50, 101]`

We can even use `bins2=[10.0009, 20, 50, 101]` that starts from the values>10, due to the fact that we aim to categorize high values of the target column into different groups.

The `NumPy.Digitize()` function is used to get the indices of bins to which each of these values belongs in the input array.

We list features and decide to drop highly correlated features where correlation value > 0.99 from X data (so we drop only 5 features) and then split features into X and y sets (all features were considered for X and the target variable for y). Since different features have various ranges we standardize X data. The method of standard scalar standardize features by removing the mean and scaling to unit variance.

We define the `train_test_split()` function to randomly separate training and test data with the ratio of 80:20, while for the option stratify we consider the defined bins. We apply the built model to train data.

Subsequently, we perform ordinary least squares (OLS) regression analysis.

`X2 = sm.add_constant(X_train)`<br />
    `est = sm.OLS(y_train, X2)`<br />
    `est2 = est.fit()`<br />
    `print(est2.summary())`

The above-mentioned code performs OLS regression by fitting a model to the training data and then printing a summary of the regression results.

Note: The `add_constant()` function from the sm module is used to add a column of ones to the beginning of X_train and later X_test `(X2_test = sm.add_constant(X_test))`, ensuring that the model estimates a constant term. The constant term ensures that the model considers the baseline value captured by the intercept term when predicting the values of the dependent variable based on the independent variables in X_train and X_test.

In the final section, we make predictions on both training and test data to calculate and print the RMSE, MSE, MAE, Max Error, and Average Error.

Now we estimate the prediction error `abs(y_pred - y_test)` and show a plot of True vs absolute values of prediction error.



Code 08: [Random Forest & XGBoost Typhoon Split](08_Typhoon_train-test-split-RandomForest-and-XGBoost-bins.ipynb)

Goal: This code is implemented to see how the model performs when we have train and test split based on typhoons (with a ratio of 80:20).

After importing the required libraries, we also import a function named `get_clean_dataset()` from module utils, then we call the function so we have the input data.

Since we have a low number of samples for high damage values we stratify data to make sure we will have a range of all samples for both training and test sets bins2 = [0, 0.00009, 1, 10, 50, 101]`. Indeed, the bins categorize target values into different groups called bins.

We separate the typhoons column and target columns from the input data frame and apply the standard scalar method to standardize the values of features by removing the mean and scaling to unit variance. Then we again add target values to the data frame.

Note: Since the target values are damage percentages with a range of 0 to 100 and we want to keep this range for prediction we only want to standardize the X data.

We need to add the columns' headers after standardization and insert all typhoons' names into a list named typhoons_lst.

In the next step, we check the mean damage value for each typhoon to figure out the most severe ones. According to these mean values we choose a balanced list in terms of typhoon severity including 8 typhoons while half of them are severe half of them are with low mean values. We consider this list as our test set and divide the data frame into test and train data.

Then we can choose a model between Random Forest and XGBoost and apply it to train data.

Subsequently, we perform ordinary least squares (OLS) regression analysis.

`X2 = sm.add_constant(X_train)`<br />
    `est = sm.OLS(y_train, X2)`<br />
    `est2 = est.fit()`<br />
    `print(est2.summary())`

The above-mentioned code performs OLS regression by fitting a model to the training data and then printing a summary of the regression results.

Note: The `add_constant()` function from the sm module is used to add a column of ones to the beginning of X_train and later X_test `(X2_test = sm.add_constant(X_test))`, ensuring that the model estimates a constant term. The constant term ensures that the model considers the baseline value captured by the intercept term when predicting the values of the dependent variable based on the independent variables in X_train and X_test.

Finally, we make predictions on both training and test data to calculate and print the RMSEs per bin. We even estimate the model performance based on different evaluation metrics.

Note: We can choose another group of typhoons as the test list so the result will change.



Codes <br />
09.1: [Random Forest Typhoon’s Time Split](09.1_Typhoons_by_time-RandomForest-main.ipynb)<br />
09.2: [XGBoost Typhoon’s Time Split](09.2_Typhoons_by_time-XGBoost-main.ipynb)

Goal: This code is implemented to do a train and test split based on the typhoon’s time (80:20). The idea is to determine how well the model performs in learning from older typhoons' characteristics to make predictions on the target value of the most recent ones.

After importing the required libraries, we also import a function named `get_clean_dataset()` from module utils, then we call the function so we have the input data.

Since we have a low number of samples for high damage values we stratify data to make sure we will have a range of all samples for both training and test sets bins2 = [0, 0.00009, 1, 10, 50, 101]`. Indeed, the bins categorize target values into different groups called bins.

We separate the typhoons column and target columns from the input data frame and apply the standard scalar method to standardize the values of features by removing the mean and scaling to unit variance. Then we again add target values to the data frame.

Note: Since the target values are damage percentages with a range of 0 to 100 and we want to keep this range for prediction we only want to standardize the X data.

We need to add the columns' headers after standardization and insert all typhoons' names into a list named typhoons_lst.

We separate the year of the typhoon from each one and put them in ascending order.

We determine two empty lists to save the total RMSEs of test and training data and five more empty lists to append the estimated RMSEs for each bin to each list.

We make a test list with the 8 recent typhoons in terms of time and then we keep the rest as the train list so we can divide the data frame into test and train data. Now, we can apply the built model to train data.

Note: Considering that we need to run the model 8 times and in every run one of the less recent typhoons will remove from the test set and add to the train set (the length of the train set is not fixed and it increases)

Finally, calculate the RMSEs per bin and estimate the different evaluation metrics in each run and print the results.

For a better comparison of the model performance based on each typhoon in the test list, we can use the lists that we save the RMSEs and make a plot of RMSE per typhoon in total and per bin (typhoons vs model performance).
