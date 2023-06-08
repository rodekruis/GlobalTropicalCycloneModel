## Documentation of all codes in this branch

Code 02: <br />
[Naive baseline model](02_model_training-baselines.ipynb)

Goal: This code is utilized to train a simple baseline model on the input data and estimate the performance of this model. 

In this code, after importing the required libraries, we also import a module named utils that includes some functions, particularly get_training_dataset() which is the one we need to read input data stored as the CSV file and insert it in a data frame.  

The percentage of damage as our target in the dataset, known as "percent_buildings_damaged", we make a hist plot to check its distribution. Since we have an imbalance of dataset we stratify data to make sure we will have the lower samples for both training and test sets. 

bins2=[0, 0.00009, 1, 10, 50, 101]

There are some rows in the dataset in which windspeed equals zero, we decided to remove those rows before we train the model on the input data. We also check the hist plot after removing these rows just to see any differences.

The NumPy Digitize() function is used to get the indices of bins to which each of these values belongs in the input array.

We split features into X and y sets and standardize only X data since different features have various ranges. The method of standard scalar standardize features by removing the mean and scaling to unit variance. 

We define the train_test_split() function to randomly separate training and test data with the ratio of 80:20, while for the option stratify we consider the defined bins.

We define a function named create_dummy(), the input of the function are test and train data (X and y), and the defined bins, we create a dummy_regressor by using 'mean' as the strategy and fit it to train data. So basically this model returns the average of the target from training data.

Finally, we predict both train and test data totally and per bin.



Code 03: XGBoost regression model (building counts)

Goal: This code is utilized to train the build model on the input data and estimate the model performance. 

In this code, after importing the required libraries, we also import a module named utils that includes some functions, particularly get_training_dataset() which is the one we need to read input data stored as the CSV file and insert it in a data frame.  

The percentage of damage as our target in the dataset, known as "percent_buildings_damaged", we make a hist plot to check its distribution. Since we have an imbalance of dataset we stratify data to make sure we will have the lower samples for both training and test sets. 

bins2=[0, 0.00009, 1, 10, 50, 101]

The value_counts() function checks the bins' intervals, so we can figure out how many data points we have in each bin.

There are some rows in the dataset in which windspeed equals zero, we decided to remove those rows before we train the model on the input data. We also check the hist plot after removing these rows just to see any differences.

The NumPy Digitize() function is used to get the indices of bins to which each of these values belongs in the input array.

We define two empty lists to save the total and per bin RMSE, later.

We split features into X and y sets (all features were considered for X and the target variable for y).    Since different features have various ranges we standardize X data. The method of standard scalar standardize features by removing the mean and scaling to unit variance. 

We define the train_test_split() function to randomly separate training and test data with the ratio of 80:20, while for the option stratify we consider the defined bins.

We specify the XGBoost model (with all the specified hyperparameters) and then fit the model to train data.

Subsequently, we perform ordinary least squares (OLS) regression analysis. 

X2 = sm.add_constant(X_train)
    est = sm.OLS(y_train, X2)
    est2 = est.fit()
    print(est2.summary())

The above-mentioned code performs OLS regression by fitting a model to the training data and then printing a summary of the regression results.

Note: The add_constant function from the sm module is used to add a column of ones to the beginning of X_train and later X_test (X2_test = sm.add_constant(X_test)), ensuring that the model estimates a constant term. The constant term ensures that the model considers the baseline value captured by the intercept term when predicting the values of the dependent variable based on the independent variables in X_train and X_test.

Finally, we make predictions on both test and train data wrt the XGBoost model and calculate the total RMSE per bin using a for loop in range(1, 6).

At the very end to get a plot of RMSEs we define a function named rmse_bin_plot() to receive the list of inputs including a list of RMSE for test and train plus min, max ranges, and the step. We can estimate the mean and standard deviation of the RMSE lists in this function and then the output will show you a bar plot of RMSE for test and train data including the dashed line of the test and train mean.



Code 03: XGBoost regression model (houses counts)

Goal: This code is utilized to train the build model on the input data and estimate the model performance. 

Generally, code 03 was updated to train the model on input data using the ‘total_houses' instead of ‘total_building’ since there are distinct concepts for buildings and houses. Consequently, the target variable ‘percent_damage_building’ was replaced with 'percent_damage_houses’ to accurately represents the percentage of damaged houses. 

Besides the fact that all this code’s steps are the same as code03, we have more features. 

Relative Wealth Index ("rwi") as one of the new features includes some NaN values for some grid cells. Therefore, after importing the input data into a data frame we decide to fill those rows with the mean value of "rwi". 

Due to the replacement of the building with houses, some values in the target dataset have exceeded 100. To ensure that all values fall within the range of 0 to 100, we have set those numbers to a maximum of 100.



Code 04: Correlation Matrix

Goal: This code is utilized to compute the correlation among various features present in the input dataset.

In this code, after importing the required libraries, we also read a CSV file (obtained from previous code (code 03) after cleaning data related to 'rwi' and target values) and import it into a data frame.  

Then we remove rows in the dataset in which windspeed equals zero, and we drop "grid_point_id" and "typhoon_year".

We estimate the Pearson correlation coefficients of features and plot the correlation matrix. We also ignore the sign to provide a measure of their overall importance or contribution to the model which could be useful for feature selection, so we take the absolute value of the coefficients. 

corrMatrix_abs = df.corr().abs()

To figure out the highly correlated features we consider 80 as the threshold to print out pairs of highly correlated features if exist.

At the very end, we create a scatter plot to show the relation between ‘total_houses’ and ‘rwi’, colored by damage data (percent_houses_damaged), for all damage values and for those values greater than 10 which we know as the high damage values.



Code 05: Simple Moving Average(SMA)

Goal: To better understand the relationship between some of the variables together or with the target.

In this code, we import the required libraries and read the CSV file as our input data to insert it in a data frame, then we remove windspeed == 0. 

We define a function named ave(ftr_1, ftr_2) to estimate and plot the relationship between two selected features from the list of features. So the inputs for the functions are two features and the output is a plot to show their correlation. To find the relationship we use the simple moving average that helps to identify trends or patterns by reducing noise and random variations in the data.

SMA is the most common type of moving average, which is calculated by taking the average of a fixed number of previous data points. Here is the formula:

SMA = (Sum of values in the window) / (Number of data points in the window)

The result of the following code that we have in the ave() function, is a variable containing the rolling mean values of feature2 calculated using a window size of 500. The length of this variable is the same as the length of our data frame, as each value in the rolling window is associated with a specific position in the original data.

df.sort_values(feature1).reset_index(drop=True)[feature2].rolling(window=500).mean()

The last part of this section is a piece of code that allows us to plot the trend line between rainfall_max_6h and rainfall_max_24h.



Code 07: Feature Importance (XGBoost regression model)

Goal: This code is utilized to find which features are most important in predicting the target variable.

In this code, after importing the required libraries, we also read a CSV file (obtained from previous code (code 03) after cleaning data related to 'rwi' and target values) and import it into a data frame. 

The percentage of damage as our target in the dataset, known as "percent_buildings_damaged", we make a hist plot to check its distribution. Since we have an imbalance of dataset we stratify data to make sure we will have the lower samples for both training and test sets. 

bins2=[0, 0.00009, 1, 10, 50, 101]

There are some rows in the dataset in which windspeed equals zero, we decided to remove those rows before we train the model on the input data. We also check the hist plot after removing these rows just to see any differences.

The NumPy Digitize() function is used to get the indices of bins to which each of these values belongs in the input array.

We split features into X and y sets (all features were considered for X and the target variable for y).    Since different features have various ranges we standardize X data. The method of standard scalar standardize features by removing the mean and scaling to unit variance. 

We define the train_test_split() function to randomly separate training and test data with the ratio of 80:20, while for the option stratify we consider the defined bins. We apply the XGBoost model to train data. 

The following line of code creates a DataFrame called X_train4shapely using the training data X_train and assigns column names from the features list.

X_train4shapely = pd.DataFrame(data=X_train, columns=features)

In the next line, we create an explainer object using the SHAP library's Explainer class. It takes two arguments: the trained XGBoost model xgb_model and the training data DataFrame (X_train4shapely). The explainer_xgb object will be used to explain the model's predictions.

explainer_xgb = shap.Explainer(xgb_model, X_train4shapely)

The last line generates the SHAP values for the training data using the explainer object. It calculates the contributions of each feature to the predictions made by the XGBoost model. The resulting shap_values_xgb object contains the SHAP values for each data point in X_train4shapely.

shap_values_xgb = explainer_xgb(X_train4shapely)

To have a visualization of the results we show the bar and beeswarm plot based on the estimated SHAP values. In the end, we also use the XGBoost Built-in Feature Importance to illustrate the relative importance of each feature in making predictions in a bar plot and compare it with the results according to SHAP values.



Code 08: Creating a historical variable

Goal: This code is utilized to generate a historical variable and incorporate it into the data to improve the performance of the built model when training on the input data.

In this code, after importing the required libraries, we also import a module named utils that includes some functions, particularly get_training_dataset_primary() which is the one we need to call.  

Basically, wegroupby() based on "typhoon_year", "grid_point_id", and "percent_houses_damaged". Then, we calculate an average of damages over the past 5 years for every data point and recorded it as the value for the new variable ("percent_houses_damaged_5years"). 

For a more clear description of the piece of code below, we groupby() on df_avgDmgCell_and_Year based on a specific columngrid_point_id. We compute the rolling mean of the “percent_houses_damaged" column and the maximum value of the "typhoon_year" column within a rolling window of size 5 (representing 5 years).

df_res2 = (
    df_avgDmgCell_and_Year.groupby("grid_point_id")
    .rolling(5, min_periods=1)
    .agg({"percent_houses_damaged": "mean", "typhoon_year": "max"}))

Note: min_periods is set to 1, which means that the rolling window requires at least one valid observation in the window before computing the result.

Finally, we merge this DataFrame with the original one based on ["typhoon_year", "grid_point_id"] and fill missing values in the new column of a merged DataFrame with the value 0.



Code 09: Binary Model (Random Forest)

Goal: This code is implemented a classification algorithm (Random Forest) to construct a model and train it on binary input data.

In this code, after importing the required libraries, we also import a module named utils that includes some functions, particularly get_training_dataset() which is the one we need to read input data stored as the CSV file and insert it in a data frame. 

Relative Wealth Index ("rwi") as one of the features includes some NaN values for some grid cells. Therefore, we decide to fill those rows with the mean value of "rwi", and since some values in the target dataset have exceeded 100 we set those numbers to the maximum value (100) to ensure that all values fall within the range of 0 to 100.

In the next step, to generate a two-class binary, we define a threshold to separate the continuous values of the target into damaged (1) and not-damaged (0) classes. 10% is considered a reasonable house damage threshold to convert the continuous target into a binary one. Therefore, all values lower than this threshold will belong to class 0, and the ones greater than the threshold will consider class 1.

Now, we have a data frame with a new column named "binary_damage" which is the target variable for the binary model.

Then as in the same steps in other codes, we remove rows in the dataset in which windspeed equals zero, and we drop "grid_point_id" and "typhoon_year".

We used sns.countplot() which is a function to plot the count of observations in each category and also we provide a hist plot.

At this step, we defined a new set of bins bins2 = [0, 0.1, 1] that is proper for the binary model and counted the number of samples per bin. df["binary_damage"].value_counts(bins=binsP2)

We use NumPy Digitize() function to get the indices of bins to which each of these values belongs in the input array. Before building the binary model we describe the features and split the data into X and y.

The function train_test_split() allows us to split our dataset into two subsets for training and testing the built machine-learning model (with a ratio of 80:20). 

We use the Counter() function to check the number of samples in each class for training data. Since we have an imbalanced dataset, we can perform over or under-sampling or even a combination of both to provide a more balanced training dataset. The most proper sampling strategy is defined for each resampling strategy based on test and trial methods. 

After resampling, we define the Random Forest classification model and fit it to train data, then make predictions on test data. There are different ways to check the performance of binary models, like Confusion Matrix (TP, FP, TN, FN) and classification report (precision, recall, f1-score).

At the very end, we also check the importance of features in the Random Forest classification model using the feature_importances_ attribute that fits the model and represents the importance of each feature in the trained model.



Code 09: Binary Model (XGBoost)

Goal: This code is implemented a classification algorithm (XGBoost) to construct a model and train it on binary input data.

In this code, after importing the required libraries, we also import a module named utils that includes some functions, particularly get_training_dataset() which is the one we need to read input data stored as the CSV file and insert it in a data frame. 

Relative Wealth Index ("rwi") as one of the features includes some NaN values for some grid cells. Therefore, we decide to fill those rows with the mean value of "rwi", and since some values in the target dataset have exceeded 100 we set those numbers to the maximum value (100) to ensure that all values fall within the range of 0 to 100.

In the next step, to generate a two-class binary, we define a threshold to separate the continuous values of the target into damaged (1) and not-damaged (0) classes. 10% is considered a reasonable house damage threshold to convert the continuous target into a binary one. Therefore, all values lower than this threshold will belong to class 0, and the ones greater than the threshold will consider class 1.

Now, we have a data frame with a new column named "binary_damage" which is the target variable for the binary model.

Then as in the same steps in other codes, we remove rows in the dataset in which windspeed equals zero, and we drop "grid_point_id" and "typhoon_year".

We used seaborn.countplot() which is a function to plot the count of observations in each category and also we provide a hist plot.

At this step, we defined a new set of bin bins2 = [0, 0.1, 1] that is proper for the binary model and counted the number of samples per bin. df["binary_damage"].value_counts(bins=binsP2)

We use NumPy Digitize() function to get the indices of bins to which each of these values belongs in the input array. Before building the binary model we describe the features and split the data into X and y.

The function train_test_split() allows us to split our dataset into two subsets for training and testing the built machine-learning model (with a ratio of 80:20). 

We use the Counter() function to check the number of samples in each class for training data. Since we have an imbalanced dataset, we can perform over or under-sampling or even a combination of both to provide a more balanced training dataset. The most proper sampling strategy is defined for each resampling strategy based on test and trial methods. 

After resampling, we define the XGBoost classification model and fit it to train data, then make predictions on test data. There are different ways to check the performance of binary models, like the Confusion Matrix (TP, FP, TN, FN) and classification report (precision, recall, f1-score), while in this model, we also plot the Log Loss and Classification Error.

At the very end, we also check the importance of features in the XGBoost classification model using the feature_importances_ attribute that fits the model and represents the importance of each feature in the trained model.



Code 09: Binary Logistic Regression Model

Goal: This code is implemented a classification algorithm (Logistic Regression) to construct a model and train it on binary input data.

In this code, after importing the required libraries, we also import a module named utils that includes some functions, particularly get_training_dataset() which is the one we need to read input data stored as the CSV file and insert it in a data frame. 

Relative Wealth Index ("rwi") as one of the features includes some NaN values for some grid cells. Therefore, we decide to fill those rows with the mean value of "rwi", and since some values in the target dataset have exceeded 100 we set those numbers to the maximum value (100) to ensure that all values fall within the range of 0 to 100.

In the next step, to generate a two-class binary, we define a threshold to separate the continuous values of the target into damaged (1) and not-damaged (0) classes. 10% is considered a reasonable house damage threshold to convert the continuous target into a binary one. Therefore, all values lower than this threshold will belong to class 0, and the ones greater than the threshold will consider class 1.

Now, we have a data frame with a new column named "binary_damage" which is the target variable for the binary model.

Then as in the same steps in other codes, we remove rows in the dataset in which windspeed equals zero, and we drop "grid_point_id" and "typhoon_year".

We used seaborn.countplot() which is a function to plot the count of observations in each category and also we provide a hist plot.

At this step, we defined a new set of bin bins2 = [0, 0.1, 1] that is proper for the binary model and counted the number of samples per bin. df["binary_damage"].value_counts(bins=binsP2)

We use NumPy Digitize() function to get the indices of bins to which each of these values belongs in the input array. Before building the binary model we describe the features (we remove highly correlated features for this model) and split the data into X and y.

The function train_test_split() allows us to split our dataset into two subsets for training and testing the built machine-learning model (with a ratio of 80:20). 

We use the Counter() function to check the number of samples in each class for training data. Since we have an imbalanced dataset, we can perform over or under-sampling or even a combination of both to provide a more balanced training dataset. The most proper sampling strategy is defined for each resampling strategy based on test and trial methods. 

After resampling, we define the Binary Logistic Regression model and fit it to train data, then make predictions on test data. There are different ways to check the performance of binary models, like the Confusion Matrix (TP, FP, TN, FN) and classification report (precision, recall, f1-score), while in this model, we also plot the Log Loss and Classification Error.

At the very end, we also check the importance of features in the Binary Logistic Regression using the coefficient values (coef_ attribute) that fit the model and represent the importance of each feature in the trained model. We sort the data frame of coefficient values in descending order.



Code 09: XGBoost (different n_estimators)

Goal: This code is implemented to compare the performance of the XGBoost classification model with respect to different n-estimators.

In this code, after importing the required libraries, we also import a module named utils that includes some functions, particularly get_training_dataset() which is the one we need to read input data stored as the CSV file and insert it in a data frame. 

Relative Wealth Index ("rwi") as one of the features includes some NaN values for some grid cells. Therefore, we decide to fill those rows with the mean value of "rwi", and since some values in the target dataset have exceeded 100 we set those numbers to the maximum value (100) to ensure that all values fall within the range of 0 to 100.

In the next step, to generate a two-class binary, we define a threshold to separate the continuous values of the target into damaged (1) and not-damaged (0) classes. 10% is considered a reasonable house damage threshold to convert the continuous target into a binary one. Therefore, all values lower than this threshold will belong to class 0, and the ones greater than the threshold will consider class 1.

Now, we have a data frame with a new column named "binary_damage" which is the target variable for the binary model.

Then as in the same steps in other codes, we remove rows in the dataset in which windspeed equals zero, and we drop "grid_point_id" and "typhoon_year".

We used seaborn.countplot() which is a function to plot the count of observations in each category and also we provide a hist plot.

At this step, we defined a new set of bin bins2 = [0, 0.1, 1] that is proper for the binary model and counted the number of samples per bin. df["binary_damage"].value_counts(bins=binsP2)

We use NumPy Digitize() function to get the indices of bins to which each of these values belongs in the input array. Before building the binary model we describe the features and split the data into X and y.

The function train_test_split() allows us to split our dataset into two subsets for training and testing the built machine-learning model (with a ratio of 80:20). 

We use the Counter() function to check the number of samples in each class for training data. Since we have an imbalanced dataset, we can perform over or under-sampling or even a combination of both to provide a more balanced training dataset. The most proper sampling strategy is defined for each resampling strategy based on test and trial methods. 

We define an empty list to keep the f1_score of each n_estimator f1_lst = [], and a list of 10 different n_estimator from 10 to 100 n_estimator_lst = [12, 22, 32, 42, 52, 62, 72, 82, 92, 102]. Afterward, we define a for loop with the range of len(n_estimator_lst)and use the XGBClassifier to fit the data. So in each iteration, we have a new n_estimator to estimate F1_score and save it in its list. Finally,  we create a plot to compare n_estimator vs F1_score.



Code 10: Feature Importance (XGBoost classification model)

Goal: This code is implemented to find which features are most important in predicting the binary target variable.

In this code, after importing the required libraries, we also import a module named utils that includes some functions, particularly get_training_dataset() which is the one we need to read input data stored as the CSV file and insert it in a data frame. 

Relative Wealth Index ("rwi") as one of the features includes some NaN values for some grid cells. Therefore, we decide to fill those rows with the mean value of "rwi", and since some values in the target dataset have exceeded 100 we set those numbers to the maximum value (100) to ensure that all values fall within the range of 0 to 100.

In the next step, to generate a two-class binary, we define a threshold to separate the continuous values of the target into damaged (1) and not-damaged (0) classes. 10% is considered a reasonable house damage threshold to convert the continuous target into a binary one. Therefore, all values lower than this threshold will belong to class 0, and the ones greater than the threshold will consider class 1.

Now, we have a data frame with a new column named "binary_damage" which is the target variable for the binary model. We used seaborn.countplot() which is a function to plot the count of observations in each category.

Then as in the same steps in other codes, we remove rows in the dataset in which windspeed equals zero, and we drop "grid_point_id" and "typhoon_year".

 We provide a hist plot and define a new set of bin bins2 = [0, 0.1, 1] that is proper for the binary model and counted the number of samples per bin. df["binary_damage"].value_counts(bins=binsP2).

We use NumPy Digitize() function to get the indices of bins to which each of these values belongs in the input array. Before building the binary model we describe the features and split the data into X and y.

The function train_test_split() allows us to split our dataset into two subsets for training and testing the built machine-learning model (with a ratio of 80:20). 

We use XGBClassifier as a Machine Learning model to fit the data and make predictions on test data then by using the feature_importances_ attribute check the most important features in the data set.

The following line of code creates a DataFrame called X_train4shapely using the training data X_train and assigns column names from the features list.

X_train4shapely = pd.DataFrame(data=X_train, columns=features)

In the next line, we create an explainer object using the SHAP library's Explainer class. It takes two arguments: the trained XGBoost model xgb_model and the training data DataFrame (X_train4shapely). The explainer_xgb object will be used to explain the model's predictions.

explainer_xgb = shap.Explainer(xgb_model, X_train4shapely)

The last line generates the SHAP values for the training data using the explainer object. It calculates the contributions of each feature to the predictions made by the XGBoost model. The resulting shap_values_xgb object contains the SHAP values for each data point in X_train4shapely.

shap_values_xgb = explainer_xgb(X_train4shapely)

To have a visualization of the results we show the bar and beeswarm plot based on the estimated SHAP values. 



Code 11: ROC Curve for binary models

Goal: This code is utilized to have a graphical representation of the performance of the three binary models and compare their performance using the ROC curve.

In this code, after importing the required libraries, we also import a module named utils that includes some functions, particularly get_training_dataset() which is the one we need to read input data stored as the CSV file and insert it in a data frame. 

Relative Wealth Index ("rwi") as one of the features includes some NaN values for some grid cells. Therefore, we decide to fill those rows with the mean value of "rwi", and since some values in the target dataset have exceeded 100 we set those numbers to the maximum value (100) to ensure that all values fall within the range of 0 to 100.

In the next step, to generate a two-class binary, we define a threshold to separate the continuous values of the target into damaged (1) and not-damaged (0) classes. 10% is considered a reasonable house damage threshold to convert the continuous target into a binary one. Therefore, all values lower than this threshold will belong to class 0, and the ones greater than the threshold will consider class 1.

Now, we have a data frame with a new column named "binary_damage" which is the target variable for the binary model. We used seaborn.countplot() which is a function to plot the count of observations in each category.

Then as in the same steps in other codes, we remove rows in the dataset in which windspeed equals zero, and we drop "grid_point_id" and "typhoon_year".

 We provide a hist plot and define a new set of bin bins2 = [0, 0.1, 1] that is proper for the binary model and counted the number of samples per bin. df["binary_damage"].value_counts(bins=binsP2).

We use NumPy Digitize() function to get the indices of bins to which each of these values belongs in the input array. Before building the binary model we describe the features and split the data into X and y.

The function train_test_split() allows us to split our dataset into two subsets for training and testing the built machine-learning model (with a ratio of 80:20). 

We use the Counter() function to check the number of samples in each class for training data. Since we have an imbalanced dataset, we can perform over or under-sampling or even a combination of both to provide a more balanced training dataset. The most proper sampling strategy is defined for each resampling strategy based on test and trial methods. 

We define the three binary models and for each model fits training data, and use predict_proba(X_test)[:, 1] that predicts probabilities of the positive class for the test data using that trained model.

Finally, we perform an evaluation of all three binary classification models by calculating the AUC-ROC scores and generating the ROC curves based on the true labels and predicted probabilities, and plotting them.

For example in the two following lines of code related to Logistic Regression, first, the code computes the AUC-ROC score (area under the ROC Curve), which is a measure of the model's ability to distinguish between the positive and negative classes and then calculates the ROC curve.  

auc_lr = roc_auc_score(y_test_int, probs_lr)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test_int, probs_lr)

Note: The fpr_lr and tpr_lr arrays contain the false positive rate and true positive rate values at different thresholds, respectively.



Code 12: Data Resampling in Regression Model

Goals: This code is implemented the SMOTE technique to reduce the class imbalance in the continuous target data by oversampling the minority class.

In this code, after importing the required libraries, we also import a module named utils that includes some functions, particularly get_training_dataset() which is the one we need to read input data stored as the CSV file and insert it in a data frame. For simplicity of our work we move the target to be the last column of the data frame.

Relative Wealth Index ("rwi") as one of the features includes some NaN values for some grid cells. Therefore, we decide to fill those rows with the mean value of "rwi", and since some values in the target dataset have exceeded 100 we set those numbers to the maximum value (100) to ensure that all values fall within the range of 0 to 100.

In the next step, we define a threshold == 10% to separate the continuous values of the target into damaged (1) and not-damaged (0) classes. Therefore, all values lower than this threshold will belong to class 0, and the ones greater than the threshold will consider class 1.

Then as in the same steps in other codes, we remove rows in the dataset in which windspeed equals zero, and we drop "grid_point_id" and "typhoon_year".

Now, the last two columns of our data frame are "percent_houses_damaged"and "binary_damage", so we define the set of bin that we basically used for continous target: bins2 = [0, 0.00009, 1, 10, 50, 101], and we check the bin’s intervals for both continous and binary targets.

We use NumPy Digitize() function to get the indices of bins to which each of these values belongs in the input array. Before building the binary model we describe the features and split the data into X and y. 

Note: Generally, we only want to use binary target to resample dataset so after that we can remove it and use continuous target. Hence, we keep continous target in features list and we define binary target as y.  y = df["binary_damage"]

The function train_test_split() allows us to split our dataset into two subsets for training and testing the built machine-learning model (with a ratio of 80:20). 

We use the Counter() function to check the number of samples in each class for training data. Since we have an imbalanced dataset, we apply SMOTE technique to oversample the minority class (class 1) of training data. The result after this method indicates that the number of sample data in both class are the same (the number of data points in minority class increased to be equal number of samples in majority class).

Then we make a hist plot of training and test data and we can compare how the oversampling change the samples in the train data while test data remains the same as past.

This time we define a new list of features while we remove continous target from this new list and keep as the target. We used the oversampled train data as the train data of XGBoost regression model and the previous test data will consider as the test data for regression model while we only replaced continous target with binary one. Now it is time to standardize both train and test data, build the regression model and fit it to training data.

In the end, we make prediction on both test and train data and calculate the total RMSE and per bin for both sets.

Note: We figure out that some predicted values are negatives, we clip the predicted values to be within the range of zero to 100. y_pred_clipped = y_pred.clip(0, 100)



Code 13: RMSE estimation for regions (ADM1)

Goal: This code is utilized to check how the model can perform for a wide area. (region ADM1 instead of Municipality ADM3)

In this code, after importing the required libraries, we also import a module named utils that includes some functions, particularly get_training_dataset() which is the one we need to read input data stored as the CSV file and insert it in a data frame. For simplicity of our work we move the target to be the last column of the data frame.

Relative Wealth Index ("rwi") as one of the features includes some NaN values for some grid cells. Therefore, we decide to fill those rows with the mean value of "rwi", and since some values in the target dataset have exceeded 100 we set those numbers to the maximum value (100) to ensure that all values fall within the range of 0 to 100. We remove rows in the dataset in which windspeed equals zero, and we drop"typhoon_year".

We define a new set of bin bins2 = [0, 0.00009, 1, 10, 50, 101] and counted the number of samples per bin. We use NumPy Digitize() function to get the indices of bins to which each of these values belongs in the input array. Before building the binary model we describe the features and split the data into X and y.

The function train_test_split() allows us to split our dataset into two subsets for training and testing the built machine-learning model (with a ratio of 80:20). 

We use MinMaxScaler() function for data standardization that normalize data in range of [0,1], but earlier we separate the first two columns which are “typhoon_name“ and “grid_point_id“ and the last column which is target.



Code 14: Combined Model

Goal: This code is implemented to build a hybrid model leading to reduce the RMSE estimation of high bins (high damaged values). 



Code 14: Combined Model in a loop

Goal: This code is utilized to define a loop of m iteration to have an average of m RMSE estimations leading to more secure result.

Since we faced some variation in the result of hybrid model for multiple runs, we inserted the whole code of hybrid model in a loop to get an average of RMSE. Therefore, the only difference with the description of the combined model is a for loop. 



Code 15: Combined Model typhoon split (leave-one-out cross-validation)

Goal: This code is implemented to evaluate the performance of the combined model while train and test split is typhoon based (leave-one-out cross-validation). 



Code 15: Combined Model typhoonTime split (walk forward evaluation)

Goal: This code is implemented to evaluate the performance of the combined model while train and test split is typhoon based (walk forward evaluation).
