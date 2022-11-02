# GlobalTropicalCycloneModel
# ReportOfAnalysis_Codes&Steps

All the codes from number 01 to 09 are related to the analysis of the input data, features, and even the two regression models.


Codes number 01

As the first step in code number 01_Main_Correlation_Matrix, the correlation value for each feature was estimated. There are 38 features including the target and the idea was to figure out those pairs with high correlation values based on the correlation matrix approach. Value of 80 was considered as the threshold to express the high correlated pairs.

After removing one feature in each pair, another method known as VIF was applied to the remained features to check if still there are some other highly correlated ones. 

VIF stands for Variance Inflation Factor, and it finds the correlation value of one feature with respect to all other remained ones. The threshold was considered as 7 so features with a value more than the VIF threshold should remove from the list of features in the input data of the model.


Codes number 02

The next step is Feature_Importance estimation, for three regression models.  The three regression models applied to input data are Linear Regression, Random Forest, and XGBoost.
For Linear Regression, all the high correlated features were removed with respect to VIF result and then the most important features were determined by applying regression coefficient. Finally the features were organized in descending order according to their coef values and plotted in bar chart.

For both Random Forest and XGBoost, two different methods were used. One is searching important features by SHAP values which let us display most important features in three various plots: Bar plot, Beeswarm plot and Heatmap.  The other method is applying the algorithm Built-in Feature Importance.
Random Forest algorithm was based on reducing in impurity and XGBoost library provides a built-in function to plot features ordered by their importance.(we used feature_importances_ attribute).

It is good to mention that computing the most important features according to the two above algorithms(Random Forest & XGBoost) was done two times: 1.after removing very highly correlated features(with 0.99 as thresholds) and 2.after removing features based on VIF result.


Codes number 03

The input data includes 8073 samples while 4086 rows have zero values for the target(damage percentage for the houses). 
The point is most of these zeros were missing values in the originally collected data so they were replaced with zeros with respect to the wind field value and rainfall measurement.  
A set of bin was defined to stratify dataset according to target values(target value represent percentage of damage which is between 0 and 100 while 0 means no damage existance and 100 means the highest possible damage).

Some of the error metrics such as RMSE, MSE, MAE, and Average Error selected to calculated model performance in a 20 runs while every run has a different 20/80 test training split of the same size, and with the stratified data. The errors were estimated per each bins. In order to make a comparison of error estimation values a new set of bin also defined.

It is good to highlight that in the whole error estimation in stratified process, highly correlated features(>0.99) was removed from features in dataset.


D. Codes number 04

After specifying two set of bins, a decision was made to make a comparison of error estimation per bins for data with and without imbued zeros.


E. Codes number 05

It is necessery to remark that, after studying the performance of linear regression model on the input dataset, we will only keep on analyzing data according to Random Forest and XGBoost models.
For both Random Forest and XGBoost models tried to use other hyperparameters to reduce overfitting, and finally a best possible ReducedOverfitting model was obtained for each made regression model which at least aided to increase train error while keeping the test error same as before.
Subsequently the done analysis in this code was, RMSE and Weighted RMSE estimation per bins for dataset with all target values and where damaged values are greater than 10.


F. Codes number 06

In addition of RMSE estimation per bins, the RMSE estimated in total(not per bins) for dataset with all damaged values and dataset where damaged values are greater than 10.  


G. Codes number 07

In the all above scenarios (such as data with and without zeros, different binsets, target greater than 10, ...), a scatter plot displayed True target values versus Predicted values for two Regression Models.


H. Codes number 08

RMSE estimation while not using train_test_split function but instead test and train sets are fixed based on two different manually selected list of typhoons as test data. Hence, 8 typhoons considered as the test set over the total 39 typhoons.
In each chosen list there are 4 sever and 4 mild typhoons in terms of their mean values.


I. Codes number 09

The idea is to do a train-test-split based on the typhoon's time. Therefore, typhoons are ordered with respect to their time of occurrence and then the 8 recent typhoons are selected as the test set.
In the next, 8 different RMSEs are estimated while for each calculation one of the less recent typhoons was dropped from the test set and added to the train set. The idea is to determine how well the model performs in learning from older typhoons' characteristics to make predictions on the target value of the most recent ones.

