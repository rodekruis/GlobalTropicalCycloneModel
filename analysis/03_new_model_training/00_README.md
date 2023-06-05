## Report of analysis: code and steps

All the codes from number 01 to 15 are related to the analysis of the
input data, features, and the three main implemented models (regression, classification, and hybrid one).

Code 01 <br />
[Input data collection](01_collate_data.ipynb)

Feature selection was applied by using Correlation and Variance Inflation Factor(VIF)
among all the 38 features including the target.  Value of 80 was considered as the threshold
to express the highly correlated pairs in correlation matrix and 7 was considered as the 
threshold for VIF result.
