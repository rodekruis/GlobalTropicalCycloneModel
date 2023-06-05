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
[Moving Average](05_Moving_Average.ipynb)
