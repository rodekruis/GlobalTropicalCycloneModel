## Report of analysis: code and steps

All the codes from number 01 to 15 are related to the analysis of the
input data, features, and the three main implemented models (regression, classification, and hybrid one).

Code 01 <br />
[Input data collection](01_collate_data.ipynb)

In this code we collect and combine the data needed to train the model including windspeed, rainfall and etc.

Code 02 <br />
[Naive baseline model](02_model_training-baselines.ipynb)

In this code, we implemented a Naive baseline model as a simple algorithm to provide predictions without complex computations.

Code 02 <br />
[XGBoost regression model](03_model_training.ipynb)

In this code, we trained the XGBoost regression model obtained from the baseline model analysis, on the collected dataset (data in grid format).
