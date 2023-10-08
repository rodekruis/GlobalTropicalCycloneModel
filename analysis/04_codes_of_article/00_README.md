# Report of article codes

All the codes from numbers 01 to 13 are implemented to get the result in
different sections of the paper.

Codes
[01_Distribution of target variable for both Grid and Municipality datasets]
(01_Distribution_of_Damage.ipynb)

In this code we compare the target variable in the municipality and grid-based
datasets, to better understand the target variable's distribution.
Therefore, we get the frequency of damage data, and as it is obvious we have a
low number of samples for high values.

Achievements: This bar plot provides insights into class imbalances and target
variable characteristics. In addition, this data visualization will support the
decision-making throughout the modeling process.

Code
[02_Correlation Matrix of features in Municipality dataset]
(02_Correlation_Matrix.ipynb)

This code is to find the relationship between all the features in the
municipality-based dataset by the measure of the linear relationship between
every two variables using the [Pearson correlation coefficient]
(<https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>). We also search
for highly correlated features that are specified in strong red
(positive relationship) or strong blue (negative relationship).

Achievements: Since the Pearson correlation coefficient is used to identify
relevant features in a dataset, it could be helpful in the variable selection
process for the machine learning model.

Code
[03_XGBoost Regression model on Municipality dataset (M-Local & M-Global)]
(03_Municipality_regression_model.ipynb)

In this code, we implement the XGBoost regression algorithm on two different
datasets: 1- Municipality-based data after removing the highly correlated features,
2- Municipality-based data only with globally available features) and
we estimate an average of RMSE, Average Error, and Standard Deviation for
20 runs in total and per bin.

Achievements: We add the results to Tables 2 and 3 in the Result section of our
paper.
These results represent that M-Local including all variable perform better than
M-Global overall and even in high bins.

Code
[04_Naive baseline model on Municipality target variable]
(04_Municipality_naive_baseline.ipynb)

In this code, we implement a Naive baseline model as a simple algorithm to provide
predictions without complex computations. The Baseline returns the average of the
target from training data. It should be mentioned that the Naive baseline assumes
independence among features.

Achievements: We add this result to Tables 2 and 3 in the Result section of our
paper.
Since this algorithm is easy to implement and even efficient for large datasets,
it was a good option to give us a basic result to compare with models M-Local and
M-Global.

Code
[05_XGBoost Regression model on transformed Grid To Municipality dataset
(G-Global & G-Global+)](05_grid_To_mun_regression_model.ipynb)

In this code, we implement the XGBoost regression algorithm on two different
datasets: 1- Grid-based data including only globally available features,
2- Grid dataset after adding more globally available features) and we estimate
an average of RMSE, Average Error, and Standard Deviation for 20 runs in total
and per bin. The point is that, in order to have a fair comparison we transformed
the data from grid format to municipality and got the results.

Achievements: We add the results to Tables 2 and 3 in the Result section of our
paper.
These results represent that Grid-based models outperform with respect to
Municipality-based models while G-Global+ even performs better than G-Global+.

Code
[06_Naive baseline model on transformed Grid To Municipality target variable]
(06_grid_To_mun_naive_baseline.ipynb)

In this code, we implement a Naive baseline model as a simple algorithm to provide
predictions without complex computations. The Baseline returns the average of the
target from training data. It should be mentioned that the Naive baseline assumes
independence among features. The point is that, in order to have a fair comparison
we transformed the data from grid format to municipality and get the results.

Achievements: We add this result to Tables 2 and 3 in the Result section of our
paper.
Since this algorithm is easy to implement and even efficient for large datasets,
it was a good option to give us a basic result to compare with models G-Global and
G-Global+.

Code
[07_Combined model on transformed Grid To Municipality dataset (2SG-Global+)]
(07_grid_To_mun_2SG-Global+.ipynb)

In this code, we build a hybrid model using XGBoost regression and XGBoost
classifier and then implement this model on the grid dataset based on all
globally available features. The aim is to achieve better performance on
high-damage data which includes a lower number of samples. The point is that,
in order to have a fair comparison we transformed the data from grid format
to municipality and got the results.

Achievements: We add this result to Tables 2 and 3 in the Result section of
our paper. The model could slightly outperform the last two bins
(high-damage bins) while it was almost worse on the first three bins.

Code
[08_Beeswarm plot for Grid dataset with all globally available features]
(08_Beeswarm_plot_G-Global+.ipynb)

Feature Importance based on XGBoost regression model on grid dataset including
all globally available features. We applied feature importance to input data
using SHAP values to find which features are the most effective ones in model
prediction. Then we display the result using a Beeswarm plot.

Achievements: Figure 10 in the paper shows this result and wind speed is the
most effective feature in the predictive model with a strong positive impact on
the prediction of the target variable.

Code
[09_Naive baseline model Typhoon split on transformed Grid To Municipality
target variable (G-Naive)](09_TyphoonSplit_G-Naive.ipynb)

In this code, we perform train/test splits with stratification by typhoon in
two ways: iterative walk-forward evaluation and leave-one-out cross-validation
(LOOCV) and we estimate the RMSE, Average Error, and Standard Deviation in an
average of 20 runs of Naive baseline model for the Grid dataset.
The walk forward evaluation uses a chronologically ordered set of typhoons,
starting with an initial training set of 27 typhoons (70% of the data). In each
iteration, a new typhoon is added to the training set, and the model is tested on
the next one (making for 12 iterations). In the LOOCV scenario, we exclude one
typhoon for testing, and train the model on the rest, making available more data
than the walk-forward evaluation.
The point is that, in order to have a fair comparison we transformed the data from
grid format to municipality and get the results.

Achievements: We add the results to Tables 5 and 6 in the Result section of our
paper. The total RMSE displays that this model has the worst performance in the
typhoon's split.

Code
[10_XGBoost Regression model based on Typhoon split (Municipality dataset)]
(10_TyphoonSplit_M-Local.ipynb)

In this code, we perform train/test splits with stratification by typhoon in
two ways: iterative walk-forward evaluation and leave-one-out cross-validation
(LOOCV) and we estimate the RMSE, Average Error, and Standard Deviation in an
average of 20 runs of XGBoost Regression model for the Municipality dataset.
The walk forward evaluation uses a chronologically ordered set of typhoons,
starting with an initial training set of 27 typhoons (70% of the data). In each
iteration, a new typhoon is added to the training set, and the model is tested on
the next one (making for 12 iterations). In the LOOCV scenario, we exclude one
typhoon for testing, and train the model on the rest, making available more data
than the walk-forward evaluation.

Achievements: We add the results to Tables 5 and 6 in the Result section of our
paper. The total RMSE displays that the model outperforms in typhoon's split
compared to random train/test split and LOOCV performs slightly better than
walk-forward evaluation.

Code
[11_The hybrid model based on Typhoon split (2SG-Global+)]
(11_TyphoonSplit_2SG-Global+.ipynb)

In this code, we perform train/test splits with stratification by typhoon in
two ways: iterative walk-forward evaluation and leave-one-out cross-validation
(LOOCV) and we estimate the RMSE, Average Error, and Standard Deviation in an
average of 20 runs of Combined model for the Grid dataset.
The walk forward evaluation uses a chronologically ordered set of typhoons,
starting with an initial training set of 27 typhoons (70% of the data). In each
iteration, a new typhoon is added to the training set, and the model is tested
on the next one (making for 12 iterations). In the LOOCV scenario, we exclude
one typhoon for testing, and train the model on the rest, making available more
data than the walk-forward evaluation.
The point is that, in order to have a fair comparison we transformed the data
from grid format to municipality and get the results.

Achievements: We add the results to Tables 5 and 6 in the Result section of our
paper. The total RMSE displays that the model outperforms in typhoon's split
compared to the XGBoost Regression on Municipality dataset.

Code
[12_The prediction error for Typhoon MELOR]
(12_Typhoon(MELOR)_prediction_error.ipynb)

In this code, we estimate the predicted damage using the 2SG-Global+ model for
a certain typhoon named MELOR and plot three different figures to visualize the
model's performance.

Achievements: In our paper, three figures 16, 17, and 18 respectively indicate
the predicted damage, the ground truth, and the prediction error of the model for
typhoon MELOR. Mostly our model underestimates the damage.

Code
[13_The three models based on Typhoon split (LOOCV evaluation) for damage>10]
(13_RMSE&Ave_damage>10_LOOCV.ipynb)

In this code, we apply the two models (Naive baseline and 2SG-Global+) for those
typhoons that include only grid cells with damage > 10 that were evaluated using
LOOCV.

Achievements: Figures 11 and 12 on the paper show the performance and it seems
that for most typhoons, the error is reduced.
