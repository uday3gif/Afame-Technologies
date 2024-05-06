# Afame-Technologies
 Data Science Internship Projects 
 
 Overview:-
 The Iris Flower Classification project aims to develop a machine learning model capable of accurately classifying Iris flowers into one of three species based on physical measurements. The Iris dataset, introduced by British statistician Ronald Fisher, comprises 150 observations of Iris flowers from three different species: Iris Setosa, Iris Versicolor, and Iris Virginica. Each observation details four features: sepal length, sepal width, petal length, and petal width.

 Objective:-
 The primary goal of this project is to train a classifier that can predict the species of an Iris flower based on the given physical attributes. This task will help demonstrate the efficacy of classification algorithms in distinguishing between similar species using quantifiable data.

 Dataset Description:-
The dataset includes 150 entries with the following columns:

 Sepal Length: the length of the sepals (in cm)
 Sepal Width: the width of the sepals (in cm)
 Petal Length: the length of the petals (in cm)
 Petal Width: the width of the petals (in cm)
 Species: the species of the Iris (Iris Setosa, Iris Versicolor, Iris Virginica)

 Methodology:-
 Data Preprocessing:
 The dataset is first checked for any missing or inconsistent data entries. Data preprocessing involves encoding the categorical variable 'Species' into a numerical format suitable for model training.

 Exploratory Data Analysis (EDA):
 Initial data exploration is performed to understand the distributions of various features and the relationship between them. This includes visualizations such as histograms, scatter plots, and box plots to inspect the characteristics of each species.

 Model Selection:
 The Random Forest classifier is selected due to its robustness and ability to handle non-linear data without extensive hyperparameter tuning. It's effective in classification tasks and provides important features influencing the predictions.

 Model Training and Evaluation:
 The dataset is split into a training set and a testing set. The model is trained on the training set and evaluated on the testing set using metrics such as accuracy, precision, recall, and the F1-score. The confusion matrix is also generated to visualize the model’s performance across different classes.

 Feature Importance Analysis:
 After model training, an analysis of feature importance is conducted to determine which characteristics are most influential in predicting the Iris species.
