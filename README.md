
### Files Overview

##### implementations.py: This code provides a collection of functions for performing linear and logistic regression in a machine learning context. It includes implementations of:

- Least Squares Regression: A method to fit a linear model to data using the mean squared error (MSE) loss.
- Ridge Regression: A variant of linear regression with L2 regularization to prevent overfitting.
- Logistic Regression: A classification algorithm that estimates the probability of a binary outcome.
- Regularized Logistic Regression: Logistic regression with L2 regularization to improve generalization.
- Weighted Logistic Regression: Logistic regression with customizable weights for class balancing.

These functions can be used to train and evaluate regression models using various optimization techniques, including gradient descent. 

 ##### Data_Preprocessing.ipynb: This Jupyter Notebook encapsulates the initial phases of our project. It encompasses the entirety of our data exploration and preparation efforts. In this notebook, we conducted the following key tasks:
- Initial Exploration: Comprehensively exploring the dataset, gaining insights and understanding its characteristics. Many of the visualizations and summaries we created during this exploration phase were subsequently included in our project report.
- Feature Engineering: This process involved creating, transforming, or selecting features that would be most informative for our machine learning models.
- Model Training and Tuning: Using the functions from 'implementations.py,' we trained and fine-tuned our machine learning models. We experimented with various algorithms, optimizing hyperparameters to improve model performance.
- Model Selection: Based on the results of our model training and evaluations, both locally (using data splitting for training and testing) and on the AI Crowd platform, we identified the models that produced the best results. These selected models are the ones we further investigated and refined in our subsequent project phases.


##### run.ipynb: The main executable script, crafted within a Jupyter Notebook, employs functions from implementations.py. It manages tasks such as loading data, streamlining preprocessing, setting model parameters, running model training, evaluating its performance, and generating the submission file. The results achieved at the end of this script align with those submitted to the AI Crowd submission system."


### Prerequisites

- Python 3.x
- numpy library
- (optional) Matplotlib & Seaborn (for visualization only)


### Installation

pip install -r requirements.txt

### Usage

In order to get the results on AI crowd, you need to run the notebook run.ipynb

jupyter notebook run.ipynb
