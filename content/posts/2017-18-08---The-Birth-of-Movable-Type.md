---
title: "AutoML Libraries for Python"
date: "2019-11-18T22:12:03.284Z"
template: "post"
draft: false
slug: "auto-ml"
category: "Machine Learning"
tags:
  - "Python"
  - "Machine Learning"
description: "AutoML provides tools to automatically discover good machine learning model pipelines for a dataset with very little user intervention."
socialImage: "/media/gutenberg.jpg"
---

AutoML provides tools to automatically discover good machine learning model pipelines for a dataset with very little user intervention.

It is ideal for domain experts new to machine learning or machine learning practitioners looking to get good results quickly for a predictive modeling task.

Open-source libraries are available for using AutoML methods with popular machine learning libraries in Python, such as the scikit-learn machine learning library.

In this tutorial, you will discover how to use top open-source AutoML libraries for scikit-learn in Python.

After completing this tutorial, you will know:

-   AutoML are techniques for automatically and quickly discovering a well-performing machine learning model pipeline for a predictive modeling task.
-   The three most popular AutoML libraries for Scikit-Learn are Hyperopt-Sklearn, Auto-Sklearn, and TPOT.
-   How to use AutoML libraries to discover well-performing models for predictive modeling tasks in Python.

Let’s get started.

## Tutorial Overview

This tutorial is divided into four parts; they are:

1.  Automated Machine Learning
2.  Auto-Sklearn
3.  Tree-based Pipeline Optimization Tool (TPOT)
4.  Hyperopt-Sklearn

## Automated Machine Learning

Automated Machine Learning, or AutoML for short, involves the automatic selection of data preparation, machine learning model, and model hyperparameters for a predictive modeling task.

It refers to techniques that allow semi-sophisticated machine learning practitioners and non-experts to discover a good predictive model pipeline for their machine learning task quickly, with very little intervention other than providing a dataset.

> … the user simply provides data, and the AutoML system automatically determines the approach that performs best for this particular application. Thereby, AutoML makes state-of-the-art machine learning approaches accessible to domain scientists who are interested in applying machine learning but do not have the resources to learn about the technologies behind it in detail.

— Page ix,  [Automated Machine Learning: Methods, Systems, Challenges](https://amzn.to/2w2gVf4s), 2019.

Central to the approach is defining a large hierarchical optimization problem that involves identifying data transforms and the machine learning models themselves, in addition to the hyperparameters for the models.

Many companies now offer AutoML as a service, where a dataset is uploaded and a model pipeline can be downloaded or hosted and used via web service (i.e. MLaaS). Popular examples include service offerings from Google, Microsoft, and Amazon.

Additionally, open-source libraries are available that implement AutoML techniques, focusing on the specific data transforms, models, and hyperparameters used in the search space and the types of algorithms used to navigate or optimize the search space of possibilities, with versions of  [Bayesian Optimization](https://machinelearningmastery.com/what-is-bayesian-optimization/)  being the most common.

There are many open-source AutoML libraries, although, in this tutorial, we will focus on the best-of-breed libraries that can be used in conjunction with the popular scikit-learn Python machine learning library.

They are: Hyperopt-Sklearn, Auto-Sklearn, and TPOT.

We will take a closer look at each, providing the basis for you to evaluate and consider which library might be appropriate for your project.

## Auto-Sklearn

Auto-Sklearn is an open-source Python library for AutoML using machine learning models from the scikit-learn machine learning library.

It was developed by  [Matthias Feurer](https://ml.informatik.uni-freiburg.de/people/feurer/index.html), et al. and described in their 2015 paper titled “[Efficient and Robust Automated Machine Learning](https://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning).”

> … we introduce a robust new AutoML system based on scikit-learn (using 15 classifiers, 14 feature preprocessing methods, and 4 data preprocessing methods, giving rise to a structured hypothesis space with 110 hyperparameters).

—  [Efficient and Robust Automated Machine Learning](https://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning), 2015.

The first step is to install the Auto-Sklearn library, which can be achieved using pip, as follows:
```python
sudo pip install autosklearn
```

Once installed, we can import the library and print the version number to confirm it was installed successfully:

```python
# print autosklearn version
import autosklearn
print('autosklearn: %s' % autosklearn.__version__)
```

Running the example prints the version number. Your version number should be the same or higher.
```python
autosklearn: 0.6.0
```
Next, we can demonstrate using Auto-Sklearn on a synthetic classification task.

We can define an AutoSklearnClassifier class that controls the search and configure it to run for two minutes (120 seconds) and kill any single model that takes more than 30 seconds to evaluate. At the end of the run, we can report the statistics of the search and evaluate the best performing model on a holdout dataset.

The complete example is listed below.
```python 
# example of auto-sklearn for a classification dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from autosklearn.classification import AutoSklearnClassifier
# define dataset
X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# define search
model = AutoSklearnClassifier(time_left_for_this_task=2*60, per_run_time_limit=30, n_jobs=8)
# perform the search
model.fit(X_train, y_train)
# summarize
print(model.sprint_statistics())
# evaluate best model
y_hat = model.predict(X_test)
acc = accuracy_score(y_test, y_hat)
print("Accuracy: %.3f" % acc)
```


Running the example will take about two minutes, given the hard limit we imposed on the run.

At the end of the run, a summary is printed showing that 599 models were evaluated and the estimated performance of the final model was 95.6 percent.
```python 
auto-sklearn results:
Dataset name: 771625f7c0142be6ac52bcd108459927
Metric: accuracy
Best validation score: 0.956522
Number of target algorithm runs: 653
Number of successful target algorithm runs: 599
Number of crashed target algorithm runs: 54
Number of target algorithms that exceeded the time limit: 0
Number of target algorithms that exceeded the memory limit: 0
```

We then evaluate the model on the holdout dataset and see that a classification accuracy of 97 percent was achieved, which is reasonably skillful.
```python
Accuracy: 0.970
```

For more on the Auto-Sklearn library, see:

-   [Auto-Sklearn Homepage](https://automl.github.io/auto-sklearn/master/).
-   [Auto-Sklearn GitHub Project](https://github.com/automl/auto-sklearn).

## Tree-based Pipeline Optimization Tool (TPOT)

Tree-based Pipeline Optimization Tool, or  [TPOT](https://epistasislab.github.io/tpot/)  for short, is a Python library for automated machine learning.

TPOT uses a tree-based structure to represent a model pipeline for a predictive modeling problem, including data preparation and modeling algorithms, and model hyperparameters.

> … an evolutionary algorithm called the Tree-based Pipeline Optimization Tool (TPOT) that automatically designs and optimizes machine learning pipelines.

—  [Evaluation of a Tree-based Pipeline Optimization Tool for Automating Data Science](https://dl.acm.org/doi/10.1145/2908812.2908918), 2016.

The first step is to install the TPOT library, which can be achieved using pip, as follows:
```python
pip install tpot
```
Once installed, we can import the library and print the version number to confirm it was installed successfully:
```python
# check tpot version
import tpot
print('tpot: %s'  %  tpot.__version__)
```
Running the example prints the version number. Your version number should be the same or higher.

```python
tpot: 0.11.1
```
Next, we can demonstrate using TPOT on a synthetic classification task.

This involves configuring a TPOTClassifier instance with the population size and number of generations for the evolutionary search, as well as the cross-validation procedure and metric used to evaluate models. The algorithm will then run the search procedure and save the best discovered model pipeline to file.

The complete example is listed below.
```python
# example of tpot for a classification dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from tpot import TPOTClassifier
# define dataset
X,  y  =  make_classification(n_samples=100,  n_features=10,  n_informative=5,  n_redundant=5,  random_state=1)
# define model evaluation
cv  =  RepeatedStratifiedKFold(n_splits=10,  n_repeats=3,  random_state=1)
# define search
model  =  TPOTClassifier(generations=5,  population_size=50,  cv=cv,  scoring='accuracy',  verbosity=2,  random_state=1,  n_jobs=-1)
# perform the search
model.fit(X,  y)
# export the best model
model.export('tpot_best_model.py')
```
Running the example may take a few minutes, and you will see a progress bar on the command line.
The accuracy of top-performing models will be reported along the way.
Your specific results will vary given the stochastic nature of the search procedure.
```python
Generation 1 - Current best internal CV score: 0.9166666666666666
Generation 2 - Current best internal CV score: 0.9166666666666666
Generation 3 - Current best internal CV score: 0.9266666666666666
Generation 4 - Current best internal CV score: 0.9266666666666666
Generation 5 - Current best internal CV score: 0.9266666666666666

Best pipeline: ExtraTreesClassifier(input_matrix, bootstrap=False, criterion=gini, max_features=0.35000000000000003, min_samples_leaf=2, min_samples_split=6, n_estimators=100)
```
In this case, we can see that the top-performing pipeline achieved the mean accuracy of about 92.6 percent.

The top-performing pipeline is then saved to a file named “_tpot_best_model.py_“.

Opening this file, you can see that there is some generic code for loading a dataset and fitting the pipeline. An example is listed below.

```python
import numpy as  np
import pandas as  pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data  =  pd.read_csv('PATH/TO/DATA/FILE',  sep='COLUMN_SEPARATOR',  dtype=np.float64)
features  =  tpot_data.drop('target',  axis=1)
training_features,  testing_features,  training_target,  testing_target  =  \
train_test_split(features,  tpot_data['target'],  random_state=1)
# Average CV score on the training set was: 0.9266666666666666
exported_pipeline  =  ExtraTreesClassifier(bootstrap=False,  criterion="gini",  max_features=0.35000000000000003,  min_samples_leaf=2,  min_samples_split=6,  n_estimators=100)
# Fix random state in exported estimator
if  hasattr(exported_pipeline,  'random_state'):
setattr(exported_pipeline,  'random_state',  1)
exported_pipeline.fit(training_features,  training_target)
results  =  exported_pipeline.predict(testing_features)
```
You can then retrieve the code for creating the model pipeline and integrate it into your project.

For more on TPOT, see the following resources:

-   [Evaluation of a Tree-based Pipeline Optimization Tool for Automating Data Science](https://dl.acm.org/doi/10.1145/2908812.2908918), 2016.
-   [TPOT Documentation](https://epistasislab.github.io/tpot/).
-   [TPOT GitHub Project](https://github.com/EpistasisLab/tpot).

## Hyperopt-Sklearn

[HyperOpt](http://hyperopt.github.io/hyperopt/)  is an open-source Python library for Bayesian optimization developed by  [James Bergstra](https://www.linkedin.com/in/james-bergstra).

It is designed for large-scale optimization for models with hundreds of parameters and allows the optimization procedure to be scaled across multiple cores and multiple machines.

HyperOpt-Sklearn wraps the HyperOpt library and allows for the automatic search of data preparation methods, machine learning algorithms, and model hyperparameters for classification and regression tasks.

> … we introduce Hyperopt-Sklearn: a project that brings the benefits of automatic algorithm configuration to users of Python and scikit-learn. Hyperopt-Sklearn uses Hyperopt to describe a search space over possible configurations of Scikit-Learn components, including preprocessing and classification modules.

—  [Hyperopt-Sklearn: Automatic Hyperparameter Configuration for Scikit-Learn](https://conference.scipy.org/proceedings/scipy2014/pdfs/komer.pdf), 2014.

Now that we are familiar with HyperOpt and HyperOpt-Sklearn, let’s look at how to use HyperOpt-Sklearn.

The first step is to install the HyperOpt library.

This can be achieved using the pip package manager as follows:
```python
sudo pip install hyperopt
```
Next, we must install the HyperOpt-Sklearn library.

This too can be installed using pip, although we must perform this operation manually by cloning the repository and running the installation from the local files, as follows:
```
git clone git@github.com:hyperopt/hyperopt-sklearn.git
cd hyperopt-sklearn
sudo pip install .
cd ..
```
We can confirm that the installation was successful by checking the version number with the following command:
```
sudo pip show hpsklearn
```
This will summarize the installed version of HyperOpt-Sklearn, confirming that a modern version is being used.
```python
Name: hpsklearn
Version: 0.0.3
Summary: Hyperparameter Optimization for sklearn
Home-page: http://hyperopt.github.com/hyperopt-sklearn/
Author: James Bergstra
Author-email: anon@anon.com
License: BSD
Location: ...
Requires: nose, scikit-learn, numpy, scipy, hyperopt
Required-by:
```
Next, we can demonstrate using Hyperopt-Sklearn on a synthetic classification task.

We can configure a HyperoptEstimator instance that runs the search, including the classifiers to consider in the search space, the pre-processing steps, and the search algorithm to use. In this case, we will use TPE, or Tree of Parzen Estimators, and perform 50 evaluations.

At the end of the search, the best performing model pipeline is evaluated and summarized.

The complete example is listed below.
```python
# example of hyperopt-sklearn for a classification dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from hpsklearn import HyperoptEstimator
from hpsklearn import any_classifier
from hpsklearn import any_preprocessing
from hyperopt import tpe
# define dataset
X,  y  =  make_classification(n_samples=100,  n_features=10,  n_informative=5,  n_redundant=5,  random_state=1)
# split into train and test sets
X_train,  X_test,  y_train,  y_test  =  train_test_split(X,  y,  test_size=0.33,  random_state=1)
# define search
model  =  HyperoptEstimator(classifier=any_classifier('cla'),  preprocessing=any_preprocessing('pre'),  algo=tpe.suggest,  max_evals=50,  trial_timeout=30)
# perform the search
model.fit(X_train,  y_train)
# summarize performance
acc  =  model.score(X_test,  y_test)
print("Accuracy: %.3f"  %  acc)
# summarize the best model
print(model.best_model())
```
Running the example may take a few minutes.

The progress of the search will be reported and you will see some warnings that you can safely ignore.

At the end of the run, the best-performing model is evaluated on the holdout dataset and the Pipeline discovered is printed for later use.

Your specific results may differ given the stochastic nature of the learning algorithm and search process. Try running the example a few times.

In this case, we can see that the chosen model achieved an accuracy of about 84.8 percent on the holdout test set. The Pipeline involves a SGDClassifier model with no pre-processing.
```python
Accuracy: 0.848
{'learner': SGDClassifier(alpha=0.0012253733891387925, average=False,
class_weight='balanced', early_stopping=False, epsilon=0.1,
eta0=0.0002555872679483392, fit_intercept=True,
l1_ratio=0.628343459087075, learning_rate='optimal',
loss='perceptron', max_iter=64710625.0, n_iter_no_change=5,
n_jobs=1, penalty='l2', power_t=0.42312829309173644,
random_state=1, shuffle=True, tol=0.0005437535215080966,
validation_fraction=0.1, verbose=False, warm_start=False), 'preprocs': (), 'ex_preprocs': ()}
```
The printed model can then be used directly, e.g. the code copy-pasted into another project.
For more on Hyperopt-Sklearn, see:

-   [Hyperopt-Sklearn Homepage](http://hyperopt.github.io/hyperopt-sklearn/).
-   [Hyperopt-Sklearn GitHub Project](https://github.com/hyperopt/hyperopt-sklearn).

## Summary

In this tutorial, you discovered how to use top open-source AutoML libraries for scikit-learn in Python.

Specifically, you learned:

-   AutoML are techniques for automatically and quickly discovering a well-performing machine learning model pipeline for a predictive modeling task.
-   The three most popular AutoML libraries for Scikit-Learn are Hyperopt-Sklearn, Auto-Sklearn, and TPOT.
-   How to use AutoML libraries to discover well-performing models for predictive modeling tasks in Python.
