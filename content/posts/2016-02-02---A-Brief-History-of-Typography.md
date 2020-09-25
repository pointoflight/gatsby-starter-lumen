---
title: "Variable Results in your ML model"
date: "2020-01-02T22:40:32.169Z"
template: "post"
draft: false
slug: "var-accu"
category: "Machine Learning"
tags:
  - "Machine Learning"
  - "Python"
description: "Are you getting different results every time you run your machine learning algorithm?"
socialImage: "/media/image-0.jpg"
---

**Are you getting different results for your machine learning algorithm?**

Perhaps your results differ from a tutorial and you want to understand why.

Perhaps your model is making different predictions each time it is trained, even when it is trained on the same data set each time.

This is to be expected and might even be a feature of the algorithm, not a bug.

In this tutorial, you will discover why you can expect different results when using machine learning algorithms.

After completing this tutorial, you will know:

-   Machine learning algorithms will train different models if the training dataset is changed.
-   Stochastic machine learning algorithms use randomness during learning, ensuring a different model is trained each run.
-   Differences in the development environment, such as software versions and CPU type, can cause rounding error differences in predictions and model evaluations.

Let’s get started.

## Tutorial Overview

This tutorial is divided into five parts; they are:

1.  Help, I’m Getting Different Results!?
2.  Differences Caused by Training Data
3.  Differences Caused by Learning Algorithm
4.  Differences Caused by Evaluation Procedure
5.  Differences Caused by Platform

## 1. Help, I’m Getting Different Results!?

Don’t panic. Machine learning algorithms or models can give different results.

**It’s not your fault**. In fact, it is often a feature, not a bug.

We will clearly specify and explain the problem you are having.

First, let’s get a handle on the basics.

In applied machine learning, we run a machine learning “_algorithm_” on a dataset to get a machine learning “_model_.” The model can then be evaluated on data not used during training or used to make predictions on new data, also not seen during training.

-   **Algorithm**: Procedure run on data that results in a model (e.g. training or learning).
-   **Model**: Data structure and coefficients used to make predictions on data.

For more on the difference between machine learning algorithms and models, see the tutorial:

-   [Difference Between Algorithm and Model in Machine Learning](https://machinelearningmastery.com/difference-between-algorithm-and-model-in-machine-learning/)

Supervised machine learning means we have examples (rows) with input and output variables (columns). We cannot write code to predict outputs given inputs because it is too hard, so we use machine learning algorithms to learn how to predict outputs from inputs given historical examples.

This is called  [function approximation](https://machinelearningmastery.com/neural-networks-are-function-approximators/), and we are learning or searching for a function that maps inputs to outputs on our specific prediction task in such a way that it has skill, meaning the performance of the mapping is better than random and ideally better than all other algorithms and algorithm configurations we have tried.

-   **Supervised Learning**: Automatically learn a mapping function from examples of inputs to examples of outputs.

In this sense, a machine learning model is a program we intend to use for some project or application; it just so happens that the program was learned from examples (using an algorithm) rather than written explicitly with if-statements and such. It’s a type of automatic programming.

-   **Machine Learning Model**: A “_program_” automatically learned from historical data.

Unlike the programming that we may be used to, the programs may not be entirely deterministic.

The machine learning models may be different each time they are trained. In turn, the models may make different predictions, and when evaluated, may have a different level of error or accuracy.

There are at least four cases where you will get different results; they are:

-   Different results because of differences in training data.
-   Different results because of stochastic learning algorithms.
-   Different results because of stochastic evaluation procedures.
-   Different results because of differences in platform.

Let’s take a closer look at each in turn.

**Did I miss a possible cause of a difference in results?**  
Let me know in the comments below.

## 2. Differences Caused by Training Data

You will get different results when you run the same algorithm on different data.

This is referred to as the variance of the machine learning algorithm. You may have heard of it in the context of the bias-variance trade-off.

The variance is a measure of how sensitive the algorithm is to the specific data used during training.

-   **Variance**: How sensitive the algorithm is to the specific data used during training.

A more sensitive algorithm has a larger variance, which will result in more difference in the model, and in turn, the predictions made and evaluation of the model. Conversely, a less sensitive algorithm has a smaller variance and will result in less difference in the resulting model with different training data, and in turn, less difference in the resulting predictions and model evaluation.

-   **High Variance**: Algorithm is more sensitive to the specific data used during training.
-   **Low Variance**: Algorithm is less sensitive to the specific data used during training.

For more on the variance and the bias-variance trade-off, see the tutorial:

-   [Gentle Introduction to the Bias-Variance Trade-Off in Machine Learning](https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/)

All useful machine learning algorithms will have some variance, and some of the most effective algorithms will have a high variance.

Algorithms with a high variance often require more training data than those algorithms with less variance. This is intuitive if we consider the model approximating a mapping function from inputs and outputs and the  [law of large numbers](https://machinelearningmastery.com/a-gentle-introduction-to-the-law-of-large-numbers-in-machine-learning/).

Nevertheless, when you train a machine learning algorithm on different training data, you will get a different model that has different behavior. This means different training data will give models that make different predictions and have a different estimate of performance (e.g. error or accuracy).

The amount of difference in the results will be related to how different the training data is for each model, and the variance of the specific model and model configuration you have chosen.

### What Should I Do?

You can often reduce the variance of the model by changing a hyperparameter of the algorithm.

For example, the  _k_  in  _k_-nearest neighbors controls the variance of the algorithm, where small values like  _k=1_  result in high variance and large values like  _k=21_  result in low variance.

You can reduce the variance by changing the algorithm. For example, simpler algorithms like linear regression and logistic regression have a lower variance than other types of algorithms.

You can also lower the variance with a high variance algorithm by increasing the size of the training dataset, meaning you may need to collect more data.

## 3. Differences Caused by Learning Algorithm

You can get different results when you run the same algorithm on the same data due to the nature of the learning algorithm.

This is the most likely reason that you’re reading this tutorial.

You run the same code on the same dataset and get a model that makes different predictions or has a different performance each time, and you think it’s a bug or something.  _**Am I right?**_

It’s not a bug, it’s a feature.

Some machine learning algorithms are deterministic. Just like the programming that you’re used to. That means, when the algorithm is given the same dataset, it learns the same model every time. An example is a linear regression or logistic regression algorithm.

Some algorithms are not deterministic; instead, they are stochastic. This means that their behavior incorporates elements of randomness.

**Stochastic does not mean random**. Stochastic machine learning algorithms are not learning a random model. They are learning a model conditional on the historical data you have provided. Instead, the specific small decisions made by the algorithm during the learning process can vary randomly.

The impact is that each time the stochastic machine learning algorithm is run on the same data, it learns a slightly different model. In turn, the model may make slightly different predictions, and when evaluated using error or accuracy, may have a slightly different performance.

For more on stochastic and what it means in machine learning, see the tutorial:

-   [What Does Stochastic Mean in Machine Learning?](https://machinelearningmastery.com/stochastic-in-machine-learning/)

Adding randomness to some of the decisions made by an algorithm can improve performance on hard problems. Learning a supervised learning mapping function with a limited sample of data from the domain is a very  [hard problem](https://machinelearningmastery.com/applied-machine-learning-is-hard/).

Finding a good or best mapping function for a dataset is a type of search problem. We test different algorithms and test algorithm configurations that define the shape of the search space and give us a starting point in the search space. We then run the algorithms, which then navigate the search space to a single model.

Adding randomness can help avoid the good solutions and help find the really good and great solutions in the search space. They allow the model to escape local optima or deceptive local optima where the learning algorithm might get such, and help find better solutions, even a global optima.

For more on thinking about supervised learning as a search problem, see the tutorial:

-   [A Gentle Introduction to Applied Machine Learning as a Search Problem](https://machinelearningmastery.com/applied-machine-learning-as-a-search-problem/)

An example of an algorithm that uses randomness during learning is a neural network. It uses randomness in two ways:

-   Random initial weights (model coefficients).
-   Random shuffle of samples each epoch.

Neural networks (deep learning) are a stochastic machine learning algorithm. The random initial weights allow the model to try learning from a different starting point in the search space each algorithm run and allow the learning algorithm to “_break symmetry_” during learning. The random shuffle of examples during training ensures that each gradient estimate and weight update is slightly different.

For more on the stochastic nature of neural networks, see the tutorial:

-   [Why Initialize a Neural Network With Random Weights?](https://machinelearningmastery.com/why-initialize-a-neural-network-with-random-weights/)

Another example is ensemble machine learning algorithms that are stochastic, such as bagging.

Randomness is used in the sampling procedure of the training dataset that ensures a different decision tree is prepared for each contributing member in the ensemble. In ensemble learning, this is called ensemble diversity and is an approach to simulating independent predictions from a single training dataset.

For more on the stochastic nature of bagging ensembles, see the tutorial:

-   [How to Develop a Bagging Ensemble With Python](https://machinelearningmastery.com/bagging-ensemble-with-python/)

### What Should I Do?

The randomness used by learning algorithms can be controlled.

For example, you set the seed used by the pseudorandom number generator to ensure that each time the algorithm is run, it gets the same randomness.

For more on random number generators and setting fixing the seed, see the tutorial:

-   [Introduction to Random Number Generators for Machine Learning in Python](https://machinelearningmastery.com/introduction-to-random-number-generators-for-machine-learning/)

This can be a good approach for tutorials, but not a good approach in practice. It leads to questions like:

-   _What is the best seed for the pseudorandom number generator?_

There is no best seed for a stochastic machine learning algorithm. You are fighting the nature of the algorithm, forcing stochastic learning to be deterministic.

You could make a case that the final model is fit using a fixed seed to ensure the same model is created from the same data before being used in production prior to any pre-deployment system testing. Nevertheless, as soon as the training dataset changes, the model will change.

A better approach is to embrace the stochastic nature of machine learning algorithms.

Consider that there is not a single model for your dataset. Instead, there is a stochastic process (the algorithm pipeline) that can generate models for your problem.

For more on this, see the tutorial:

-   [Embrace Randomness in Machine Learning](https://machinelearningmastery.com/randomness-in-machine-learning/)

You can then summarize the performance of these models — of the algorithm pipeline — as a distribution with mean expected error or accuracy and a standard deviation.

You can then ensure you achieve the average performance of the models by fitting multiple final models on your dataset and averaging their predictions when you need to make a prediction on new data.

For more on the ensemble approach to final models, see the tutorial:

-   [How to Reduce Variance in a Final Machine Learning Model](https://machinelearningmastery.com/how-to-reduce-model-variance/)

## 4. Differences Caused by Evaluation Procedure

You can get different results when running the same algorithm with the same data due to the evaluation procedure.

The two most common evaluation procedures are a train-test split and k-fold cross-validation.

A  [train-test split](https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/)  involves randomly assigning rows to either be used to train the model or evaluate the model to meet a predefined train or test set size.

For more on the train-test split, see the tutorial:

-   [Train-Test Split for Evaluating Machine Learning Algorithms](https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/)

The k-fold cross-validation procedure involves dividing a dataset into k non-overlapping partitions and using one fold as the test set and all other folds as the training set. A model is fit on the training set and evaluated on the holdout fold and this process is repeated k times, giving each fold an opportunity to be used as the holdout fold.

For more on k-fold cross-validation, see the tutorial:

-   [A Gentle Introduction to k-fold Cross-Validation](https://machinelearningmastery.com/k-fold-cross-validation/)

Both of these model evaluation procedures are stochastic.

Again, this does not mean that they are random; it means that small decisions made in the process involve randomness. Specifically, the choice of which rows are assigned to a given subset of the data.

This use of randomness is a feature, not a bug.

The use of randomness, in this case, allows the resampling to approximate an estimate of model performance that is independent of the specific data sample drawn from the domain. This approximation is biased because we only have a small sample of data to work with rather than the complete set of possible observations.

Performance estimates provide an idea of the expected or average capability of the model when making predictions in the domain on data not seen during training. Regardless of the specific rows of data used to train or test the model, at least ideally.

For more on the more general topic of statistical sampling, see the tutorial:

-   [A Gentle Introduction to Statistical Sampling and Resampling](https://machinelearningmastery.com/statistical-sampling-and-resampling/)

As such, each evaluation of a deterministic machine learning algorithm, like a linear regression or a logistic regression, can give a different estimate of error or accuracy.

### What Should I Do?

The solution in this case is much like the case for stochastic learning algorithms.

The seed for the pseudorandom number generator can be fixed or the randomness of the procedure can be embraced.

Unlike stochastic learning algorithms, both solutions are quite reasonable.

If a large number of machine learning algorithms and algorithm configurations are being evaluated systematically on a predictive modeling task, it can be a good idea to fix the random seed of the evaluation procedure. Any value will do.

The idea is that each candidate solution (each algorithm or configuration) will be evaluated in an identical manner. This ensures an apples-to-apples comparison. It also allows for the use of paired statistical hypothesis tests later, if needed, to check if differences between algorithms are statistically significant.

Embracing the randomness can also be appropriate. This involves repeating the evaluation procedure many times and reporting a summary of the distribution of performance scores, such as the mean and standard deviation.

Perhaps the least biased approach to repeated evaluation would be to use  [repeated k-fold cross-validation](https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/), such as three repeats with 10 folds (3×10), which is common, or five repeats with two folds (5×2), which is commonly used when comparing algorithms with statistical hypothesis tests.

For a gentle introduction to using statistical hypothesis tests for comparing algoritms, see the tutorial:

-   [Statistical Significance Tests for Comparing Machine Learning Algorithms](https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/)

For a tutorial on comparing mean algorithm performance with a hypothesis test, see the tutorial:

-   [Hypothesis Test for Comparing Machine Learning Algorithms](https://machinelearningmastery.com/hypothesis-test-for-comparing-machine-learning-algorithms/)

## 5. Differences Caused by Platform

You can get different results when running the same algorithm on the same data on different computers.

This can happen even if you fix the random number seed to address the stochastic nature of the learning algorithm and evaluation procedure.

The cause in this case is the platform or development environment used to run the example, and the results are often different in minor ways, but not always.

This includes:

-   Differences in the  **system architecture**, e.g. CPU or GPU.
-   Differences in the  **operating system**, e.g. MacOS or Linux.
-   Differences in the  **underlying math libraries**, e.g. LAPACK or BLAS.
-   Differences in the  **Python version**, e.g. 3.6 or 3.7.
-   Differences in the  **library version**, e.g. scikit-learn 0.22 or 0.23.
-   …

Machine learning algorithms are a type of numerical computation.

This means that they typically involve a lot of math with floating point values. Differences in aspects, such as the architecture and operating system, can result in differences in round errors, which can compound with the number of calculations performed to give very different results.

Additionally, differences in the version of libraries can mean the fixing of bugs and the changing of functionality that too can result in different results.

Additionally, this also explains why you will get different results for the same algorithm on the same machine implemented by different languages, such as R and Python. Small differences in the implementation and/or differences in the underlying math libraries used will cause differences in the resulting model and predictions made by that model.

### What Should I Do?

This does not mean that the platform itself can be treated as a hyperparameter and tuned for a predictive modeling problem.

Instead, it means that the platform is an important factor when evaluating machine learning algorithms and should be fixed or fully described to ensure full reproducibility when moving from development to production, or in reporting performance in academic studies.

One approach might be to use virtualization, such as docker or a virtual machine instance to ensure the environment is kept constant, if full reproducibility is critical to a project.

Honestly, the effect is often very small in practice (at least in my limited experience) as long as major software versions are a good or close enough match.