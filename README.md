# Kaggle digit recognizer problem
The url for this project on kaggle is 
https://www.kaggle.com/c/digit-recognizer.
I have also included Dr. Zhang's write up in this git repo.

We need to use some combination of original code and open--sourced
packages to claim that we have explored the parameter space of 
the problem. 

# methods
Okay, so I ditched a number of the ideas that I put forward earlier.
So far, I am using a Multi-layer-perceptron (neural net) 
with various settings. At this point, the code is at such 
a catastrophic level of spaghetti that I don't really 
want anyone to see it. I promise I will clean it up later.




According to some reading I have done, all of these things should achieve at 95% accuracy or better.
Check n1try/mlp.py for some working code if you're interested.


# resources
I am going to try to collect documentation in a way that is easy to refer to 
when making the report and presentation. Everything here is easily googlable
but I figured that aggregating it couldn't hurt anything.

1. [visualizing hidden layers](https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html)

2. [on the optimism of cross validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html#sphx-glr-auto-examples-model-selection-plot-nested-cross-validation-iris-py)

3. [some usage tips for neural nets in scikit learn](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification)

4. [MLPClassifier docs](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)

5. [cross_val_score docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
