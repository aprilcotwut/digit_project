# Kaggle digit recognizer problem
The url for this project on kaggle is 
https://www.kaggle.com/c/digit-recognizer.
I have also included Dr. Zhang's write up in this git repo.

We need to use some combination of original code and open--sourced
packages to claim that we have explored the parameter space of 
the problem. 

# methods
We have essentially already solved this problem with a neural net, 
and so I suggest that we attempt to use the following data structures.
  1. entropy reducing decision tree
  2. random forest with several different numbers of trees (non entropy reducing)
  3. k-nearest neighbors 

According to some reading I have done, all of these things should achieve at 95% accuracy or better.
# Architecture
We need to talk about how to organize this program. 
Perhaps we could put all of the IO in a single file, 
and each method in another file? I'm working on the IO as
we speak, but it's pretty trivial
