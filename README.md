## Gaussian Mixture Model Classifier

In this project, Gaussian Mixture Model (GMM) is used as a generative classifier. We use the scikit-learn library from python which uses the Expectation Maximization (EM) to train a GMM model. 
A GMM model can be employed to estimate the PDF of some samples (like a parametric density estimator). 

Here, we train an individual GMM model (with K Components, K = 1,5,10,) for each class. Therefore, N GMM models will be created where N shows the number of classes. The label of a sample can be determined using Maximum Likelihood(ML) criteria. In another words, we should find the likelihood of a sample in all classes and then
select the class with the maximum likelihood as the label of the sample. Also, use five-time-five-fold cross validation to determine the best K.

### **Datasets** : 

1. User Knowledge Modeling Data Set (UKM): https://archive.ics.uci.edu/ml/datasets/User+Knowledge+Modeling

2. Iris: https://archive.ics.uci.edu/ml/datasets/Iris

3. Vehicle.dat

4. Health.dat

### Plot the training Data (Iris dataset)

![training Data](https://github.com/Ghafarian-code/GMM/blob/master/images/Iris/Figure_1.png)

### Plot the test data (Iris dataset)
![test data](https://github.com/Ghafarian-code/GMM/blob/master/images/Iris/Figure_2.png)

### Plot the test data classified by the GMM classifier for each k (Iris dataset)
![test data](https://github.com/Ghafarian-code/GMM/blob/master/images/Iris/Figure_3.png)
![test data](https://github.com/Ghafarian-code/GMM/blob/master/images/Iris/Figure_4.png)
![test data](https://github.com/Ghafarian-code/GMM/blob/master/images/Iris/Figure_5.png)
