# Feature Engineering
Automated Machine Learning Feature Engineering in Python

<p align="center">
  <img width="1000" src="https://github.com/mlaskowski17/Feature-Engineering/blob/master/images/afe.png">
</p>


### General Idea

Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work. If feature engineering is done correctly, it increases the predictive power of machine learning algorithms by creating features from raw data that help facilitate the machine learning process. Feature Engineering is an art.

Steps which are involved while solving any problem in machine learning are as follows:
- Gathering data.
- Cleaning data.
- Feature engineering.
- Defining model.
- Training, testing model and predicting the output.

Feature engineering is the most important art in machine learning which creates the huge difference between a good model and a bad model. Let’s see what feature engineering covers.

In feature engineering it is important to distinguish several tasks:
- **feature extraction** and **feature engineering**: transformation of raw data into features suitable for modeling;
- **feature transformation**: transformation of data to improve the accuracy of the algorithm;
- **feature selection**: removing unnecessary features.

So summing up:  **Automated feature engineering aims to help the data scientist by automatically creating many candidate features out of a dataset from which the best can be selected and used for training.**


### Featuretools

Featuretools operates on an idea known as [Deep Feature Synthesis](https://docs.featuretools.com/api_reference.html#deep-feature-synthesis). You can read the [original paper](http://www.jmaxkanter.com/static/papers/DSAA_DSM_2015.pdf). The concept of Deep Feature Synthesis is to use basic building blocks known as feature primitives (like the transformations and aggregations done above) that can be stacked on top of each other to form new features. The depth of a "deep feature" is equal to the number of stacked primitives. Featuretools builds on simple ideas to create a powerful method, and we will build up our understanding in much the same way.


### Dataset

To show the basic idea of feature tools the following dataset consisting of three tables was used as an example:
- **clients**: information about clients at a credit union
- **loans**: previous loans taken out by the clients
- **payments**: payments made/missed on the previous loans
