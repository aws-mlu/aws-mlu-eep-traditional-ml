# Fundamentals of Machine Learning
## A comprehensive learning path for getting started with classical machine learning

Welcome to our Fundamentals of Machine Learning content! This repository contains educational materials designed to help you understand and work with machine learning through application and the application of deep learning to text and image data.

## Content Overview

This content is structured into two focus areas:
+ Machine Learning through Application
+ Application of Deep Learning to Text and Image

Both focus areas contain 3 comprehensive modules that are broken up into lessons. To enhance learning, lessons are accompanied by interactive laboratory exercises in the form of Jupyter notebooks. The notebooks contain detailed instructions, explanations, and activities to reinforce theoretical concepts with practical implementation.

## Repository Structure

This material is organized into two primary components:

### 📚 Lessons
Contains instructional materials for each module:
- Detailed PowerPoint presentations
- PDF versions of presentations

### 🔬 Labs
Interactive Jupyter notebooks that include:
- Code examples and implementations
- Real-world use cases
- Challenge exercises

ach lesson and its corresponding lab(s) are designed to be completed sequentially, building upon concepts from previous modules and lessons. All code examples are thoroughly documented and include explanatory comments to facilitate learning.

Note: Make sure to review the lesson material before attempting the associated lab exercises for the best learning experience.
### Lab environment
All labs were last tested in Amazon SageMaker Studio in a JupyterLab space using a `ml.g4dn.large` instance running the `SageMaker Distribution 2.0.1` Image

## Content
### ### Machine Learning through Application
#### Module 1: What is ML?
| Lesson | Topic & Content | Associated Videos |
|:--|:--|:--|
Lesson 0: Introduction to ML | Introduced to the material, startting with a definition of ML along with a few examples of an ML model | TODO: add Youtube video |
| Lesson 1: Jupyter and SageMaker | fundamentals of the Jupyter Notebook application and Amazon SageMaker | TODO: add Youtube video |
| Lab 1: Getting Started with Jupyter Notebooks | hands-on practice using SageMaker and gain experience using the core UI components ||
| Lab 2: Extending Your Jupyter Notebook Understanding | practice using Jupyter notebooks more efficiently ||
| Lesson 2: Types of ML and Aspects of Successful ML Problems | common types of ML problems and the characteristics of successful ML problems | TODO: add Youtube video |
| Lesson 3: Examples of ML Applications and Why They Are Successful | examples of successful ML use cases and examine how these use cases have characteristics that help to make them successful | TODO: add Youtube video |
| Lesson 4: The ML Lifecycle, and Overfitting and Underfitting | the fundamental concepts of overfitting, underfitting, and the ML lifecycle | TODO: add Youtube video |
| Lesson 5: AutoML and AutoGluon | AutoML concepts an use of AutoGluon, an open-source AutoML library, to automate repetitive portions of the ML lifecycle | TODO: add Youtube video |
| Lab 3: Getting Started with AutoGluon | walk through the process to build a basic ML model by using AutoGluon to train a model on a dataset ||
| Lab 4: Refining Models by Using AutoGluon | continue from the previous lab, and learn how to assess the performance of the models that AutoGluon trains and how to make predictions by using the best model ||
| Lab 5: Using Advanced AutoGluon Techniques | continue from the previous lab by using a full dataset and a time limit to train the model, and by making predictions with the new model. The core purpose of this lab is to emphasize the importance of data in the process of constructing models ||

#### Module 2: Core ML Concepts for Tabular Data
| Lesson | Topic & Content | Associated Videos |
|:--|:--|:--|
| Lesson 1: Exploratory Data Analysis | basics of exploratory data analysis (EDA) so that they can understand why it is important to clean a dataset before using it to create a model | TODO: add Youtube video |
| Lab 1: Performing EDA for Categorical Variables | basic steps of EDA of categorical data, and perform initial data investigations to discover patterns, spot anomalies, and look for insights to inform later ML modeling choices ||
| Lab 2: Performing EDA for Numerical Variables | basic steps of EDA of numerical data, and perform initial data investigations to discover patterns, spot anomalies, and look for insights to inform later ML modeling choices ||
| Lesson 2: Basic Feature Engineering | basic feature engineering methods for processing categorical features, preprocessing text features, and vectorizing text features | TODO: add Youtube video |
| Lab 3: Performing Basic Feature Engineering | common techniques that are used to transform numerical features, encode categorical features, and vectorize processed text features ||
| Lesson 3: Tree-Based Models | Introduction to decision trees, the Iterative Dichotomiser (ID3) algorithm, and information gain or impurity | TODO: add Youtube video |
| Lesson 4: Optimization, Regression Models, and Regularization | how to use gradient descent to optimize ML models, when to use linear and logistic regression, and when to use regularization | TODO: add Youtube video |
| Lab 4: Using Logistic Regression | build a logistic regression model to predict a field of a dataset. They also look at how probability threshold calibration can help improve a classifier's performance ||
| Lesson 5: Hyperparameter Tuning | hyperparameters and how to tune them by using grid search, random search, and Bayesian search | TODO: add Youtube video |
| Lab 5: Using Hyperparameter Tuning | become familiar with two main types of hyperparameter tuning: grid search and randomized search. Students will use a decision tree model to learn the basics of hyperparameter tuning that can be applied to any model ||
| Lesson 6: Ensembling | Introduction to ensembling methods. This includes bagging with random forests, boosting, and stacking | TODO: add Youtube video |
| Lab 6: Using Ensemble Learning | use of ensemble methods to create a strong model by combining the predictions of multiple weak models that were built with a given dataset and a given learning algorithm ||

#### Module 3: Responsible ML
| Lesson | Topic & Content | Associated Videos |
|:--|:--|:--|
| Lesson 1: Introduction to Fairness and Bias Mitigation in ML | Introduced to fair and responsible AI. They will see an overview of how fairness needs to be incorporated into all stages of the ML lifecycle, and see how to identify different types of bias that are present in data | TODO: add Youtube video |
| Lab 1: Introduction to Fairness and Bias Mitigation in ML | fairness and bias mitigation in ML by exploring different types of bias that are present in data and practice how to build various documentation sheets ||
| Lesson 2: Designing Fair Models, and Data Integrity and Analysis | various metrics that can be used to measure bias that can be present in training data | TODO: add Youtube video |
| Lab 2: Exploring Data for Bias | how to measure bias in data and apply various measures of bias quantification (including accuracy difference and difference in proportion of labels) ||
| Lesson 3: Fairness Criteria | different approaches to define fairness mathematically in ML, and discuss the trade-offs between ML performance and fairness | TODO: add Youtube video |
| Lab 3: Implementing a DI Remover | fairness criteria by practicing how to quantify disparate impact (DI) and how to implement basic DI removers ||
| Lesson 4: Bias Mitigation in Preprocessing, Model Training, and Postprocessing | how to perform bias mitigation during preprocessing, model training, and postprocessing | TODO: add Youtube video |
| Lab 4: Bias Mitigation during Preprocessing, Training, and Postprocessing | bias mitigation during preprocessing, model training, and postprocessing by using CI (reweighted), DPL, fairness penalty terms, equalized odds, and ROC/calibration curves ||

### Deep Learning for Text and Image Data
#### Module 1: Neural networks


| Lesson | Topic & Content | Associated videos |
|--------|----------------|-------------------|
| Lesson 0: Introduction to Deep Learning on Text and Images || TODO: add Youtube video |
| Lesson 1: Introduction to Neural Networks: Layers and Activations || TODO: add Youtube video |
| Lab 1: Getting Started with PyTorch |||
| Lesson 2: How Neural Networks Learn || TODO: add Youtube video |
| Lab 2: Creating a Multilayer Perception and Dropout Layers |||
| Lesson 3: First Examples of Neural Networks || TODO: add Youtube video |
| Lab 3: Building an End-to-End Neural Network Solution |||
| Lesson 4: Neural Network Engineering || TODO: add Youtube video |
| Lab 4: Introducing CNNs |||
| Lab 4: Refining Models by Using AutoGluon |||
| Lab 5: Using Advanced AutoGluon Techniques |||

#### Module 2: Modeling Text Data
| Lesson | Topic & Content | Associated videos |
|--------|----------------|-------------------|
| Lesson 1: Challenges of Textual Data and Domains of NLP || TODO: add Youtube video |
| Lesson 2: Processing Text || TODO: add Youtube video |
| Lab 1: Processing Text |||
| Lab 2: Using the BoW Method || TODO: add Youtube video |
| Lesson 3: Word Embeddings |||
| Lab 3: Using GloVe Embeddings |||
| Lesson 4: Recurrent Neural Networks || TODO: add Youtube video |
| Lesson 5: RNN Example with a Practical Dataset |||
| Lab 4: Introducing RNNs || TODO: add Youtube video |
| Lesson 6: Transformers |||
| Lab 5: Fine-Tuning BERT || TODO: add Youtube video |
| Lab 6: Using Ensemble Learning |||

#### Module 3: Computer Vision
| Lesson | Topic & Content | Associated videos |
|--------|----------------|-------------------|
| Lesson 1: How Are Images Stored in a Computer? || TODO: add Youtube video |
| Lab 1: Reading Image Data to Find Descriptors and Create Plots |||
| Lesson 2: The Concept of Convolution || TODO: add Youtube video |
| Lesson 3: Convolutional Neural Networks |||
| Lab 2: Using a CNN for Basic Image Operations || TODO: add Youtube video |
| Lab 3: Implementing a CNN by Using PyTorch |||
| Lesson 4: ResNet: The Trade-Offs of Depth and Model Performance || TODO: add Youtube video |
| Lab 4: Using Residual Layers |||
| Lesson 5: Modern Architectures |||
| Lab 5: Using the ConvNeXt Model |||
| Lesson 6: Transfer Learning |||
| Lab 6: Fine-Tuning ConvNeXt |||


To view all videos access the [Machine Learning playlist](TODO: ADD LINK HERE).

## Prerequisites for Machine Learning Through Application

To get the most out of this content, you should have:
- Basic understanding of Python programming
- AWS account with appropriate permissions
- Familiarity with Jupyter notebooks

## Prerequisites for Deep Learning for Text and Image Data
This topic requires a strong foundation in IT concepts and skills. To ensure success in this topic, students are strongly recommended to have the following:
- Understanding of linear algebra
- Understanding of statistical probability
- Knowledge of a programming language (such as Python, C++, or Java)
- Completed the Machine Learning University (MLU) Machine Learning through Application or have equivalent experience

## Getting Started

1. Clone this repository:
```bash
git clone [repository-url]
```

2. Environment Requirements:
- Python 3.10+
- Jupyter Notebook environment
- AWS CLI configured with appropriate credentials
- Required Python packages (specified within the notebooks)

1. Start Learning:
- Begin with the presentation materials in the Lessons folder
- Follow along with the corresponding lab notebooks
- Each notebook is self-contained with all necessary instructions and explanations

**Note**: The Jupyter notebooks will run anywhere you have Jupyter correctly configured, however the notebooks in this repo are designed to run on Amazon SageMaker using the `conda_python3` kernel.

## Additional Resources

- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/)

## Support

If you need help or have questions:
- Open an issue in this repository

## License

This material is licensed under [appropriate license]. See the [LICENSE](LICENSE) file for details.

## Contributing
If you have questions, comments, suggestions, etc. please feel free to cut tickets in this repo.

Also, please refer to the [CONTRIBUTING](CONTRIBUTING.md) document for further details on contributing to this repository.

---

We hope you enjoy learning about Machine Learning! Happy learning! 🚀
