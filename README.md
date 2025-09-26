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

Each lesson and its corresponding lab(s) are designed to be completed sequentially, building upon concepts from previous modules and lessons. All code examples are thoroughly documented and include explanatory comments to facilitate learning.

Note: Make sure to review the lesson material before attempting the associated lab exercises for the best learning experience.
### Lab environment
All labs were last tested in Amazon SageMaker Studio in a JupyterLab space using a `ml.g4dn.large` instance running the `SageMaker Distribution 3.0.1` Image

## Content
### Machine Learning through Application

To view all MLTA content, videos access the [Machine Learning Through Application](https://www.youtube.com/playlist?list=PL8P_Z6C4GcuU0qt4mBSm-XmWcYX8YWXGO).

#### Module 1: What is ML?
| Lesson | Topic & Content | Associated Videos |
|:--|:--|:--|
|Course Content Summary | Overview of Machine Learning Through Application (MLTA) course content, Modules 1-3 | [https://www.youtube.com/watch?v=bQ8bnylgSYY](https://www.youtube.com/watch?v=bQ8bnylgSYY) |
Lesson 0: Introduction to ML | Introduced to the material, starting with a definition of ML along with a few examples of an ML model | [https://www.youtube.com/watch?v=Cdczxan0gas](https://www.youtube.com/watch?v=Cdczxan0gas)|
| Lesson 1: Jupyter and SageMaker | Fundamentals of the Jupyter Notebook application and Amazon SageMaker | [https://www.youtube.com/watch?v=H2lz3DHXrg0](https://www.youtube.com/watch?v=H2lz3DHXrg0) |
| Lab 1: Getting Started with Jupyter Notebooks | Practice using SageMaker and gain experience using the core UI components ||
| Lab 2: Extending Your Jupyter Notebook Understanding | Practice using Jupyter notebooks more efficiently ||
| Lesson 2: Types of ML and Aspects of Successful ML Problems | Common types of ML problems and the characteristics of successful ML problems | [https://www.youtube.com/watch?v=xIBSLuXCjDQ](https://www.youtube.com/watch?v=xIBSLuXCjDQ) |
| Lesson 3: Examples of ML Applications and Why They Are Successful | Examples of successful ML use cases and examine how these use cases have characteristics that help to make them successful | [https://www.youtube.com/watch?v=yd9SLuAVIbc](https://www.youtube.com/watch?v=yd9SLuAVIbc) |
| Lesson 4: The ML Lifecycle, and Overfitting and Underfitting | The fundamental concepts of overfitting, underfitting, and the ML lifecycle | [https://www.youtube.com/watch?v=O6aSxyxU7Xk](https://www.youtube.com/watch?v=O6aSxyxU7Xk) |
| Lesson 5: AutoML and AutoGluon | AutoML concepts an use of AutoGluon, an open-source AutoML library, to automate repetitive portions of the ML lifecycle | [https://www.youtube.com/watch?v=dts8nmeca7M](https://www.youtube.com/watch?v=dts8nmeca7M) |
| Lab 3: Getting Started with AutoGluon | Walk through the process to build a basic ML model by using AutoGluon to train a model on a dataset ||
| Lab 4: Refining Models by Using AutoGluon | Continue from the previous lab, and learn how to assess the performance of the models that AutoGluon trains and how to make predictions by using the best model ||
| Lab 5: Using Advanced AutoGluon Techniques | Continue from the previous lab by using a full dataset and a time limit to train the model, and by making predictions with the new model. The core purpose of this lab is to emphasize the importance of data in the process of constructing models ||

#### Module 2: Core ML Concepts for Tabular Data
| Lesson | Topic & Content | Associated Videos |
|:--|:--|:--|
| Lesson 1: Exploratory Data Analysis | Basics of exploratory data analysis (EDA) so that they can understand why it is important to clean a dataset before using it to create a model | [https://www.youtube.com/watch?v=6lmSWrpU__8](https://www.youtube.com/watch?v=6lmSWrpU__8) |
| Lab 1: Performing EDA for Categorical Variables | Basic steps of EDA of categorical data, and perform initial data investigations to discover patterns, spot anomalies, and look for insights to inform later ML modeling choices ||
| Lab 2: Performing EDA for Numerical Variables | Basic steps of EDA of numerical data, and perform initial data investigations to discover patterns, spot anomalies, and look for insights to inform later ML modeling choices ||
| Lesson 2: Basic Feature Engineering | Basic feature engineering methods for processing categorical features, preprocessing text features, and vectorizing text features | [https://www.youtube.com/watch?v=JA9-p7pOFX0](https://www.youtube.com/watch?v=JA9-p7pOFX0) |
| Lab 3: Performing Basic Feature Engineering | Common techniques that are used to transform numerical features, encode categorical features, and vectorize processed text features ||
| Lesson 3: Tree-Based Models | Introduction to decision trees, the Iterative Dichotomiser (ID3) algorithm, and information gain or impurity | [https://www.youtube.com/watch?v=ueDGJcr9hn0](https://www.youtube.com/watch?v=ueDGJcr9hn0) |
| Lesson 4: Optimization, Regression Models, and Regularization | How to use gradient descent to optimize ML models, when to use linear and logistic regression, and when to use regularization | [https://www.youtube.com/watch?v=f70xHuPxKEY](https://www.youtube.com/watch?v=f70xHuPxKEY) |
| Lab 4: Using Logistic Regression | Build a logistic regression model to predict a field of a dataset. They also look at how probability threshold calibration can help improve a classifier's performance ||
| Lesson 5: Hyperparameter Tuning | Hyperparameters and how to tune them by using grid search, random search, and Bayesian search | [https://www.youtube.com/watch?v=wfFjkaZBH04](https://www.youtube.com/watch?v=wfFjkaZBH04) |
| Lab 5: Using Hyperparameter Tuning | Become familiar with two main types of hyperparameter tuning: grid search and randomized search. Students will use a decision tree model to learn the basics of hyperparameter tuning that can be applied to any model ||
| Lesson 6: Ensembling | Introduction to ensembling methods. This includes bagging with random forests, boosting, and stacking | [https://www.youtube.com/watch?v=abEnmXBuxfY](https://www.youtube.com/watch?v=abEnmXBuxfY) |
| Lab 6: Using Ensemble Learning | Use of ensemble methods to create a strong model by combining the predictions of multiple weak models that were built with a given dataset and a given learning algorithm ||

#### Module 3: Responsible ML
| Lesson | Topic & Content | Associated Videos |
|:--|:--|:--|
| Lesson 1: Introduction to Fairness and Bias Mitigation in ML | Introduced to fair and responsible AI. They will see an overview of how fairness needs to be incorporated into all stages of the ML lifecycle, and see how to identify different types of bias that are present in data | [https://www.youtube.com/watch?v=MMxPX-3LEUI](https://www.youtube.com/watch?v=MMxPX-3LEUI) |
| Lab 1: Introduction to Fairness and Bias Mitigation in ML | Fairness and bias mitigation in ML by exploring different types of bias that are present in data and practice how to build various documentation sheets ||
| Lesson 2: Designing Fair Models, and Data Integrity and Analysis | Various metrics that can be used to measure bias that can be present in training data | [https://www.youtube.com/watch?v=EbGG32DwUQQ](https://www.youtube.com/watch?v=EbGG32DwUQQ) |
| Lab 2: Exploring Data for Bias | How to measure bias in data and apply various measures of bias quantification (including accuracy difference and difference in proportion of labels) ||
| Lesson 3: Fairness Criteria | Different approaches to define fairness mathematically in ML, and discuss the trade-offs between ML performance and fairness | [https://www.youtube.com/watch?v=YQ5jkl_gwBA](https://www.youtube.com/watch?v=YQ5jkl_gwBA) |
| Lab 3: Implementing a DI Remover | Fairness criteria by practicing how to quantify disparate impact (DI) and how to implement basic DI removers ||
| Lesson 4: Bias Mitigation in Preprocessing, Model Training, and Postprocessing | How to perform bias mitigation during preprocessing, model training, and postprocessing | [https://www.youtube.com/watch?v=Qd-HLS0pYk0](https://www.youtube.com/watch?v=Qd-HLS0pYk0) |
| Lab 4: Bias Mitigation during Preprocessing, Training, and Postprocessing | Bias mitigation during preprocessing, model training, and postprocessing by using CI (reweighted), DPL, fairness penalty terms, equalized odds, and ROC/calibration curves ||

### Deep Learning for Text and Image Data

To view all DLTI content, videos access the [Application of Deep Learning to Text and Image Data playlist](https://www.youtube.com/playlist?list=PL8P_Z6C4GcuU9-i94bz5WMfCZU-8s0RkY).


#### Module 1: Neural networks


| Lesson | Topic & Content | Associated videos |
|--------|----------------|-------------------|
|Course Content Summary | Overview of Application of Deep Learning to Text and Image Data (DLTI) course content, Modules 1-3 | [https://www.youtube.com/watch?v=cyUk0BXijc4](https://www.youtube.com/watch?v=cyUk0BXijc4) |
| Lesson 0: Introduction to Deep Learning on Text and Images |In this lecture, students will review what they learned in the Machine Learning through Application course.| [https://www.youtube.com/watch?v=foLERJ94cGo](https://www.youtube.com/watch?v=foLERJ94cGo) |
| Lesson 1: Introduction to Neural Networks: Layers and Activations |In this lecture, students will be introduced to neural networks. They will see how neural networks are related to deep learning and explore their first single-layer neural network (a perceptron).| [https://www.youtube.com/watch?v=36i_egTaoUw](https://www.youtube.com/watch?v=36i_egTaoUw) |
| Lab 1: Getting Started with PyTorch |In this lab, students will be introduced to PyTorch, which is a deep learning framework. Students will use PyTorch to implement a minimum viable neural network to see the different architecture components of a neural network.||
| Lesson 2: How Neural Networks Learn |In this lecture, students will review single-layer perceptrons before learning how to create multilayer perceptrons. Students will then review optimization and see how to train a neural network.| [https://www.youtube.com/watch?v=r-FzNBUgelk](https://www.youtube.com/watch?v=r-FzNBUgelk) |
| Lab 2: Creating a Multilayer Perception and Dropout Layers |In this lab, students will implement a simple neural network with multiple layers and analyze the training process. Students will then implement dropout layers to prevent overfitting of the neural network.||
| Lesson 3: First Examples of Neural Networks |In this lecture, students will see examples of neural network optimization including stochastic, full-batch, and mini-batch gradient descent. They will also discuss regularization methods including early stopping, weight decay, and dropout.| [https://www.youtube.com/watch?v=gsi7zAGiiMQ](https://www.youtube.com/watch?v=gsi7zAGiiMQ) |
| Lab 3: Building an End-to-End Neural Network Solution |In this lab, students will continue from the last lab and process text data by building an end-to-end neural network solution. The solution will incorporate all the data processing techniques that they have learned so far.||
| Lesson 4: Neural Network Engineering |In this lecture, students will learn about the role of neural network architectures and will explore the data structure for image and text data. Students will learn how weight sharing can be used to customize the design of a neural network for a particular domain.| [https://www.youtube.com/watch?v=WGP6QBI_mVo](https://www.youtube.com/watch?v=WGP6QBI_mVo) |
| Lab 4: Introducing CNNs |In this lab, students will learn how to build a simple CNN. They will train this network on an image dataset, make predictions with it, and evaluate it.||

#### Module 2: Modeling Text Data
| Lesson | Topic & Content | Associated videos |
|--------|----------------|-------------------|
  | Lesson 1: Challenges of Textual Data and Domains of NLP |In this lecture, students will learn about the challenges of textual data, be introduced to NLP, and common  tools that are used for NLP.| [https://www.youtube.com/watch?v=hynV1CZ4Q4U](https://www.youtube.com/watch?v=hynV1CZ4Q4U) |
| Lesson 2: Processing Text |In this lecture, students will learn about preprocessing and vectorization of text data. These techniques will help students to overcome the challenges that were identified in the previous lesson.| [https://www.youtube.com/watch?v=69L9L8iv3gg](https://www.youtube.com/watch?v=69L9L8iv3gg) |
| Lab 1: Processing Text |In this lab, students will review techniques to analyze and process text data, including how to make better performing and more useful models when using text data.||
| Lab 2: Using the BoW Method |In this lab, students will use the BoW method to convert text data into numerical values. These values will be used in a later lab to train a model||
| Lesson 3: Word Embeddings |In this lecture, students will learn about word embeddings and semantics in linguistics.| [https://www.youtube.com/watch?v=p0iPY34gHOw](https://www.youtube.com/watch?v=p0iPY34gHOw)|
| Lab 3: Using GloVe Embeddings |In this lab, students will see how to use word embeddings. They will learn how to represent words as numeric vectors in a high-dimensional space, use the embeddings to capture the meaning of the words, and see the relationships between words.||
| Lesson 4: Recurrent Neural Networks |In this lecture, students will learn the differences between neural networks and RNNs. They will also discuss the advantages and disadvantages of RNNs.| [https://www.youtube.com/watch?v=HZuYLB5cKG8](https://www.youtube.com/watch?v=HZuYLB5cKG8) |
| Lesson 5: RNN Example with a Practical Dataset |In this lecture, students will see a step-by-step example of implementing an RNN by using PyTorch.| [https://www.youtube.com/watch?v=7IKMPAaM6pY](https://www.youtube.com/watch?v=7IKMPAaM6pY) |
| Lab 4: Introducing RNNs |In this lab, students will learn how to use RNNs and apply them to a text classification problem.||
| Lesson 6: Transformers |In this lecture, students will learn about transformer architecture-based neural networks that power a variety of state-of-the-art NLP applications.| [https://www.youtube.com/watch?v=qxn7IAOVgUs](https://www.youtube.com/watch?v=qxn7IAOVgUs) |
| Lab 5: Fine-Tuning BERT |In this lab, students will learn about Bidirectional Encoder Representations from Transformers (BERT) and how to use BERT to process a reviews dataset.||

#### Module 3: Computer Vision
| Lesson | Topic & Content | Associated videos |
|--------|----------------|-------------------|
| Lesson 1: How Are Images Stored in a Computer? |In this lecture, students will review the different types of CV problems, learn how images are stored on computers, and see the differences between using color and grayscale images in ML.| [https://www.youtube.com/watch?v=GMnTve9Pkf8](https://www.youtube.com/watch?v=GMnTve9Pkf8) |
| Lab 1: Reading Image Data to Find Descriptors and Create Plots |In this lab, students will learn how to read image data, extract features, plot images, and manipulate images by using a CNN.||
| Lesson 2: The Concept of Convolution |In this lecture, students will learn the core concept of a convolution, which is a type of mathematical operation that is core to the design of neural networks that are applied to CV. Students will also explore different convolution examples.| [https://www.youtube.com/watch?v=gdCUfTDGt0o](https://www.youtube.com/watch?v=gdCUfTDGt0o) |
| Lesson 3: Convolutional Neural Networks |In this lecture, students will combine the concept of convolution with a neural network to create a CNN architecture that is designed for use on image data.| [https://www.youtube.com/watch?v=hehpSVeIbcw](https://www.youtube.com/watch?v=hehpSVeIbcw) |
| Lab 2: Using a CNN for Basic Image Operations |In this lab, students will strengthen their understanding of CNNs by using built-in PyTorch architectures to train a multiclass classification model.||
| Lab 3: Implementing a CNN by Using PyTorch |In this lab, students will use a CNN in PyTorch to process a real-world dataset.||
| Lesson 4: ResNet: The Trade-Offs of Depth and Model Performance |In this lecture, students will learn about the concept of residual connections in neural networks and how they allow for the training of very deep models.| [https://www.youtube.com/watch?v=4P7XBdDLM-4](https://https://www.youtube.com/watch?list=PL8P_Z6C4GcuU9-i94bz5WMfCZU-8s0RkY&v=4P7XBdDLM-4www.youtube.com/watch?v=4P7XBdDLM-4)( |
| Lab 4: Using Residual Layers |In this lab, students will compare the performance of different residual layers by plotting them over a variety of depths.||
| Lesson 5: Modern Architectures |In this lecture, students will combine various components from previous lectures to learn about three modern CNN architectures: LeNet, AlexNet, and ConvNeXt, which is the current state of the art in CNNs.| [https://www.youtube.com/watch?v=mPprZJb7eus](https://www.youtube.com/watch?v=mPprZJb7eus) |
| Lab 5: Using the ConvNeXt Model |In this lab, students will be introduced to ConvNeXt, a modern CNN that is accurate, efficient, scalable, and simple in design.||
| Lesson 6: Transfer Learning |In this lecture, students will learn the concept of transfer learning. They will apply the techniques of fine-tuning to the ConvNeXt architecture neural networks to leverage large pretrained models for tasks in CV.| [https://www.youtube.com/watch?v=_KDpdUe9orM](https://www.youtube.com/watch?v=_KDpdUe9orM) |
| Lab 6: Fine-Tuning ConvNeXt |In this lab, students will fine-tune a ConvNeXt-based model to get the best results.||


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

- [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/)

## Support

If you need help or have questions:
- Open an issue in this repository

## License

Lab materials are licensed under MIT-0 and lecture materials are licensed under CC-BY-4.0. See the [LICENSE](LICENSE) file for details.

## Contributing
If you have questions, comments, suggestions, etc. please feel free to cut tickets in this repo.

Also, please refer to the [CONTRIBUTING](CONTRIBUTING.md) document for further details on contributing to this repository.

---

We hope you enjoy learning about Machine Learning! Happy learning! 🚀
