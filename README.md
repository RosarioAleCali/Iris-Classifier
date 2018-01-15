Iris Classifier is a simple machine learning application used to distinguish the species of some iris flowers.
The species that this application can identify are *setosa*, *versicolor*, and *virginica* (classes).
The charateristics used to distinguish the species are the length and width of the petals and sepals.
Everything is measured in centimeters.

This is an example of a *classification* problem where a supervised learning algorithm is used to make a prediction.

The data used in this application is the Iris dataset included in scikit-learn in the datasets module.
The data provided by the dataset is a _Bunch_ object, which is very similar to a dictionary (key and value pairs).
The dataset contains 150 samples containing the required features.
112 samples for training. 38 samples for testing.

The application uses a *k*-nearest neighbors classifier for its machine learning model.

This is simply an adaptation from a code example from the book Introduction to Machine Learning with Python
by Andreas C. Müller & Sarah Guido. This application also uses their library, called mglearn, which takes care of plotting and data loading.