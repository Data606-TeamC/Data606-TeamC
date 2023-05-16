# CUSTOMER REVIEW CLASSIFICATION USING NATURAL LANGUAGE PROCESSING
## DATA606 Capstone Data Science
### Contributors:
* Bhanu Harish Surisetti
* Jessie Caroline Merugu
* Sreeja Pendota

### Abstract:
The objective of the project is to use Natural Language Processing and Machine Learning techniques to predict customer satisfaction with products based on their reviews. 
* Predict customer satisfaction level (1 – 5 Rating) with products based on their reviews. 
* Predict the fake reviews based on the review text.

The dataset contains 71,044 rows and 25 columns, including the product ID, brand, category, review text, and whether the reviewer purchased the product or not. The project will use various classifiers such as Logistic Regression, Decision Tree, K Nearest Neighbor, and Support Vector Machine to predict the outcome, with the maximum accuracy obtained so far being 70%. The project also involves data preprocessing to remove null and duplicate rows, and exploratory data analysis (EDA) to analyze the correlation between review rating and the number of words used in the review text, as well as to visualize which words are used most and check if there is any regional impact on the reviews provided. The project also plans to use future classifiers such as Random Forest, Decision Tree classifier, KNN, and Logistic regression to improve accuracy. Overall, the project seeks to use NLP and machine learning to better understand customer satisfaction and to identify factors that influence it.

The active learning process takes as input a set of labeled examples, as well as a larger set of unlabeled examples, and produces a classifier and a relatively small set of newly labeled data. Active learning has been successfully applied to a number of natural language processing tasks, such as, information extraction, named entity recognition, text categorization, part-of-speech tagging, parsing, and word sense disambiguation. 

With the advent of neural network models for NLP, salient features were learned jointly with the training of the model itself, and hence focus shifted to architecture engineering, where inductive bias was rather provided through the design of a suitable network architecture conducive to learning such features.

The process of reducing a word to its base form is called lemmatization. Unlike stemming, which only removes the last few characters of a word, lemmatization takes into account the context of the word and converts it to its most meaningful base form. As a result, stemming often results in incorrect meanings and spelling errors.

A method called stemming is utilized to shorten an inflected word to its stem, as illustrated by the words "programming," "programmer," and "programs," which can all be reduced to the root word stem "program."
Stemming and lemmatization are methods used by search engines and chatbots to analyze the meaning behind a word. Stemming uses the stem of the word, while lemmatization uses the context in which the word is being used.

Lemmatization has higher accuracy than stemming. Lemmatization is preferred for context analysis, whereas stemming is recommended when the context is not important.

Precision: It measures the proportion of correctly predicted instances for each class. Higher precision values indicate a lower false positive rate. Recall: It calculates the proportion of correctly predicted instances out of the actual instances for each class. Higher recall values indicate a lower false negative rate. F1-score: It is the harmonic mean of precision and recall, providing a balanced measure of the model's performance. Support: It represents the number of instances in each class. The summary of the overall model performance is as follows:

Accuracy: The overall accuracy of the model on the dataset is 72%. Macro average: The average performance across all classes, giving equal weight to each class. Weighted average: The average performance, weighted by the number of instances in each class. Based on the provided metrics, the model performs well in predicting class 5, with high precision, recall, and F1 score. However, it struggles with classes 2 and 3, as indicated by low precision, recall, and F1-score values.

#### Oversampling - SMOTE

Oversampling is a technique used to address the class imbalance in machine learning datasets by increasing the number of instances in the minority class(es). It helps to mitigate bias and improve the performance of models on underrepresented classes.

SMOTE is an oversampling technique that creates synthetic instances of the minority class to address class imbalance. By generating new samples, SMOTE helps to balance the dataset, improve model performance, and address bias in machine learning tasks.

Under sampling data
Undersampling data is a technique used to address class imbalance by reducing the number of instances in the majority class(es). It helps prevent bias in model predictions and allows for better learning of patterns from the minority class. However, careful consideration should be given to the potential loss of information and the overall impact on model performance.

Random Under Sampler is an undersampling technique that randomly selects instances from the majority class to reduce its size and balance the class distribution. It helps mitigate bias in models caused by class imbalance and allows for improved focus on the minority class during training.

lr_model: This is a logistic regression model with the solver parameter set to 'liblinear'. Logistic regression is a linear model used for binary classification tasks.

t_model: This is a decision tree classifier. Decision trees are a non-linear model that makes predictions based on a series of binary decisions at each node of the tree.

rf_model: This is a random forest classifier. Random forests are an ensemble method that combines multiple decision trees to make predictions. Each tree is trained on a different subset of the data, and the final prediction is based on a majority vote of the individual tree predictions.

svc_model: This is a support vector classifier (SVC) with a radial basis function (RBF) kernel. SVC is a powerful and versatile classification model that can handle both linear and non-linear decision boundaries.


### Rating classification
#### Under Sampling: 

The classification model trained using under-sampling techniques showed modest performance on the test data, with an overall accuracy of 44%. The model exhibited varying precision, recall, and F1 scores across different classes, indicating different levels of effectiveness for each class. While some classes achieved higher scores, others performed relatively poorer. The macro average suggested a balanced performance across classes, but the weighted average F1-score of 0.46 indicated moderate overall effectiveness. Further improvements are required to enhance the model's performance on unseen data.

#### Oversampling: 

The logistic regression model trained using oversampling techniques demonstrated good overall performance, with high precision, recall, and F1 scores for most classes. Class 1, 2, and 3 showed particularly accurate identification, while classes 4 and 5 had slightly lower scores, suggesting some challenges in classification. The model achieved an accuracy of 89%, indicating a high rate of correct predictions. Both the macro and weighted averages of precision, recall and F1-score were consistent, indicating a balanced performance across all classes. Overall, the logistic regression model exhibited effective classification ability, with minor variations in performance for different classes.

### Conclusion

#### Fake review detection
Based on the results, the sentiment analysis component achieved a precision of 0.95 for identifying genuine reviews and a precision of 0.50 for identifying fake reviews. The recall was 0.93 for genuine reviews and 0.58 for fake reviews. The F1 scores were 0.94 for genuine reviews and 0.54 for fake reviews. This indicates that the sentiment analysis component shows strong performance in identifying genuine reviews, while its performance in identifying fake reviews is relatively weaker.

Moving on to the fake review detection models, the CNN model achieved a test accuracy of 89%, demonstrating reasonable performance in predicting the target variable of fake review detection. Both the LSTM and CNN models achieved similar high accuracies, suggesting that both architectures are suitable for this task.

In conclusion, the sentiment analysis component provides valuable insights into the sentiment of reviews, but it may require further improvement to enhance its ability to detect fake reviews. On the other hand, the CNN model shows promise in accurately identifying fake reviews, highlighting the effectiveness of neural network models for this task. Further experimentation and refinement could potentially improve the performance of both the sentiment analysis and fake review detection models.

### Future Work:
Conduct a systematic search for optimal hyperparameter values using techniques like grid search, random search, or Bayesian optimization to maximize model performance.
Explore data augmentation techniques to generate synthetic samples or introduce variations to address class imbalance and enhance the models' ability to generalize to unseen data.
Consider utilizing ensemble methods such as bagging, boosting, or stacking to combine predictions from multiple models and improve overall performance. This helps mitigate biases and errors from individual models and increases accuracy.

### Data Source:
* https://www.kaggle.com/code/duttadebadri/detailed-nlp-project-prediction-visualization/data

### Reference:
* Olsson, Fredrik. "A literature survey of active machine learning in the context of natural language processing." (2009).
  https://www.diva-portal.org/smash/record.jsf?dswid=-8182&pid=diva2%3A1042586 
* Liu, P., Yuan, W., Fu, J., Jiang, Z., Hayashi, H., & Neubig, G. (2023). Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language     processing. ACM Computing Surveys, 55(9), 1-35.
  https://dl.acm.org/doi/full/10.1145/3560815
  

