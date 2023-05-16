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

The dataset used consists of 71,044 rows and 25 columns, containing information such as product ID, brand, category, review text, and whether the reviewer purchased the product. Various classifiers, including Logistic Regression, Decision Tree, K Nearest Neighbor, and Support Vector Machine, are employed to predict customer satisfaction levels, achieving a maximum accuracy of 70%. The project involves data preprocessing, removing null and duplicate rows, and performing exploratory data analysis (EDA) to analyze the correlation between review ratings and the number of words used. Additionally, visualization techniques are utilized to identify commonly used words and potential regional impacts on reviews.

To further improve accuracy, future classifiers such as Random Forest, Decision Tree classifier, KNN, and Logistic Regression are planned to be implemented. The project's goal is to leverage NLP and machine learning to gain deeper insights into customer satisfaction and identify key factors influencing it.

### Active Learning in NLP and Neural Network Models

The project explores the utilization of active learning, a technique that involves training a classifier using both labeled and unlabeled examples. Active learning has been successfully applied to various NLP tasks, including information extraction, named entity recognition, text categorization, part-of-speech tagging, parsing, and word sense disambiguation.

With the advent of neural network models in NLP, focus shifted towards architecture engineering, where the model's design facilitated the learning of salient features. Neural networks allowed for joint learning of features and model parameters, enhancing performance in NLP tasks.

### Lemmatization and Stemming for Word Analysis

Lemmatization and stemming are techniques employed in search engines and chatbots to analyze the meaning behind words. Lemmatization converts words to their most meaningful base form, considering the word's context, while stemming shortens words to their stem form.

Lemmatization exhibits higher accuracy compared to stemming as it maintains the correct meaning and avoids spelling errors. However, stemming can be preferred when the context is not crucial.

### Model Evaluation and Performance Metrics

Several performance metrics are utilized to evaluate classification models. Precision measures the proportion of correctly predicted instances for each class, while recall calculates the proportion of correctly predicted instances out of the actual instances for each class. The F1-score is the harmonic mean of precision and recall, providing a balanced measure of the model's performance. Support represents the number of instances in each class.

The overall model performance on the dataset is summarized as follows: Accuracy: 72%. Macro average: Balanced performance across classes. Weighted average: Performance weighted by the number of instances in each class. Class 5 prediction shows high precision, recall, and F1-score, while classes 2 and 3 exhibit lower values.

### Addressing Class Imbalance with Oversampling and Undersampling

Class imbalance in machine learning datasets is addressed using oversampling and undersampling techniques. Oversampling increases the number of instances in the minority class(es), mitigating bias and improving model performance. SMOTE is an oversampling technique that generates synthetic instances of the minority class to balance the dataset.

Undersampling reduces the number of instances in the majority class(es) to address class imbalance. It helps prevent bias in predictions and allows better pattern learning from the minority class. Random Under Sampler is an undersampling technique that randomly selects instances from the majority class.

### Classification and Results

The classification models trained using undersampling and oversampling techniques yield different performance outcomes. The undersampling model achieves an overall accuracy of 44%, exhibiting varying precision, recall, and F1-scores across classes. The oversampling model, specifically logistic regression, demonstrates good overall performance, with high precision, recall, and F1-scores for most classes. However, some challenges are observed in classifying certain classes.

### Conclusion and Future Work

The sentiment analysis component provides valuable insights into review sentiments, but further improvement is required to enhance fake review detection. The CNN model exhibits promise in accurately identifying fake reviews, showcasing the effectiveness of neural network models for this task. Future work includes optimizing hyperparameters, exploring data augmentation techniques, and leveraging ensemble methods to improve overall model performance and mitigate biases.

Overall, this project enhances customer satisfaction prediction, provides insights into review sentiments, and improves fake review detection using NLP and machine learning techniques.

### Data Source:
* https://www.kaggle.com/code/duttadebadri/detailed-nlp-project-prediction-visualization/data

### Reference:
* Olsson, Fredrik. "A literature survey of active machine learning in the context of natural language processing." (2009).
  https://www.diva-portal.org/smash/record.jsf?dswid=-8182&pid=diva2%3A1042586 
* Liu, P., Yuan, W., Fu, J., Jiang, Z., Hayashi, H., & Neubig, G. (2023). Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language     processing. ACM Computing Surveys, 55(9), 1-35.
  https://dl.acm.org/doi/full/10.1145/3560815
  

