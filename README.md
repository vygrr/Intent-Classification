# Intent Classification

## Course: NLP (Semester 6) - Pillai College of Engineering

## Project Overview:
This project is part of the Natural Language Processing (NLP) course for Semester 6 students at Pillai College of Engineering. The project focuses on Intent Classification, where we apply various Machine Learning (ML), Deep Learning (DL), and Transformer-based Language Models to categorize user queries into predefined intent categories. This implementation involves exploring techniques like text preprocessing, feature extraction, model training, and evaluating the models for their effectiveness in classifying user intents.

You can learn more about the college by visiting the official website of [Pillai College of Engineering](https://www.pillai.edu.in/).

## Acknowledgements:
We would like to express our sincere gratitude to the following individuals:

**Theory Faculty:**  
- Dhiraj Amin  
- Sharvari Govilkar  

**Lab Faculty:**  
- Dhiraj Amin  
- Neha Ashok  
- Shubhangi Chavan  

Their guidance and support have been invaluable throughout this project.

---

## Project Title:
Intent Classification using Natural Language Processing

## Project Abstract:
The Intent Classification project aims to categorize user queries into different intent classes such as order cancellation, shipping inquiries, product information, etc. This task involves applying Machine Learning, Deep Learning, and Language Models to accurately identify user intents from text input. The project explores different algorithms, including traditional machine learning techniques, deep learning models, and state-of-the-art pre-trained language models. The goal is to evaluate the performance of each approach and select the best-performing model for intent classification.

---

## Algorithms Used:

### Machine Learning Algorithms:
- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest Classifier  

### Deep Learning Algorithms:
- Convolutional Neural Networks (CNN)  
- Long Short-Term Memory (LSTM)  
- Bidirectional LSTM (BiLSTM)  
- Combined CNN-BiLSTM  

### Language Models:
- BERT (Bidirectional Encoder Representations from Transformers)  
- RoBERTa (Robustly Optimized BERT Pre-training Approach)  

---

## Comparative Analysis:
The comparative analysis of different models highlights their effectiveness in classifying user intents. The following tables summarize the performance metrics:

### Task 1: Machine Learning Models
| No. | Model Name            | Feature      | Precision | Recall | F1 Score | Accuracy |
|-----|-----------------------|--------------|-----------|--------|----------|----------|
| 0   | Random Forest         | NLP Features | 0.06      | 0.06   | 0.06     | 0.0617   |
| 1   | SVM                   | NLP Features | 0.04      | 0.08   | 0.05     | 0.0777   |
| 2   | Logistic Regression   | NLP Features | 0.03      | 0.08   | 0.04     | 0.0737   |
| 3   | Random Forest         | BoW          | 0.87      | 0.87   | 0.86     | 0.8619   |
| 4   | SVM                   | BoW          | 0.96      | 0.96   | 0.96     | 0.9558   |
| 5   | Logistic Regression   | BoW          | 0.97      | 0.97   | 0.96     | 0.9638   |
| 6   | Random Forest         | TFIDF        | 0.83      | 0.83   | 0.82     | 0.8231   |
| 7   | SVM                   | TFIDF        | 0.95      | 0.95   | 0.95     | 0.9477   |
| 8   | Logistic Regression   | TFIDF        | 0.94      | 0.94   | 0.93     | 0.9330   |
| 9   | Random Forest         | FastText     | 0.79      | 0.80   | 0.79     | 0.7869   |
| 10  | SVM                   | FastText     | 0.00      | 0.04   | 0.00     | 0.0255   |
| 11  | Logistic Regression   | FastText     | 0.40      | 0.21   | 0.16     | 0.1743   |
| 12  | Random Forest         | Combined     | 0.82      | 0.82   | 0.82     | 0.8204   |
| 13  | SVM                   | Combined     | 0.94      | 0.95   | 0.94     | 0.9437   |
| 14  | Logistic Regression   | Combined     | 0.91      | 0.91   | 0.91     | 0.9115   |

### Task 2: Deep Learning Models
| No. | Model Name   | Feature  | Precision | Recall | F1 Score | Accuracy |
|-----|-------------|----------|-----------|--------|----------|----------|
| 1   | CNN         | BoW      | 0.96      | 0.96   | 0.96     | 0.9577   |
| 2   | LSTM        | BoW      | 0.00      | 0.04   | 0.00     | 0.0221   |
| 3   | BiLSTM      | BoW      | 0.04      | 0.06   | 0.03     | 0.0523   |
| 4   | CNN-BiLSTM  | BoW      | 0.10      | 0.11   | 0.06     | 0.1107   |
| 5   | CNN         | TF-IDF   | 0.00      | 0.04   | 0.00     | 0.0221   |
| 6   | LSTM        | TF-IDF   | 0.00      | 0.04   | 0.00     | 0.0241   |
| 7   | BiLSTM      | TF-IDF   | 0.00      | 0.04   | 0.00     | 0.0241   |
| 8   | CNN-BiLSTM  | TF-IDF   | 0.00      | 0.04   | 0.00     | 0.0241   |
| 9   | CNN         | FastText | 0.13      | 0.13   | 0.10     | 0.1328   |
| 10  | LSTM        | FastText | 0.00      | 0.04   | 0.00     | 0.0241   |
| 11  | BiLSTM      | FastText | 0.12      | 0.12   | 0.09     | 0.1187   |
| 12  | CNN-BiLSTM  | FastText | 0.07      | 0.11   | 0.08     | 0.1107   |

### Task 3: Transformer Models
| No. | Model Name | Precision | Recall | F1 Score | Accuracy | MCC   |
|-----|------------|-----------|--------|----------|----------|-------|
| 1   | BERT       | 0.97      | 0.97   | 0.97     | 0.97     | 0.969 |
| 2   | RoBERTa    | 0.99      | 0.99   | 0.99     | 0.99     | 0.989 |

---

## Conclusion:
This Intent Classification project demonstrates the potential of Machine Learning, Deep Learning, and Language Models for text classification tasks, particularly for categorizing user intents. The comparative analysis reveals that:  
- **BoW-based Logistic Regression and SVM** achieve high performance (~96% accuracy) among traditional ML models.  
- **Deep Learning models (CNN with BoW)** perform comparably to ML models, but others (LSTM, BiLSTM) struggle, likely due to insufficient data or tuning.  
- **Transformer models (RoBERTa, BERT)** outperform all others, with RoBERTa achieving **99% accuracy**, highlighting their superiority in capturing contextual semantics.  
