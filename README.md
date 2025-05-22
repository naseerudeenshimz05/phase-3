```markdown
# Fake News Detection System

**Student Name:** S.Naseeruddeen
**Register Number:** 510623104073
**Institution:** C. ABDUL HAKEEM COLLEGE OF ENGINEERING AND TECHNOLOGY
**Department:** COMPUTER SCIENCE OF ENGINEERING
**Date of Submission:** 09/05/2025
**Github Repository Link:** (Please add your GitHub repository link here)

## 1. Abstract
This project tackles the critical issue of fake news detection in online articles. The primary objective is to develop a classification model that accurately distinguishes between reliable and unreliable news articles. The approach involves collecting a dataset of news articles, preprocessing the text data using Natural Language Processing (NLP) techniques, extracting relevant features, and training machine learning models. We employed several models, including Logistic Regression and Random Forest, to identify patterns indicative of fake news. The results demonstrate the effectiveness of the developed model in classifying news articles with a high degree of accuracy. [cite: 3]

## 2. Problem Statement
This project addresses the pervasive issue of fake news online by developing an automated system for URL analysis and advanced Natural Language Processing. The core problem is the binary classification of news articles as either reliable or unreliable. Successfully tackling this challenge empowers users to critically evaluate online information, combats the spread of misinformation, and contributes to a more informed digital environment. This classification capability has significant societal benefits in fostering trust and enabling sound decision-making. [cite: 1, 2]

## 3. Objectives
This project aims to develop a machine learning model for accurate classification of news articles as reliable or unreliable. The primary output will be a binary classification for given article URLs. Key objectives include achieving high accuracy, precision, and recall in fake news detection. Ultimately, this work contributes to mitigating misinformation and enhancing trust in online information. [cite: 4, 5]

## 4. System Requirements

**Hardware:**
* Minimum 8GB RAM [cite: 4]
* Intel Core i5 processor or equivalent [cite: 4]
* 1GB of free storage space [cite: 4]

**Software:**
* Python 3.9 - 3.11 [cite: 4]
* Required libraries: pandas, NumPy, scikit-learn, matplotlib, seaborn, nltk, spacy, transformers, flask (A complete list should be provided in a `requirements.txt` file). [cite: 4]
* IDE: Jupyter Notebook or Google Colab [cite: 4]

## 5. Flowchart of Project Workflow
The project workflow is as follows:
1.  **Start**
2.  **Data Collection**
3.  **Data Preprocessing**
4.  **EDA (Exploratory Data Analysis)**
5.  **Feature Engineering**
6.  **Model Building**
7.  **Model Evaluation**
8.  **Deployment**
9.  **End**

*(Based on the flowchart image provided in the document [cite: 6])*

## 6. Dataset Description
* **Source:** Kaggle [cite: 6]
* **Type:** Public [cite: 6]
* **Size and structure:** 101 rows / 2 columns (initially, before further processing) [cite: 6]

## 7. Data Preprocessing
1.  **Handling Missing Values:** Missing values were addressed by removing articles with excessive missing data and imputing missing numerical features using the mean or median. [cite: 7] During web scraping, if crucial article content was absent, the scrape was discarded to maintain data integrity. [cite: 7] This ensures the model trains on complete and reliable information. [cite: 8]
2.  **Removing Duplicates:** Exact duplicate articles within the training dataset were identified and removed using pandas to prevent redundancy. [cite: 9] To optimize application performance, duplicate URL submissions will be handled via caching or prevention mechanisms. [cite: 10]
3.  **Handling Outliers:** Outliers in the training data were detected through statistical analysis and visual exploration. [cite: 11] Removal or transformation techniques were applied to mitigate their impact. [cite: 12] For scraped data, robust practices and error handling were employed. [cite: 12]
4.  **Feature Encoding and Scaling:** The target variable ("reliable"/"unreliable") was label-encoded, and nominal categorical features (e.g., article source) were one-hot encoded. [cite: 13] Numerical features were normalized or standardized as needed to ensure consistent scaling and improve model performance. [cite: 14]

## 8. Exploratory Data Analysis (EDA)
* **Feature: label (target variable)**
    * Plot: Countplot [cite: 15]
    * Explanation: Shows class distribution (reliable/unreliable). [cite: 15]
    * Insight: Reveals class balance/imbalance, impacting model evaluation. [cite: 15]
* **Feature: article length**
    * Plot: Histogram, Boxplot [cite: 16]
    * Explanation: Shows distribution and outliers of article lengths. [cite: 16]
    * Insight: Potential length differences between reliable/unreliable articles; outlier handling. [cite: 16, 17]
* **Feature: source**
    * Plot: Countplot, Bar chart [cite: 17]
    * Explanation: Shows article count per news source. [cite: 17]
    * Insight: Source influence on reliability; data sufficiency per source. [cite: 17, 18]
* **Feature: sentiment score**
    * Plot: Histogram, Boxplot [cite: 18]
    * Explanation: Shows distribution and outliers of sentiment scores. [cite: 18]
    * Insight: Sentiment tendencies of reliable/unreliable articles. [cite: 19]
* **Features: article length and sentiment score**
    * Plot: Scatter plot [cite: 19]
    * Explanation: Shows relationship between article length and sentiment. [cite: 19]
    * Insight: Correlation between length and sentiment. [cite: 20]

## 9. Feature Engineering
* **New Feature Creation:**
    * The code employs TF-IDF vectorization to generate new numerical features from the 'clean_text' column. Each word becomes a feature, with its TF-IDF score representing its importance. [cite: 21]
* **Feature Selection:**
    * The code implicitly reduces features using `TfidfVectorizer` parameters:
        * `stop_words='english'` removes common, less informative words. [cite: 22]
        * `max_df=0.7` ignores words appearing in over 70% of documents (Note: source states `max_df=0`[cite: 23], which seems like a typo and likely meant `max_df=0.7` as used in typical scenarios and implied by the explanation of ignoring words in over 70% of documents. The provided code snippet uses `max_df=0.7` [cite: 52]).
    * Initial column selection (e.g., `df[['title', 'text']]`) also acts as feature selection. [cite: 24]
* **Transformation Techniques:**
    * Text is cleaned by lowercasing, removing URLs, mentions, punctuation, and numbers. [cite: 24, 25]
    * Cleaned text is transformed into numerical vectors using TF-IDF. [cite: 25]
* **Impact of Features on Model:**
    * Cleaning text ensures consistency and focuses the model on relevant words, improving generalization. [cite: 27, 28]
    * `TfidfVectorizer` transforms cleaned text into numerical TF-IDF vectors, which models require. [cite: 29] TF-IDF weighs word importance, giving more weight to discriminative terms. [cite: 30]
    * These techniques create a structured, numerical representation allowing the Logistic Regression model to learn effectively from text data. [cite: 31, 32] Cleaning reduces noise, leading to more accurate pattern recognition. [cite: 33] TF-IDF highlights important words, enabling better distinction between fake and real news. [cite: 34]
    * Collectively, these enhance the model's ability to classify news articles. [cite: 35, 36]

## 10. Model Building
* **Models Tried:**
    * Logistic Regression (Baseline) [cite: 37]
    * (The abstract also mentions Random Forest [cite: 3])
* **Explanation of Model Choices:**
    * **Logistic Regression (Baseline):** Chosen due to its simplicity and effectiveness in binary classification, especially with text data. [cite: 37, 38] It's a linear model, provides a good starting point for comparison, and offers some interpretability via feature coefficients. [cite: 38, 39]
* **Classification Report (for Logistic Regression as per document):**
    * The provided report indicates:
        * Precision for class 0: 0.10, Recall for class 0: 0.11, F1-score for class 0: 0.10 [cite: 40]
        * Precision for class 1: 0.11, Recall for class 1: 0.10, F1-score for class 1: 0.10 [cite: 40]
        * Accuracy: 0.10 [cite: 40]
    *(Note: The code snippet shows a data preparation step where 'real' data is simulated by duplicating 'fake' data and assigning a different label. This would logically lead to poor model performance if not a placeholder dataset.)*

## 11. Model Evaluation
* **Classification Report:** The evaluation presented the same classification report as in the Model Building section. [cite: 44]
    * Overall accuracy: 0.10 [cite: 44]
    * Macro avg: precision 0.10, recall 0.10, f1-score 0.10 [cite: 42, 44]
    * Weighted avg: precision 0.10, recall 0.10, f1-score 0.10 [cite: 42, 44]
* **Confusion Matrix:** A confusion matrix was generated to visualize the actual vs. predicted labels. [cite: 45, 81] The values shown are 2 for True Negatives, 17 for False Positives, 19 for False Negatives, and 2 for True Positives. [cite: 45]

## 12. Deployment
* **Deployment Method:** Gradio Interface [cite: 45]
* **Public Link:** https://4c7f664c6f905007d9.gradio.live [cite: 45] (Note: This link may be temporary or expired)
* **Output Screenshot:** The document includes a screenshot of the Gradio interface titled "Fake News Detection" with input for "news_text" and an "output" field. [cite: 46, 47]
    * A sample prediction is shown: `news_text = By now, everyone knows that disgraced...` with `Output = Real News`. [cite: 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]

## 13. Source Code Overview
The project utilizes Python with the following key libraries and steps:
* **Imports:** `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `string`, `re`, `sklearn.model_selection.train_test_split`, `sklearn.feature_extraction.text.TfidfVectorizer`, `sklearn.linear_model.LogisticRegression`, `sklearn.metrics`. [cite: 57]
* **Data Collection:** Loads data using `pd.read_csv('Fakenews_data.csv')`. [cite: 57]
* **Data Preprocessing:** Selects 'title' and 'text' columns, combines them, and includes a text cleaning function (`clean_text`) involving lowercasing, and removal of URLs, mentions, punctuation, and numbers. [cite: 57, 58]
* **Feature Engineering:** Applies `TfidfVectorizer` to the cleaned text. [cite: 58] (The code also includes a section to simulate 'real' labels by duplicating the dataset and assigning label 0 [cite: 58]).
* **Model Building & Training:** Splits data into training and testing sets and trains a `LogisticRegression` model. [cite: 58]
* **Evaluation:** Generates a classification report and a confusion matrix. [cite: 59]
* **Gradio Deployment:** Includes code to set up and launch a Gradio interface for prediction. [cite: 60, 61, 62]

## 14. Future Scope
* Implement continuous learning mechanisms to adapt to evolving misinformation tactics. [cite: 63]
* Expand analysis to include multimodal data (images, videos). [cite: 64]
* Enhance model interpretability to provide more transparent explanations for predictions. [cite: 65]

## 15. Team Members and Roles
* **Data cleaning:** Mohammed Sharuk. I [cite: 66]
* **EDA:** Mubarak basha [cite: 66]
* **Feature engineering:** Rishi kumar baskar. R [cite: 66, 67]
* **Model development:** B. Mohammed Sakhee [cite: 67]
* **Documentation and reporting:** Naseerudin [cite: 67]

## 16. How to Run
1.  Ensure you have Python (version 3.9 - 3.11) installed. [cite: 4]
2.  Clone the repository.
3.  Install the required libraries. It is recommended to use a virtual environment:
    ```bash
    pip install -r requirements.txt
    ```
    (The `requirements.txt` file should list all dependencies like pandas, numpy, scikit-learn, matplotlib, seaborn, nltk, spacy, transformers, flask, gradio).
4.  Run the main project script or Jupyter Notebook (e.g., `app.py` or `notebook.ipynb`) using an IDE like Jupyter Notebook or Google Colab. [cite: 4]
    ```bash
    python your_main_script.py
    ```
    Or open and run cells in the Jupyter Notebook.

```
