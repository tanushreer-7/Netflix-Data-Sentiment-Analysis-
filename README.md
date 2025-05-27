# Netflix Review Sentiment Analysis

This project performs sentiment analysis on Netflix user reviews. It classifies reviews as Positive, Neutral, or Negative based on their textual content using Natural Language Processing (NLP) techniques and a Multinomial Naive Bayes classifier.

## Features

*   **Sentiment Classification:** Categorizes reviews into Positive, Neutral, or Negative.
*   **Text-Based Prediction:** Uses the review text ('content') as the primary feature for sentiment prediction.
*   **Text Preprocessing:** Includes steps like:
    *   Lowercasing
    *   Removal of punctuation and numbers
    *   Removal of common English stopwords (using NLTK)
*   **Feature Extraction:** Employs TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into numerical features.
*   **Machine Learning Model:** Utilizes a Multinomial Naive Bayes classifier, well-suited for text classification tasks.
*   **Model Evaluation:** Assesses model performance using:
    *   Accuracy Score
    *   Classification Report (precision, recall, F1-score)
    *   Confusion Matrix
*   **Prediction on New Data:** Includes a function to predict the sentiment of new, unseen review text.
*   **Data Visualization:**
    *   Distribution of review scores.
    *   Distribution of derived sentiments.
    *   Confusion matrix heatmap.

## Dataset

The project uses the `netflix_reviews.csv` dataset.
*   **Source:** https://github.com/tanushreer-7/Netflix-Data-Sentimental-Analysis/blob/main/dataset.zip
*   **Key Columns Used:**
    *   `content`: The textual content of the user review.
    *   `score`: The numerical rating given by the user (1 to 5).
*   **Derived Target Variable:**
    *   `sentiment`: Categorical (Positive, Neutral, Negative), derived from the `score` column as follows:
        *   Score >= 4: Positive
        *   Score == 3: Neutral
        *   Score < 3: Negative

## Technologies Used

*   Python 3.x
*   Jupyter Notebook / Google Colab
*   **Libraries:**
    *   Pandas: For data manipulation and analysis.
    *   NumPy: For numerical operations.
    *   Matplotlib & Seaborn: For data visualization.
    *   NLTK (Natural Language Toolkit): For stopword removal.
    *   Scikit-learn:
        *   `TfidfVectorizer`: For TF-IDF feature extraction.
        *   `MultinomialNB`: For the classification model.
        *   `train_test_split`: For splitting data.
        *   `LabelEncoder`: For encoding the target variable.
        *   `accuracy_score`, `classification_report`, `confusion_matrix`: For model evaluation.
    *   re: For regular expression operations (text cleaning).

---

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tanushreer-7/Netflix-Data-Sentiment-Analysis-
    cd Netflix-Data-Sentiment-Analysis-
    ```

## Usage

1.  **Launch Jupyter Notebook/Lab:**
    ```bash
    jupyter lab
    ```
    or
    ```bash
    jupyter notebook
    ```
2.  Open the `Sentimental_Analysis_Upgraded.ipynb` notebook.
3.  Run the cells sequentially from top to bottom.
4.  The notebook will guide you through:
    *   Data loading and initial inspection.
    *   Data preprocessing (handling missing values, cleaning text).
    *   Exploratory Data Analysis (EDA).
    *   Feature extraction (TF-IDF).
    *   Model training (Multinomial Naive Bayes).
    *   Model evaluation.
    *   Prediction on sample new reviews.

## Expected Results

*   The model's accuracy in predicting sentiment based on review text will be displayed.
*   A detailed classification report showing precision, recall, and F1-score for each sentiment class.
*   A confusion matrix visualizing the model's predictions against actual sentiments.
*   *Note: The accuracy will be more realistic (likely not 100%) compared to a model predicting sentiment directly from the score, as text-based analysis is inherently more complex.*
