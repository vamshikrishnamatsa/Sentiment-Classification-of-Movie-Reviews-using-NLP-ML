# Sentiment-Classification-of-Movie-Reviews-using-NLP-ML

A Natural Language Processing (NLP) project to classify movie reviews as **positive** or **negative** using machine learning models.

![Screenshot 2025-06-13 133202](https://github.com/user-attachments/assets/f39d6e56-d2a5-475a-8e1a-24c9117ba4f7)
 <!-- Replace with your actual image path -->

---

## 📌 Project Overview

This project demonstrates an end-to-end sentiment analysis pipeline:

- 🧹 Text cleaning and preprocessing  
- 🔤 Vectorization using **TF-IDF**  
- 🧠 Machine learning model training (Logistic Regression, Naive Bayes, etc.)  
- 📊 Evaluation using **precision, recall, accuracy, and F1-score**  
- 🔮 Sentiment prediction on new/unseen movie reviews  

---

## 🧰 Technologies Used

- **Python**
- **Pandas**, **NumPy** – Data manipulation
- **Scikit-learn** – ML models and evaluation metrics
- **NLTK**, **re** – Text preprocessing
- **Matplotlib**, **Seaborn**, **WordCloud** – Visualization (optional)
- **Pickle**, **Joblib** – Model serialization (optional)

---

## 📂 Project Structure
├── movie_review_sentiment_analysis.ipynb # Main notebook
├── data
├── model.pkl # Trained model (optional)
├── requirements.txt # List of dependencies (optional)
└── README.md
---

## 🔍 Key Steps

1. **Data Loading** – Movie reviews and sentiment labels  
2. **Text Preprocessing** – Cleaning, tokenization, stopword removal  
3. **TF-IDF Vectorization** – Converting text to numeric features  
4. **Model Training** – Logistic Regression, Naive Bayes, etc.  
5. **Model Evaluation** – Confusion matrix, precision, recall, F1-score  
6. **Prediction** – Sentiment prediction on new or user-provided reviews  

---

## ✅ Sample Results

| Metric     | Score  |
|------------|--------|
| Accuracy   | 87%    |
| Precision  | 0.89   |
| Recall     | 0.85   |
| Model Used | Logistic Regression + TF-IDF |
![2](https://github.com/user-attachments/assets/36bd8551-271a-4c82-b681-ba608596bc81)



---

## 💡 Future Improvements

- 🖥️ Add Streamlit app for user input and real-time predictions  
- 🧠 Integrate BERT-based models for improved accuracy  
- 📈 Expand dataset for better generalization  

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/movie-review-sentiment.git
   cd movie-review-sentiment

