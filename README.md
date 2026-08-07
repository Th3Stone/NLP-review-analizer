# Amazon Product Review Sentiment Analyzer

This project is an NLP-based Deep Learning solution designed to classify Amazon product reviews into "Positive" or "Negative" sentiments. It trains a custom Long Short-Term Memory (LSTM) neural network using PyTorch and features a simple, interactive web interface built with Gradio.

## Features & Assignment Requirements Fulfilled
* **Data Parsing:** Extracts raw text and star ratings from pseudo-XML files across four product domains (Kitchen, Books, DVDs, Electronics).
* **Ethical Data Processing:** Implements an ethical validation check to ensure neutral (3-star) or contradictory reviews weren't forced into binary labels, preventing biased model training.
* **Data Preprocessing:** Cleans text (punctuation/spelling), removes outliers (short reviews < 4 words), and applies word encoding and pre-padding.
* **Deep Learning Architecture:** Utilizes a custom PyTorch LSTM model with Embedding, Dropout, and Fully Connected layers.
* **Rearchitecting for Performance:** Improved initial model performance by switching from post-padding to pre-padding, scaling down network dimensions to prevent overfitting, and implementing gradient clipping.
* **Deployment:** Deploys locally via a Gradio web application for easy inference on custom inputs.

## Prerequisites
Ensure you have Python 3.8+ installed. You will also need the Multi-Domain Sentiment Dataset extracted in your project directory (e.g., in a folder named `sorted_data_acl`).
