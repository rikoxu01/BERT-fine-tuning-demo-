# Sentiment Classification using BERT

This repository contains code for fine-tuning a pre-trained BERT model for sentiment classification (positive, neutral, and negative), and visualising sentiment progression across long-form mixed-content text using the fine-tuned model. 

(Developed in 2023)

📁 Overview
- Uses the Transformers library by Hugging Face to load and fine-tune `BertForSequenceClassification`.
-	Tracks training loss over 20 epochs and saves model checkpoints and training logs.
- Splits text into chunks of varying sizes to test BERT’s consistency at different granularities.
- Visualises sentiment fluctuations across the document, giving insight into how sentiment polarity trends over time.


⚙️ Python Libraries
-  `transformers`, `torch`, `sklearn`, `numpy`, `random`, `matplotlib`
