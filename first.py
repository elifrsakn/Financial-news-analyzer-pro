# 📚 Gerekli kütüphaneler
import pandas as pd
import re
import string
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# 📂 Dataset Yükleme
dataset_path = "/Users/elifsakin/Desktop/dg_hediye/final_augmented_stock_sentiment.csv"
df = pd.read_csv(dataset_path)

# Dataset'teki cümleleri temizleme
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    return text

df['Clean_Sentence'] = df['Sentence'].astype(str).apply(clean_text)

# 🤗 FinBERT Model ve Tokenizer Yükleme
model_name = "yiyanghkust/finbert-tone"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# FinBERT için özel pipeline oluşturuyoruz
finbert_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

# 🧠 Prediction fonksiyonu
def predict_finbert(text):
    outputs = finbert_classifier(text)
    outputs = outputs[0]  # Sadece ilk örneği alıyoruz

    # Skorları ayıklama
    labels_scores = {out['label']: out['score'] for out in outputs}
    best_label = max(labels_scores, key=labels_scores.get)
    confidence = labels_scores[best_label]
    
    return best_label, confidence

# ---------------------------------------------

# 🧪 TEST

# Şimdi 5 rastgele cümlede prediction yapalım
sample_texts = df['Clean_Sentence'].sample(5, random_state=24)

for idx, text in enumerate(sample_texts):
    prediction, confidence = predict_finbert(text)
    print(f"\n📰 Haber {idx+1}: {text}")
    print(f"🔮 FinBERT Prediction: {prediction} ({confidence:.2%} confidence)")
