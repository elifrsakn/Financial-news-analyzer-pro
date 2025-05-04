#100 haberlik batch prediction  FinBERT
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

# FinBERT pipeline
finbert_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

# 🧠 Prediction fonksiyonu
def predict_finbert(text):
    outputs = finbert_classifier(text)
    outputs = outputs[0]

    labels_scores = {out['label']: out['score'] for out in outputs}
    best_label = max(labels_scores, key=labels_scores.get)
    confidence = labels_scores[best_label]
    
    return best_label, confidence

# ---------------------------------------------

# 🧪 BATCH PREDICTION (100 haber)

# 100 rastgele haber seçelim
sample_texts = df['Clean_Sentence'].sample(100, random_state=24)

predictions = []
confidences = []

for text in sample_texts:
    prediction, confidence = predict_finbert(text)
    predictions.append(prediction)
    confidences.append(confidence)

# Sonuçları DataFrame yapalım
batch_results = pd.DataFrame({
    "Text": sample_texts.values,
    "Predicted_Sentiment": predictions,
    "Confidence": confidences
})

# 🎯 Prediction dağılımı
print("\n✅ 100 Haberlik Batch Prediction Tamamlandı!")
print("\n🎯 Prediction Dağılımı:")
print(batch_results['Predicted_Sentiment'].value_counts())

# 🎯 Ortalama Confidence
print(f"\n🔍 Ortalama Confidence: {batch_results['Confidence'].mean()*100:.2f}%")

# 📥 (İstersek) CSV olarak da kaydedebiliriz:
# batch_results.to_csv("/Users/elifsakin/Desktop/dg_hediye/finbert_batch_results.csv", index=False)
