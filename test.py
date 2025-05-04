import pandas as pd
from transformers import pipeline
import re
import string

# 📂 Dataset yükleme
dataset_path = "/Users/elifsakin/Desktop/dg_hediye/final_augmented_stock_sentiment.csv"
df = pd.read_csv(dataset_path)

# Temizleme fonksiyonu
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    return text

df['Clean_Sentence'] = df['Sentence'].astype(str).apply(clean_text)

# 🤗 Model yükleme
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 🔮 İlk mini sample ile threshold tahmini
def estimate_threshold(sample_texts):
    confidences = []
    for text in sample_texts:
        candidate_labels = ["market optimism", "market fear", "neutral"]
        result = classifier(
            text,
            candidate_labels,
            hypothesis_template="Given the news text, determine if it indicates {}."
        )
        max_confidence = max(result['scores'])
        confidences.append(max_confidence)
    
    threshold = sum(confidences) / len(confidences)  # Ortalama
    return threshold

# 🧠 Prediction fonksiyonu
def academic_predict_stock_sentiment(text, dynamic_threshold):
    candidate_labels = ["market optimism", "market fear", "neutral"]
    result = classifier(
        text,
        candidate_labels,
        hypothesis_template="Given the news text, determine if it indicates {}."
    )
    probabilities = result['scores']
    max_confidence = max(probabilities)
    best_label = result['labels'][probabilities.index(max_confidence)]
    
    if max_confidence < dynamic_threshold:
        return "🤔 Unsure prediction (Low confidence)", max_confidence
    else:
        return best_label, max_confidence

# ---------------------------------------------
# 🧪 TEST BAŞLIYOR

# 1. İlk batch'ten threshold tahmini (50 haberle)
sample_texts_for_threshold = df['Clean_Sentence'].sample(50, random_state=42)
dynamic_threshold = estimate_threshold(sample_texts_for_threshold)

print(f"\n🔍 Otomatik Hesaplanan Threshold: {dynamic_threshold:.2f}")

# 2. 100 haberlik test seti seçelim
test_texts = df['Clean_Sentence'].sample(100, random_state=24)

# Prediction listeleri
predictions = []
confidences = []
unsure_flags = []

# 3. Prediction yapalım
for text in test_texts:
    pred_label, confidence = academic_predict_stock_sentiment(text, dynamic_threshold)
    predictions.append(pred_label)
    confidences.append(confidence)
    unsure_flags.append(1 if "Unsure" in pred_label else 0)

# 4. Sonuçları toparlayalım
test_results = pd.DataFrame({
    'Text': test_texts.values,
    'Prediction': predictions,
    'Confidence': confidences,
    'Unsure': unsure_flags
})

# 5. Rapor
print("\n✅ Batch Test Tamamlandı!")
print("\n🎯 Prediction Dağılımı:")
print(test_results['Prediction'].value_counts())

unsure_rate = test_results['Unsure'].mean() * 100
mean_confidence = test_results['Confidence'].mean() * 100

print(f"\n🔍 Unsure Prediction Oranı: {unsure_rate:.2f}%")
print(f"🔍 Ortalama Confidence: {mean_confidence:.2f}%")
import matplotlib.pyplot as plt
import pandas as pd

# (Eğer yukarıdaki test_results dataframe elimizdeyse tekrar okumaya gerek yok.)

# Test sonuçlarını yeniden yüklemek istersen:
# test_results = pd.read_csv("/path_to_your_saved_test_results.csv")

# Unsure ve Confident prediction sayıları
unsure_count = (test_results['Prediction'] == "🤔 Unsure prediction (Low confidence)").sum()
confident_count = len(test_results) - unsure_count

# Pie Chart - Unsure vs Confident
labels = ['Confident Predictions', 'Unsure Predictions']
sizes = [confident_count, unsure_count]
colors = ['#66b3ff', '#ff9999']
explode = (0.05, 0.05)  # Pasta dilimlerini hafif ayırıyoruz

plt.figure(figsize=(8,6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Prediction Confidence Distribution')
plt.axis('equal')  # Daireyi tam çizer
plt.show()

# ----------------------------------------

# (Opsiyonel) Bar Chart
plt.figure(figsize=(8,5))
bars = plt.bar(['Confident', 'Unsure'], [confident_count, unsure_count], color=['#66b3ff', '#ff9999'])
plt.title('Number of Predictions by Type')
plt.ylabel('Number of Predictions')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom')
plt.show()
