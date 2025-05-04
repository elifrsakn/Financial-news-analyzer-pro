# 📚 Gerekli Kütüphaneler (Libraries)
import re
import string
import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from langdetect import detect  # Dil tespiti için
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# 🤗 Modelleri Yükleme (Loading Models)
# FinBERT (Finansal haberler için)
finbert_model_name = "yiyanghkust/finbert-tone"
finbert_tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
finbert_model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)
finbert_classifier = TextClassificationPipeline(model=finbert_model, tokenizer=finbert_tokenizer, top_k=None)

# XLM-Roberta (Genel Multilingual haberler için)
xlm_model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
xlm_tokenizer = AutoTokenizer.from_pretrained(xlm_model_name)
xlm_model = AutoModelForSequenceClassification.from_pretrained(xlm_model_name)
xlm_classifier = TextClassificationPipeline(model=xlm_model, tokenizer=xlm_tokenizer, top_k=None)

# 🧹 Temizleme Fonksiyonu (Text Cleaning Function)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    return text

# 🔮 Prediction Fonksiyonu
def smart_batch_classify(user_inputs):
    texts = user_inputs.split('\n')  # Her satır bir haber
    texts = [t for t in texts if t.strip() != '']  # Boşları çıkar

    results = []
    for text in texts:
        original_text = text
        cleaned_text = clean_text(text)

        # 🔍 Dil tespiti
        try:
            lang = detect(cleaned_text)
        except:
            lang = "unknown"

        # 🔄 Model seçimi
        if lang == "en":
            outputs = finbert_classifier(cleaned_text)[0]
        else:
            outputs = xlm_classifier(cleaned_text)[0]

        labels_scores = {out['label'].lower(): out['score'] for out in outputs}
        best_label = max(labels_scores, key=labels_scores.get)
        best_confidence = labels_scores[best_label]

        positive_score = labels_scores.get("positive", 0) * 100
        negative_score = labels_scores.get("negative", 0) * 100
        neutral_score = labels_scores.get("neutral", 0) * 100

        # 🏷️ Auto-tagging (Otomatik etiket)
        if best_label == "positive":
            auto_tag = "📈 Opportunity (Fırsat)"
        elif best_label == "negative":
            auto_tag = "⚠️ Risk Alert (Risk Uyarısı)"
        elif best_label == "neutral":
            auto_tag = "ℹ️ Informational (Bilgilendirme)"
        else:
            auto_tag = "🤔 Unclassified (Sınıflandırılamadı)"

        results.append((
            original_text,
            lang,
            best_label,
            best_confidence * 100,
            positive_score,
            negative_score,
            neutral_score,
            auto_tag
        ))

    df_results = pd.DataFrame(results, columns=[
        "Original Text (Orijinal Metin)",
        "Detected Language (Tespit Edilen Dil)",
        "Predicted Sentiment (Tahmin Edilen Duygu)",
        "Best Confidence (%) (En Yüksek Güven %)",
        "Positive (%) (Pozitif %)",
        "Negative (%) (Negatif %)",
        "Neutral (%) (Nötr %)",
        "Auto Tag (Otomatik Etiket)"
    ])

    return df_results

# 🎨 Gradio Fonksiyonu
def smart_analyze_and_filter(user_inputs, selected_sentiment):
    df_results = smart_batch_classify(user_inputs)

    if selected_sentiment != "All":
        df_results = df_results[df_results["Predicted Sentiment (Tahmin Edilen Duygu)"] == selected_sentiment]

    fig, ax = plt.subplots(figsize=(10,6))
    sentiment_colors = {
        "positive": "green",
        "negative": "red",
        "neutral": "blue"
    }

    color_list = [sentiment_colors.get(sent, "gray") for sent in df_results["Predicted Sentiment (Tahmin Edilen Duygu)"].str.lower()]

    ax.bar(range(len(df_results)), df_results["Best Confidence (%) (En Yüksek Güven %)"], color=color_list)
    ax.set_xticks(range(len(df_results)))
    ax.set_xticklabels([f"News {i+1}" for i in range(len(df_results))], rotation=45)
    ax.set_ylabel("Best Confidence (%) (En Yüksek Güven %)")
    ax.set_title(f"Sentiment Confidence Trend ({selected_sentiment}) (Duygu Güveni Trend)")

    plt.tight_layout()

    return df_results, fig

# 🚀 Gradio Arayüzü (Gradio Interface)
with gr.Blocks() as demo:
    # 🟩 Başlığı yeşil yaptık
    gr.Markdown("<h1 style='color: green;'>🧠🌍📰 Smart Multilingual Financial News Analyzer</h1>")
    gr.Markdown("Paste multiple news headlines (any language!) and get accurate sentiment prediction!")
    gr.Markdown("## 🧠🌍📰 Çok Dilli Finansal Haber Analizörü")
    gr.Markdown("Birden fazla haber başlığını yapıştırın ve doğru modelle otomatik analiz edin!")
    gr.Markdown("Haberleri satır satır analiz ediyorum.🗞️📰")

    # 🎂✨ Doğum günü mesajı (kırmızı yaptık)
    gr.HTML("""
    <div style='text-align: center; color: red; font-size: 12px; font-weight: bold;'>
        🎂 Doğum Günün Kutlu Olsun Babacım, Ali Sakin! 🎉❤️
    </div>
    """)
    gr.HTML("""
    <div style='text-align: center; color: gray; font-size: 4px;'>
        (Happy Birthday My Father, Ali Sakin 🎂)
    </div>
    """)

    user_input = gr.Textbox(lines=10, placeholder="Enter each news headline (any language)...", label="📝 News Headlines (Haber Başlıkları)")
    sentiment_filter = gr.Dropdown(["All", "positive", "negative", "neutral"], label="🎯 Filter by Sentiment (Duyguya Göre Filtrele)", value="All")

    output_table = gr.Dataframe(label="🔮 Prediction Table (Otomatik Etiketli Tahmin Tablosu)")
    output_plot = gr.Plot(label="📈 Sentiment Confidence Trend (Duygu Güveni Trend Grafiği)")

    submit_button = gr.Button("Analyze News Batch (Haberleri Analiz Et)")

    submit_button.click(fn=smart_analyze_and_filter, inputs=[user_input, sentiment_filter], outputs=[output_table, output_plot])

# App Başlat (Launch App)
demo.launch()
