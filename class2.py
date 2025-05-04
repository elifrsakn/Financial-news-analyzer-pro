# 📚 Gerekli kütüphaneler
import re
import string
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# 🤗 FinBERT yükle
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
finbert_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None)

# Temizleme fonksiyonu
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    return text

# Prediction ve detaylı cevap
def classify_and_react(user_input):
    cleaned_text = clean_text(user_input)
    outputs = finbert_classifier(cleaned_text)
    outputs = outputs[0]

    labels_scores = {out['label']: out['score'] for out in outputs}
    best_label = max(labels_scores, key=labels_scores.get)
    confidence = labels_scores[best_label]

    # ✨ 1. Confidence'a göre özel mesaj
    if confidence >= 0.80:
        confidence_message = "✅ High confidence prediction."
    elif 0.50 <= confidence < 0.80:
        confidence_message = "🔹 Moderate confidence prediction. Please be cautious."
    else:
        confidence_message = "⚠️ Low confidence prediction. Interpretation might be unstable."

    # ✨ 2. Risk değerlendirmesi
    if best_label == "Positive":
        risk_message = "🌟 Low Risk - Good market indicators."
    elif best_label == "Negative":
        risk_message = "⚡ High Risk - Possible market instability."
    elif best_label == "Neutral":
        risk_message = "🔹 Medium Risk - Information without major impact."
    else:
        risk_message = "🤔 Unknown risk."

    # ✨ 3. Justification açıklaması
    justification = "🔍 Prediction based on detected financial terms and context."

    # Output hazırlama
    final_output = (
        f"🔮 Predicted Sentiment: {best_label} ({confidence:.2%} confidence)\n\n"
        f"{confidence_message}\n"
        f"{risk_message}\n\n"
        f"{justification}"
    )

    return final_output

# 🎨 Gradio arayüz
with gr.Blocks() as demo:
    gr.Markdown("# 🧠📈 FinBERT Advanced Financial News Analyzer")
    gr.Markdown("Analyze a financial news headline with sentiment, confidence, risk level, and justification!")

    user_input = gr.Textbox(lines=3, placeholder="Enter a financial news headline...", label="📝 News Headline")
    output_text = gr.Textbox(label="🔮 Full Analysis Result")

    submit_button = gr.Button("Analyze News")

    submit_button.click(fn=classify_and_react, inputs=user_input, outputs=output_text)

# App başlat
demo.launch()
