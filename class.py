# 📚 Gerekli kütüphaneler
import re
import string
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# 🤗 FinBERT Model ve Tokenizer Yükleme
model_name = "yiyanghkust/finbert-tone"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

finbert_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None)

# Temizleme Fonksiyonu
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    return text

# Prediction ve Yönlendirme Fonksiyonu
def classify_and_react(user_input):
    cleaned_text = clean_text(user_input)
    outputs = finbert_classifier(cleaned_text)
    outputs = outputs[0]

    labels_scores = {out['label']: out['score'] for out in outputs}
    best_label = max(labels_scores, key=labels_scores.get)
    confidence = labels_scores[best_label]

    # Sınıfa göre mesaj
    if best_label == "Positive":
        reaction = "🌟 Great news! This could boost market optimism!"
    elif best_label == "Negative":
        reaction = "⚠️ Warning! This could indicate market fear or instability."
    elif best_label == "Neutral":
        reaction = "📄 This seems to be neutral information without a strong market impact."
    else:
        reaction = "🤔 Unable to classify."

    return f"🔮 Predicted Sentiment: {best_label} ({confidence:.2%} confidence)\n\n💬 {reaction}"

# ---------------------------------------------

# 🎨 Gradio Arayüz Tasarımı
with gr.Blocks() as demo:
    gr.Markdown("# 🧠📈 FinBERT Financial News Sentiment Analyzer")
    gr.Markdown("Analyze a financial news headline and get a sentiment prediction!")
    
    user_input = gr.Textbox(lines=3, placeholder="Enter financial news headline...", label="📝 News Headline")
    output_text = gr.Textbox(label="🔮 Prediction & Reaction")
    
    submit_button = gr.Button("Analyze News")
    
    submit_button.click(fn=classify_and_react, inputs=user_input, outputs=output_text)

# 🌟 Gradio uygulamasını başlatalım
demo.launch()
