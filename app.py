
import re
import string
import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, pipeline


# FinBERT yükle
finbert_model_name = "yiyanghkust/finbert-tone"
finbert_tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
finbert_model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)
finbert_classifier = TextClassificationPipeline(model=finbert_model, tokenizer=finbert_tokenizer, top_k=None)


translator = pipeline("translation", model="Helsinki-NLP/opus-mt-tr-en", max_length=512)



def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    return text


def multilingual_batch_classify(user_inputs):
    texts = user_inputs.split('\n')  # Her satırı ayrı haber
    texts = [t for t in texts if t.strip() != '']  # Boş satırları çıkar

    results = []
    for text in texts:
        original_text = text
        cleaned_text = clean_text(text)

        # 🔍 İngilizce'ye çevir
        translation_output = translator(cleaned_text)[0]['translation_text']
        translated_text = translation_output

        # 🔮 FinBERT ile analiz
        outputs = finbert_classifier(translated_text)
        outputs = outputs[0]

        labels_scores = {out['label']: out['score'] for out in outputs}
        best_label = max(labels_scores, key=labels_scores.get)
        best_confidence = labels_scores[best_label]

        positive_score = labels_scores.get("Positive", 0) * 100
        negative_score = labels_scores.get("Negative", 0) * 100
        neutral_score = labels_scores.get("Neutral", 0) * 100

        # 🏷️ Auto-tagging
        if best_label == "Positive":
            auto_tag = "📈 Opportunity"
        elif best_label == "Negative":
            auto_tag = "⚠️ Risk Alert"
        elif best_label == "Neutral":
            auto_tag = "ℹ️ Informational"
        else:
            auto_tag = "🤔 Unclassified"

        results.append((
            original_text,
            translated_text,
            best_label,
            best_confidence * 100,
            positive_score,
            negative_score,
            neutral_score,
            auto_tag
        ))

    df_results = pd.DataFrame(results, columns=[
        "Original Text", "Translated Text", "Predicted Sentiment", "Best Confidence (%)",
        "Positive (%)", "Negative (%)", "Neutral (%)", "Auto Tag"
    ])

    return df_results


def multilingual_analyze_and_filter(user_inputs, selected_sentiment):
    df_results = multilingual_batch_classify(user_inputs)

    
    if selected_sentiment != "All":
        df_results = df_results[df_results["Predicted Sentiment"] == selected_sentiment]

   
    fig, ax = plt.subplots(figsize=(10,6))
    sentiment_colors = {
        "Positive": "green",
        "Negative": "red",
        "Neutral": "blue"
    }

    color_list = [sentiment_colors.get(sent, "gray") for sent in df_results["Predicted Sentiment"]]

    ax.bar(range(len(df_results)), df_results["Best Confidence (%)"], color=color_list)
    ax.set_xticks(range(len(df_results)))
    ax.set_xticklabels([f"News {i+1}" for i in range(len(df_results))], rotation=45)
    ax.set_ylabel("Best Confidence (%)")
    ax.set_title(f"Sentiment Confidence Trend ({selected_sentiment})")

    plt.tight_layout()

    return df_results, fig


with gr.Blocks() as demo:
    gr.Markdown("# 🧠🌍 FinBERT Multilingual Financial News Analyzer")
    gr.Markdown("Paste multiple news headlines (any language!) and see full sentiment analysis after translation!")

    user_input = gr.Textbox(lines=10, placeholder="Enter each news headline on a new line...", label="📝 News Headlines (any language)")
    sentiment_filter = gr.Dropdown(["All", "Positive", "Negative", "Neutral"], label="🎯 Filter by Sentiment", value="All")

    output_table = gr.Dataframe(label="🔮 Prediction Table (with Auto Tags and Translation)")
    output_plot = gr.Plot(label="📈 Sentiment Confidence Trend")

    submit_button = gr.Button("Analyze News Batch (Multilingual)")

    submit_button.click(fn=multilingual_analyze_and_filter, inputs=[user_input, sentiment_filter], outputs=[output_table, output_plot])

# App başlat
demo.launch()
