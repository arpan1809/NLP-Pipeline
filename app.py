
import os
from flask import Flask, render_template, request
from transformers import pipeline
from langdetect import detect

app = Flask(__name__)


sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
ner_pipeline = pipeline("ner", model="Davlan/xlm-roberta-base-ner-hrl", grouped_entities=True)
translator_pipeline = pipeline("translation", model="facebook/mbart-large-50-many-to-many-mmt", tokenizer="facebook/mbart-large-50-many-to-many-mmt")
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")


def summarize_long_text(text, max_chunk=1024):
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    summaries = []
    for chunk in chunks:
        try:
            summary = summarization_pipeline(chunk, max_length=512, min_length=50, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            summaries.append(f"[Error] {str(e)}")
    return " ".join(summaries)


lang_map = {
    "en": "en_XX",
    "fr": "fr_XX",
    "hi": "hi_IN",
    "es": "es_XX",
    "de": "de_DE",
    # Add more as needed
}

@app.route('/', methods=['GET', 'POST'])
def index():
    results = {}
    if request.method == 'POST':
        text_input = request.form.get('text_input')
        target_lang = request.form.get('target_lang')

        try:
            lang = detect(text_input)
            lang_code = lang_map.get(lang, 'en_XX')
        except:
            lang = 'unknown'
            lang_code = 'en_XX'

        try:
            sentiment = sentiment_pipeline(text_input)
        except Exception as e:
            sentiment = f"Error: {str(e)}"

        try:
            ner = ner_pipeline(text_input)
        except Exception as e:
            ner = f"Error: {str(e)}"

        try:
            translation = translator_pipeline(text_input, src_lang=lang_code, tgt_lang=target_lang)
            translated_text = translation[0]['translation_text']
        except Exception as e:
            translated_text = f"Error: {str(e)}"

        try:
            summary = summarize_long_text(text_input)
        except Exception as e:
            summary = f"Error: {str(e)}"

        results = {
            "text": text_input,
            "lang": lang,
            "sentiment": sentiment,
            "ner": ner,
            "translation": translated_text,
            "summary": summary
        }

    return render_template("index.html", results=results)

if __name__ == '__main__':
    app.run(debug=True)




