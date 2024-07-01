from transformers import pipeline

# Change `xx` to the language of the input and `yy` to the language of the desired output.
# Examples: "en" for English, "fr" for French, "de" for German, "es" for Spanish, "zh" for Chinese, etc; translation_en_to_fr translates English to French
# You can view all the lists of languages here - https://huggingface.co/languages
translator = pipeline("translation_xx_to_yy", model="stevhliu/my_awesome_opus_books_model", device = 0)
text = "translate english to spainish: Legumes share resources with nitrogen-fixing bacteria."

result = translator(text)

print(result)