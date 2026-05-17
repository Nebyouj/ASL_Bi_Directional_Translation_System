from deep_translator import GoogleTranslator


class Translator:
    def __init__(self):
        pass

    def translate(self, text, target_lang):
        if not text or not text.strip():
            return text

        try:
            return GoogleTranslator(
                source="auto",
                target=target_lang
            ).translate(text)
        except Exception as e:
            print("Translation error:", e)
            return text

    def translate_with_model(self, text, model_name=None):
        # You don't need this anymore
        return self.translate(text, "en")