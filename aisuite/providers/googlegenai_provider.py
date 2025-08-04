import os
from google import generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse

# From upstream eliasjudin:add-gemini PR #181

class GoogleGenaiProvider(Provider):
    def __init__(self, **config):
        self.api_key = config.get("api_key") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "Gemini API key is missing. Please provide it in the config or set the GEMINI_API_KEY environment variable."
            )
        #self.model_name = config.get("model") or os.getenv("GEMINI_MODEL")

        self.gemini_model_config = {
            "temperature": 1, # lower this if we want less creative responses
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "stop_sequences": ["<|think|>"],
            "response_mime_type": "text/plain",
        }            
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,   
        }
        if config.get("safety_settings") is not None:
            self.safety_settings = config.get("safety_settings")
        genai.configure(api_key=self.api_key)


    def _get_model(self, model):
        model_name = model
        if model is None:
            model_name = os.getenv("GEMINI_MODEL")
            
        self.model = genai.GenerativeModel(
                        model_name=model_name, 
                        generation_config=self.gemini_model_config,
                        safety_settings=self.safety_settings
                    )
        return self.model


    def chat_completions_create(self, model, messages, **kwargs):
        try:
            response = self._get_model(model).generate_content(
                contents=[message["content"] for message in messages],
                **kwargs
            )
            return self.normalize_response(response)
        except Exception as e:
            raise LLMError(f"Error in chat_completions_create: {str(e)}")

    def generate_content(self, model, contents, **kwargs):
        try:
            response = self._get_model().generate_content(
                contents=contents,
                safety_settings=self.safety_settings,
                **kwargs
            )
            return self.normalize_response(response)
        except Exception as e:
            raise LLMError(f"Error in generate_content: {str(e)}")

    def list_models(self):
        try:
            response = genai.list_models()
            return [model.name for model in response]
        except Exception as e:
            raise LLMError(f"Error in list_models: {str(e)}")

    def normalize_response(self, response):
        normalized_response = ChatCompletionResponse()
        normalized_response.choices[0].message.content = response.text
        return normalized_response
