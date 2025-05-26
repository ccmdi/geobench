import base64
import requests
import os
from abc import ABC, abstractmethod
from ratelimit import limits, sleep_and_retry
from typing import List

def get_image_media_type(image_path: str) -> str:
    """Determine the media type based on file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext == '.png':
        return "image/png"
    elif ext in ['.jpg', '.jpeg']:
        return "image/jpeg"
    else:
        return "image/jpeg" # Default fallback

class BaseMultimodalModel(ABC):
    """Abstract base class for multimodal models."""
    api_key_name: str = None
    model_identifier: str = None
    name: str = None
    rate_limit: int = 5
    rate_limit_period: int = 60
    max_tokens: int = 32000
    temperature: float = 0.4

    def __init__(self, api_key: str):
        if not self.name:
            self.name = self.__class__.__name__
        if not self.api_key_name:
            raise NotImplementedError(f"api_key_name must be set in {self.name} or its parent provider class")
        if not self.model_identifier:
             raise NotImplementedError(f"model_identifier must be set in {self.name}")
        self.api_key = api_key

    def _encode_image(self, image_path: str) -> tuple[str, str]:
        media_type = get_image_media_type(image_path)
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")
        return img_data, media_type

    @abstractmethod
    def _build_headers(self) -> dict: pass

    @abstractmethod
    def _build_payload(self, prompt: str, img_data: str, media_type: str) -> dict: pass

    @abstractmethod
    def _get_endpoint(self) -> str: pass

    @abstractmethod
    def _extract_response_text(self, response: requests.Response) -> str: pass

    def query(self, image_path: str, prompt: str, run_folder: str = None, location_id: str = None) -> str:
        """
        Public method to query the model.
        """

        def api():
            img_data, media_type = self._encode_image(image_path)
            headers = self._build_headers()
            payload = self._build_payload(prompt, img_data, media_type)
            endpoint = self._get_endpoint()
            try:
                response = requests.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                if run_folder and location_id:
                    os.makedirs(f"{run_folder}/json/", exist_ok=True)
                    with open(f"{run_folder}/json/{location_id}.json", "w", encoding="utf-8") as f:
                        f.write(response.text)
                return self._extract_response_text(response)
            except requests.exceptions.RequestException as e:
                status_code = e.response.status_code if e.response is not None else "N/A"
                error_text = e.response.text if e.response is not None else str(e)
                print(f"API error ({status_code}) for {self.name}: {error_text[:100]}...")
                raise Exception(f"{self.name} API error ({status_code})") from e
            except Exception as e:
                print(f"Unexpected error in {self.name} core logic: {str(e)}")
                raise

        call = sleep_and_retry(
            limits(calls=self.rate_limit, period=self.rate_limit_period)(api)
        )

        return call()



class AnthropicClient(BaseMultimodalModel):
    api_key_name = "ANTHROPIC_API_KEY"
    base_url = "https://api.anthropic.com/v1/messages"
    anthropic_version: str = "2023-06-01"
    beta_header: str = None
    enable_thinking: bool = False

    def _get_endpoint(self) -> str:
        return self.base_url

    def _build_headers(self) -> dict:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json"
        }
        effective_beta_header = getattr(self, 'beta_header', None)
        if effective_beta_header:
             headers["anthropic-beta"] = effective_beta_header
        return headers

    def _build_payload(self, prompt: str, img_data: str, media_type: str) -> dict:
        payload = {
            "model": self.model_identifier,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_data}}
            ]}]
        }
        if self.enable_thinking:
            payload["thinking"] = {"type": "enabled", "budget_tokens": self.max_tokens - 512}
            if "temperature" in payload:
                del payload["temperature"]
        return payload

    def _extract_response_text(self, response: requests.Response) -> str:
        response_json = response.json()
        if not response_json.get("content"):
             raise ValueError(f"Unexpected Anthropic response format: {response_json}")
        thinking_text, response_text = "", ""
        for block in response_json["content"]:
            if block.get("type") == "thinking": thinking_text = block.get("thinking", "")
            elif block.get("type") == "text": response_text = block.get("text", "")

        if self.enable_thinking and thinking_text:
             return f"<thinking>{thinking_text}</thinking>\n\n{response_text}"
        elif response_text:
             return response_text
        else:
             raise ValueError(f"Could not extract text from Anthropic response: {response_json}")


class GoogleClient(BaseMultimodalModel):
    api_key_name = "GEMINI_API_KEY"
    base_url = "https://generativelanguage.googleapis.com"
    api_version_path: str = "v1" # e.g., "beta/" for experimental versions
    tools: str = None

    def _get_endpoint(self) -> str:
        action = "generateContent"
        version_path = getattr(self, 'api_version_path', '')
        return f"{self.base_url}/{version_path}/models/{self.model_identifier}:{action}?key={self.api_key}"

    def _build_headers(self) -> dict:
        return {"Content-Type": "application/json"}

    def _build_payload(self, prompt: str, img_data: str, media_type: str) -> dict:
        payload = {
            "contents": [{"parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": media_type, "data": img_data}}
            ]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens
            }
        }

        if self.tools:
            payload["tools"] = self.tools

        return payload

    def _extract_response_text(self, response: requests.Response) -> str:
        response_json = response.json()
        try:
            parts = response_json['candidates'][0]['content']['parts']
            return ''.join(part.get('text', '') for part in parts)
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Could not extract text from Google response: {response_json}") from e


class OpenAIClient(BaseMultimodalModel):
    api_key_name = "OPENAI_API_KEY"
    base_url = "https://api.openai.com/v1/responses"

    reasoning_effort: str = None
    detail: str = None
    tools: List = None

    def _get_endpoint(self) -> str:
        return self.base_url

    def _build_headers(self) -> dict:
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

    def _build_payload(self, prompt: str, img_data: str, media_type: str) -> dict:
        image_url = f"data:{media_type};base64,{img_data}"

        image_content = {
            "type": "input_image",
            "image_url": image_url
        }
        if self.detail:
            image_content["detail"] = self.detail

        payload = {
            "model": self.model_identifier,
            "input": [
                {"role": "user", "content": [
                    {"type": "input_text", "text": prompt},
                    image_content
                ]}
             ],
        }


        if hasattr(self, 'temperature') and self.temperature >= 0:
             payload["temperature"] = self.temperature

        if hasattr(self, 'max_output_tokens') and self.max_output_tokens > 0:
             payload["max_output_tokens"] = self.max_output_tokens

        if self.reasoning_effort:
             payload["reasoning"] = {"effort": self.reasoning_effort}

        if self.tools:
             payload["tools"] = self.tools

        return payload

    def _extract_response_text(self, response: requests.Response) -> str:
        response_json = response.json()
        try:
            status = response_json.get("status")
            if status != "completed":
                error_details = response_json.get("error") or response_json.get("incomplete_details") or f"Status: {status}"
                raise ValueError(f"Response generation not completed: {error_details}")

            message_output = None
            for item in response_json.get('output', []):
                if item.get('type') == 'message' and item.get('status') == 'completed':
                    message_output = item
                    break
            
            if not message_output:
                raise ValueError(f"No completed message found in OpenAI response: {response_json}")

            content_items = message_output.get('content', [])
            
            response_text = ''.join(
                item.get('text', '')
                for item in content_items
                if item.get('type') == 'output_text'
            )

            if not response_text and content_items:
                raise ValueError(f"No completed message found in OpenAI response: {response_json}")

            return response_text
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            raise ValueError(f"Could not extract text from OpenAI v1/responses structure: {response_json}") from e

class OpenRouterClient(BaseMultimodalModel):
    api_key_name = "OPENROUTER_API_KEY"
    base_url = "https://openrouter.ai/api/v1/chat/completions"
    referer_url: str = "https://geobench.org"

    def _get_endpoint(self) -> str:
        return self.base_url

    def _build_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.referer_url
        }

    def _build_payload(self, prompt: str, img_data: str, media_type: str) -> dict:
        image_url = f"data:{media_type};base64,{img_data}"
        return {
            "model": self.model_identifier,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]}],
            "temperature": self.temperature
        }

    def _extract_response_text(self, response: requests.Response) -> str:
        response_json = response.json()
        try:
            return response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Could not extract text from OpenRouter response: {response_json}") from e


# Anthropic Models
class Claude3_5Haiku(AnthropicClient):
    name = "Claude 3.5 Haiku"
    model_identifier = "claude-3-haiku-20240307"
    max_tokens = 4096
class Claude3_5Sonnet(AnthropicClient):
    name = "Claude 3.5 Sonnet" 
    model_identifier = "claude-3-5-sonnet-20241022"
    max_tokens = 8192
class Claude3_7Sonnet(AnthropicClient):
    name = "Claude 3.7 Sonnet"
    model_identifier = "claude-3-7-sonnet-20250219"
class Claude3_7SonnetThinking(AnthropicClient):
    name = "Claude 3.7 Sonnet (Thinking)"
    model_identifier = "claude-3-7-sonnet-20250219"
    enable_thinking = True
    rate_limit = 2
    beta_header = "output-128k-2025-02-19"
class Claude4Sonnet(AnthropicClient):
    name = "Claude 4 Sonnet"
    model_identifier = "claude-sonnet-4-20250514"
class Claude4Opus(AnthropicClient):
    name = "Claude 4 Opus"
    model_identifier = "claude-opus-4-20250514"
class Claude4SonnetThinking(AnthropicClient):
    name = "Claude 4 Sonnet (Thinking)"
    model_identifier = "claude-sonnet-4-20250514"
    enable_thinking = True
    rate_limit = 2
    beta_header = "output-128k-2025-02-19"
class Claude4OpusThinking(AnthropicClient):
    name = "Claude 4 Opus (Thinking)"
    model_identifier = "claude-opus-4-20250514"
    enable_thinking = True
    rate_limit = 2
    beta_header = "output-128k-2025-02-19"


# Google Models
class Gemini1_5Flash(GoogleClient):
    name = "Gemini 1.5 Flash"
    model_identifier = "gemini-1.5-flash"
    rate_limit = 10
class Gemini1_5Pro(GoogleClient):
    name = "Gemini 1.5 Pro"
    model_identifier = "gemini-1.5-pro"
    rate_limit = 2
class Gemini2Flash(GoogleClient):
    name = "Gemini 2.0 Flash"
    model_identifier = "gemini-2.0-flash"
    rate_limit = 10
class Gemini2_5Pro(GoogleClient):
    name = "Gemini 2.5 Pro"
    model_identifier = "gemini-2.5-pro-preview-05-06"
    rate_limit = 2
    api_version_path = "v1beta"
class Gemini2_5ProSearch(GoogleClient):
    name = "Gemini 2.5 Pro (with Search)"
    model_identifier = "gemini-2.5-pro-preview-05-06"
    rate_limit = 2
    api_version_path = "v1beta"
    
    tools = [{"google_search": {}}]
class Gemini2_5Flash_0417(GoogleClient):
    name = "Gemini 2.5 Flash Preview (04-17)"
    model_identifier = "gemini-2.5-flash-preview-04-17"
    rate_limit = 6
    api_version_path = "v1beta"
class Gemini2_5Flash_0520(GoogleClient):
    name = "Gemini 2.5 Flash Preview (05-20)"
    model_identifier = "gemini-2.5-flash-preview-05-20"
    rate_limit = 6
    api_version_path = "v1beta"


# OpenAI Models
class GPT4oMini(OpenAIClient):
    name = "GPT-4o Mini"
    model_identifier = "gpt-4o-mini"
class GPT4o(OpenAIClient):
    name = "GPT-4o"
    model_identifier = "gpt-4o"
    rate_limit = 3
class GPT4_1(OpenAIClient):
    name = "GPT-4.1"
    model_identifier = "gpt-4.1"
    rate_limit = 3
class O1(OpenAIClient):
    name = "o1"
    model_identifier = "o1"
    rate_limit = 2

    # NOT SUPPORTED
    max_tokens = -1
    temperature = -1
class O3(OpenAIClient):
    name = "o3"
    model_identifier = "o3"
    rate_limit = 2
    reasoning_effort = "medium"

    # NOT SUPPORTED
    max_tokens = -1
    temperature = -1
class O3high(OpenAIClient):
    name = "o3-high"
    model_identifier = "o3"
    rate_limit = 2
    reasoning_effort = "high"
    detail = "high"

    # NOT SUPPORTED
    max_tokens = -1
    temperature = -1
class O4mini(OpenAIClient):
    name = "o4-mini"
    model_identifier = "o4-mini"
    rate_limit = 5

    # NOT SUPPORTED
    max_tokens = -1
    temperature = -1
class O4minihigh(OpenAIClient):
    name = "o4-mini-high"
    model_identifier = "o4-mini"
    rate_limit = 5
    reasoning_effort = "high"

    # NOT SUPPORTED
    max_tokens = -1
    temperature = -1

# OpenRouter Models
class Qwen25VL72b(OpenRouterClient):
    name = "Qwen2.5-VL-72B"
    model_identifier = "qwen/qwen2.5-vl-72b-instruct"
    rate_limit = 20
class Llama4Maverick(OpenRouterClient):
    name = "Llama 4 Maverick"
    model_identifier = "meta-llama/llama-4-maverick"
class Pixtral12b(OpenRouterClient): model_identifier = "mistralai/pixtral-12b"
class Gemma27b(OpenRouterClient):
    name = "Gemma 27B"
    model_identifier = "google/gemma-3-27b-it"
    rate_limit = 5
class Phi4Instruct(OpenRouterClient):
    name = "Phi 4 Instruct"
    model_identifier = "microsoft/phi-4-multimodal-instruct"