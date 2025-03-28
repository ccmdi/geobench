import base64
import requests
import os
from abc import ABC, abstractmethod
from ratelimit import limits, sleep_and_retry

def get_image_media_type(image_path: str) -> str:
    """Determine the media type based on file extension"""
    ext = os.path.splitext(image_path)[1].lower()
    if ext == '.png':
        return "image/png"
    elif ext in ['.jpg', '.jpeg']:
        return "image/jpeg"
    else:
        print(f"Warning: Unrecognized image extension '{ext}' for {image_path}. Defaulting to image/jpeg.")
        return "image/jpeg"

class MultimodalModel(ABC):
    def __init__(self):
        self.name = self.__class__.__name__
    
    @abstractmethod
    def query(self, image_path: str, prompt: str) -> str:
        """Query the model with an image and prompt, return the response text"""
        pass


class Claude3_5Haiku(MultimodalModel):
    """Claude 3.5 Haiku"""
    api_key_name = "ANTHROPIC_API_KEY"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        
    def query(self, image_path: str, prompt: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        media_type = get_image_media_type(image_path)
        
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")
            
        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_data}}
                    ]
                }
            ]
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            print(f"API error: {response.text}")
            raise Exception(f"Claude API error: {response.status_code}")
            
        return response.json()["content"][0]["text"]

class Claude3_7Sonnet(MultimodalModel):
    """Claude 3.7 Sonnet"""
    api_key_name = "ANTHROPIC_API_KEY"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        
    def query(self, image_path: str, prompt: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        media_type = get_image_media_type(image_path)
        
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")

        payload = {
            "model": "claude-3-7-sonnet-20250219",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_data}}
                    ]
                }
            ]
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            print(f"API error: {response.text}")
            raise Exception(f"Claude API error: {response.status_code}")
            
        return response.json()["content"][0]["text"]

class Claude3_7SonnetThinking(MultimodalModel):
    """Claude 3.7 Sonnet + extended thinking"""
    api_key_name = "ANTHROPIC_API_KEY"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        
    def query(self, image_path: str, prompt: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "output-128k-2025-02-19",
            "content-type": "application/json"
        }

        media_type = get_image_media_type(image_path)
        
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")

        payload = {
            "model": "claude-3-7-sonnet-20250219",
            "max_tokens": 16000,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 15000
            },
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_data}}
                    ]
                }
            ]
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload
        )
        
        response_json = response.json()
    
        thinking_text = ""
        response_text = ""
        
        # The content is an array of blocks
        if "content" in response_json and len(response_json["content"]) > 0:
            # Process each content block based on type
            for block in response_json["content"]:
                if block["type"] == "thinking":
                    thinking_text = block.get("thinking", "")
                elif block["type"] == "text":
                    response_text = block.get("text", "")
        
        # Combine thinking and response
        #TODO: separate file
        if thinking_text:
            combined_response = f"<thinking>{thinking_text}</thinking>\n\n{response_text}"
        else:
            combined_response = response_text
        
        if not combined_response:
            print(f"Unexpected response structure: {response_json}")
            return "Error: Couldn't extract response text"
        
        return combined_response

class Claude3_5Sonnet(MultimodalModel):
    """Claude 3.5 Sonnet"""
    api_key_name = "ANTHROPIC_API_KEY"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        
    def query(self, image_path: str, prompt: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        media_type = get_image_media_type(image_path)
        
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")

        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_data}}
                    ]
                }
            ]
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            print(f"API error: {response.text}")
            raise Exception(f"Claude API error: {response.status_code}")
            
        return response.json()["content"][0]["text"]

class Gemini1_5Flash(MultimodalModel):
    """Gemini 1.5 Flash"""
    api_key_name = "GEMINI_API_KEY"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
    
    @sleep_and_retry
    @limits(calls=10, period=60)
    def query(self, image_path: str, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        
        media_type = get_image_media_type(image_path)
        
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": media_type,
                                "data": img_data
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 1000
            }
        }
        
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        
        response = requests.post(
            url,
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            print(f"API error: {response.text}")
            raise Exception(f"Gemini API error: {response.status_code}")
        
        response_json = response.json()
        
        if 'candidates' in response_json and len(response_json['candidates']) > 0:
            candidate = response_json['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                return candidate['content']['parts'][0]['text']
        
        raise Exception("Could not extract text from Gemini response")

class Gemini1_5Pro(MultimodalModel):
    """Gemini 1.5 Pro"""
    api_key_name = "GEMINI_API_KEY"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        
    def query(self, image_path: str, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        
        media_type = get_image_media_type(image_path)
        
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": media_type,
                                "data": img_data
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 1000
            }
        }
        
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent?key={self.api_key}"
        
        response = requests.post(
            url,
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            print(f"API error: {response.text}")
            raise Exception(f"Gemini API error: {response.status_code}")
        
        response_json = response.json()
        
        if 'candidates' in response_json and len(response_json['candidates']) > 0:
            candidate = response_json['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                return candidate['content']['parts'][0]['text']
        
        raise Exception("Could not extract text from Gemini response")

class Gemini2Flash(MultimodalModel):
    """Gemini 2.0 Flash"""
    api_key_name = "GEMINI_API_KEY"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
    
    @sleep_and_retry
    @limits(calls=10, period=60)
    def query(self, image_path: str, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        
        media_type = get_image_media_type(image_path)
        
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")
            
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": media_type,
                                "data": img_data
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 1000
            }
        }
        
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        
        response = requests.post(
            url,
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            print(f"API error: {response.text}")
            raise Exception(f"Gemini API error: {response.status_code}")
        
        response_json = response.json()
        
        # Extract text from the response
        if 'candidates' in response_json and len(response_json['candidates']) > 0:
            candidate = response_json['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                return candidate['content']['parts'][0]['text']
        
        raise Exception("Could not extract text from Gemini response")   

class Gemini2ProExp(MultimodalModel):
    """Gemini 2.0 Pro Experimental"""
    api_key_name = "GEMINI_API_KEY"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
    
    @sleep_and_retry
    @limits(calls=2, period=60)
    def query(self, image_path: str, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        
        media_type = get_image_media_type(image_path)
        
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")
            
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": media_type,
                                "data": img_data
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 1000
            }
        }
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-pro-exp:generateContent?key={self.api_key}"
        
        response = requests.post(
            url,
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            print(f"API error: {response.text}")
            raise Exception(f"Gemini API error: {response.status_code}")
        
        response_json = response.json()
        
        # Extract text from the response
        if 'candidates' in response_json and len(response_json['candidates']) > 0:
            candidate = response_json['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                return candidate['content']['parts'][0]['text']
        
        raise Exception("Could not extract text from Gemini response")
    
class GPT4o(MultimodalModel):
    """OpenAI's GPT-4o multimodal model using direct REST API calls"""
    api_key_name = "OPENAI_API_KEY"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        
    def query(self, image_path: str, prompt: str) -> str:
        media_type = get_image_media_type(image_path)
        
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")

        image_url = f"data:{media_type};base64,{img_data}"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.4
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                print(f"API error: {response.text}")
                raise Exception(f"OpenAI API error: {response.status_code}")
                
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            raise Exception(f"OpenAI API error: {str(e)}")

class GPT4oMini(MultimodalModel):
    """OpenAI's GPT-4o-mini model - smaller and faster than GPT-4o"""
    api_key_name = "OPENAI_API_KEY"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
    
    @sleep_and_retry
    @limits(calls=5, period=60)
    def query(self, image_path: str, prompt: str) -> str:
        media_type = get_image_media_type(image_path)
        
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")

        image_url = f"data:{media_type};base64,{img_data}"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.4
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                print(f"API error: {response.text}")
                raise Exception(f"OpenAI API error: {response.status_code}")
                
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            raise Exception(f"OpenAI API error: {str(e)}")
            
class Llama90bVision(MultimodalModel):
    """Llama 3.2 90B Vision model"""
    api_key_name = "REPLICATE_API_TOKEN"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        os.environ["REPLICATE_API_TOKEN"] = api_key
        
    def query(self, image_path: str, prompt: str) -> str:
        import replicate
        
        media_type = get_image_media_type(image_path)

        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")
            
        image_url = f"data:{media_type};base64,{img_data}"
        
        input = {
            "prompt": prompt,
            "image": image_url,
            "temperature": 0.4,
            "max_tokens": 1000
        }
        
        try:
            output = replicate.run(
                "lucataco/ollama-llama3.2-vision-90b:54202b223d5351c5afe5c0c9dba2b3042293b839d022e76f53d66ab30b9dc814",
                input=input
            )
            
            if hasattr(output, '__iter__') and not isinstance(output, str):
                return "".join(output)
            
            return output
            
        except Exception as e:
            print(f"Error with replicate.run: {str(e)}")
            
            prediction = replicate.predictions.create(
                version="lucataco/ollama-llama3.2-vision-90b:54202b223d5351c5afe5c0c9dba2b3042293b839d022e76f53d66ab30b9dc814",
                input=input
            )
            
            prediction = replicate.predictions.wait(prediction.id)
            
            if prediction.status == "succeeded":
                if isinstance(prediction.output, list):
                    return "".join(prediction.output)
                return prediction.output
            else:
                error_msg = prediction.error or "Unknown error"
                raise Exception(f"Replicate prediction failed: {error_msg}")

class Qwen25VL72b(MultimodalModel):
    """Qwen 2.5 VL model via OpenRouter API"""
    api_key_name = "OPENROUTER_API_KEY"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
    
    @sleep_and_retry
    @limits(calls=20, period=60)
    def query(self, image_path: str, prompt: str) -> str:
        media_type = get_image_media_type(image_path)
        
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")

        image_url = f"data:{media_type};base64,{img_data}"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://geobench.org"
        }
        
        payload = {
            "model": "qwen/qwen2.5-vl-72b-instruct",
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.4
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                print(f"API error: {response.text}")
                raise Exception(f"OpenRouter API error: {response.status_code}")
                
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"OpenRouter API error: {str(e)}")
            raise Exception(f"OpenRouter API error: {str(e)}")
        
class O1(MultimodalModel):
    """OpenAI's o1 model"""
    api_key_name = "OPENROUTER_API_KEY"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
    
    @sleep_and_retry
    @limits(calls=10, period=60)
    def query(self, image_path: str, prompt: str) -> str:
        media_type = get_image_media_type(image_path)
        
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")

        image_url = f"data:{media_type};base64,{img_data}"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://geobench.org"
        }
        
        payload = {
            "model": "openai/o1",
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "max_tokens": 16046,
            "temperature": 0.4
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                print(f"API error: {response.text}")
                raise Exception(f"OpenRouter API error: {response.status_code}")
                
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"OpenRouter API error: {str(e)}")
            raise Exception(f"OpenRouter API error: {str(e)}")

class Gemini2FlashThinkingExp(MultimodalModel):
    api_key_name = "OPENROUTER_API_KEY"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
    
    @sleep_and_retry
    @limits(calls=3, period=60)
    def query(self, image_path: str, prompt: str) -> str:
        media_type = get_image_media_type(image_path)
        
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")

        image_url = f"data:{media_type};base64,{img_data}"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://geobench.org"
        }
        
        payload = {
            "model": "google/gemini-2.0-flash-thinking-exp:free",
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "max_tokens": 16046,
            "temperature": 0.4
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                print(f"API error: {response.text}")
                raise Exception(f"OpenRouter API error: {response.status_code}")
                
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"OpenRouter API error: {str(e)}")
            raise Exception(f"OpenRouter API error: {str(e)}")
        
class Pixtral12b(MultimodalModel):
    api_key_name = "OPENROUTER_API_KEY"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
    
    def query(self, image_path: str, prompt: str) -> str:
        media_type = get_image_media_type(image_path)
        
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")

        image_url = f"data:{media_type};base64,{img_data}"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://geobench.org"
        }
        
        payload = {
            "model": "mistralai/pixtral-12b",
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.4
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                print(f"API error: {response.text}")
                raise Exception(f"OpenRouter API error: {response.status_code}")
                
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"OpenRouter API error: {str(e)}")
            raise Exception(f"OpenRouter API error: {str(e)}")
        
class Gemma27b(MultimodalModel):
    api_key_name = "OPENROUTER_API_KEY"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
    
    @sleep_and_retry
    @limits(calls=10, period=60)
    def query(self, image_path: str, prompt: str) -> str:
        media_type = get_image_media_type(image_path)
        
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")

        image_url = f"data:{media_type};base64,{img_data}"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://geobench.org"
        }
        
        payload = {
            "model": "google/gemma-3-27b-it:free",
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.4
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                print(f"API error: {response.text}")
                raise Exception(f"OpenRouter API error: {response.status_code}")
                
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"OpenRouter API error: {str(e)}")
            raise Exception(f"OpenRouter API error: {str(e)}")

class Gemini2_5ProExp(MultimodalModel):
    api_key_name = "OPENROUTER_API_KEY"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
    
    @sleep_and_retry
    @limits(calls=1, period=60)
    def query(self, image_path: str, prompt: str) -> str:
        media_type = get_image_media_type(image_path)
        
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")

        image_url = f"data:{media_type};base64,{img_data}"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://geobench.org"
        }
        
        payload = {
            "model": "google/gemini-2.5-pro-exp-03-25:free",
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "max_tokens": 16046,
            "temperature": 0.4
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                print(f"API error: {response.text}")
                raise Exception(f"OpenRouter API error: {response.status_code}")
                
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"OpenRouter API error: {str(e)}")
            raise Exception(f"OpenRouter API error: {str(e)}")

class Phi4Instruct(MultimodalModel):
    api_key_name = "OPENROUTER_API_KEY"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key

    def query(self, image_path: str, prompt: str) -> str:
        media_type = get_image_media_type(image_path)
        
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")

        image_url = f"data:{media_type};base64,{img_data}"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://geobench.org",  # Required by OpenRouter
            # "X-Title": "Model Benchmarking"  # Optional
        }
        
        payload = {
            "model": "microsoft/phi-4-multimodal-instruct",
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.4
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                print(f"API error: {response.text}")
                raise Exception(f"OpenRouter API error: {response.status_code}")
                
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"OpenRouter API error: {str(e)}")
            raise Exception(f"OpenRouter API error: {str(e)}")