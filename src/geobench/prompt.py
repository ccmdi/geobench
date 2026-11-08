import base64
import mimetypes
from pathlib import Path

import verifiers.v1 as vf

PROMPT = """You are playing a geolocation challenge. Work out where the provided image was taken.

1. Carefully analyze the image for clues about its location (architecture, signage, vegetation, terrain, etc.).
2. Think step by step about which country it is likely in, and why.
3. Estimate the latitude and longitude from your analysis.

Take your time to reason through the evidence. End your reply with your final answer as these three lines:

country: [country name]
lat: [latitude as a decimal number]
lng: [longitude as a decimal number]

Only the last country/lat/lng block in your reply is graded. The score falls with the distance between your guess and the true location. A reply without that block scores 0."""


def data_url(image: Path) -> str:
    media_type, _ = mimetypes.guess_type(image)
    return f"data:{media_type};base64,{base64.b64encode(image.read_bytes()).decode()}"


def user_message(image: Path) -> vf.UserMessage:
    return vf.UserMessage(
        content=[
            vf.TextContentPart(text=PROMPT),
            vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=data_url(image))),
        ]
    )
