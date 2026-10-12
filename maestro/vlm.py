"""The vision-language models behind Maestro: the planner, object pointing and part selection.

`vlm.provider` in config/maestro.yaml selects the models for all three roles:
  gemini  Gemini plans and points; OpenAI o3 picks among the numbered points on an object part.
  astra   GPT-6 Astra does all three.
"""
from __future__ import annotations

import base64
import functools
import io
import json
import os
from typing import Annotated

import cv2
import PIL.Image as Image
from pydantic import BaseModel, Field

from maestro.config import cfg

PROVIDER = cfg.vlm.provider
if PROVIDER not in ("gemini", "astra"):
    raise ValueError(f"vlm.provider must be 'gemini' or 'astra', not {PROVIDER!r}")
SETTINGS = cfg.vlm[PROVIDER]
POINTING_MODEL = SETTINGS['pointing'] if PROVIDER == "gemini" else SETTINGS['model']


@functools.cache
def _openai():
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in .env")
    return OpenAI(timeout=300.0, max_retries=1)


@functools.cache
def _gemini():
    from google import genai
    if not os.environ.get("GEMINI_KEY"):
        raise RuntimeError("Set GEMINI_KEY in .env")
    return genai.Client(api_key=os.environ["GEMINI_KEY"], http_options={'api_version': SETTINGS['api_version']})


def _openai_content(parts):
    """OpenAI Responses input parts for a list of strings and PIL images."""
    content = []
    for part in parts:
        if isinstance(part, str):
            content.append({"type": "input_text", "text": part})
        elif isinstance(part, Image.Image):
            buffer = io.BytesIO()
            part.convert("RGB").save(buffer, format="PNG")
            content.append({"type": "input_image", "detail": "high",
                            "image_url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()})
        else:
            raise TypeError("VLM messages contain only strings and PIL images")
    return content


def _astra_parse(content, schema, history=()):
    """A validated structured Astra response; partial or refused output raises."""
    response = _openai().responses.parse(
        model=SETTINGS['model'],
        input=[*history, {"role": "user", "content": content}],
        text_format=schema,
        reasoning={"effort": SETTINGS['reasoning_effort']},
        max_output_tokens=16384,
        store=False,
    )
    if response.status != "completed":
        raise RuntimeError(f"{SETTINGS['model']} response was {response.status}")
    if not isinstance(response.output_parsed, schema):
        raise RuntimeError(f"{SETTINGS['model']} returned no valid {schema.__name__} (possibly refused)")
    return response


# ---------- planner ----------

class _GeminiChat:
    def __init__(self):
        from google.genai import types
        from maestro.chat import Chats
        config = types.GenerateContentConfig(temperature=float(SETTINGS['temperature']))
        self._chat = Chats(_gemini()._models).create(model=SETTINGS['planner'], config=config)
        self._config = config

    def send(self, message, schema):
        config = dict(self._config)
        config["response_mime_type"] = "application/json"
        config["response_schema"] = list[schema]
        response = self._chat.send_message(message=message, config=config)
        return response.parsed[0], response.text


class _AstraChat:
    def __init__(self):
        self._history = []

    def send(self, message, schema):
        content = _openai_content(message)
        response = _astra_parse(content, schema, self._history)
        # Like the Gemini chat, the history keeps the text of past turns but not their images.
        self._history.append({"role": "user", "content": [p for p in content if p["type"] == "input_text"]})
        self._history.append({"role": "assistant", "content": response.output_text})
        return response.output_parsed, response.output_text


def planner_chat():
    """A planner conversation: `send(message, schema)` returns (parsed schema, raw text).

    A message is a list of strings and PIL images; earlier turns keep only their text.
    """
    return _GeminiChat() if PROVIDER == "gemini" else _AstraChat()


# ---------- pointing ----------

Coordinate = Annotated[float, Field(ge=0, le=1000)]


class _PointDetection(BaseModel):
    point: list[Coordinate] = Field(min_length=2, max_length=2)
    label: str = Field(min_length=1)
    pose_index: int = Field(ge=1)


class _Detections(BaseModel):
    detections: list[_PointDetection]


def point(images, labels):
    """Points on `labels` across wrist images (RGB arrays).

    Returns [{'point': [y, x] normalized to 0-1000, 'label': str, 'pose_index': 1-based}].
    """
    n = len(images)
    if PROVIDER == "gemini":
        from google.genai import types
        prompt = f"""You are given {n} RGB images from different wrist camera poses (pose 1 to pose {n}).
Get all points matching the following objects: {', '.join(labels)}.
The label returned should be an identifying name for the object detected.
Also return which pose/image index (1-{n}) the object was detected in. You should always try your best to point at the center of the object.
The answer should follow the json format:

[{{"point": [y, x], "label": <object_name>, "pose_index": <1-{n}>}}, ...]

The points are in [y, x] format normalized to 0-1000.
If an object appears in multiple poses, return the best detection with the clearest view.
"""
        contents = [types.Part.from_bytes(data=cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes(),
                                          mime_type='image/jpeg') for image in images]
        response = _gemini().models.generate_content(
            model=POINTING_MODEL, contents=contents + [prompt],
            config=types.GenerateContentConfig(temperature=0.0))
        text = response.text
        print(f"Pointing response from {POINTING_MODEL}: {text[:300]}...")
        start, end = text.find('['), text.rfind(']')
        return json.loads(text[start:end + 1]) if start != -1 and end != -1 else []

    prompt = f"""You are given {n} RGB images from different wrist camera poses (pose 1 to pose {n}).
Get all points matching the following objects: {', '.join(labels)}.
Copy each returned label exactly from the requested object labels; do not rephrase it.
Also return which pose/image index (1-{n}) the object was detected in. You should always try your best to point at the center of the object.
Return a JSON object with a detections list, each containing point, label and pose_index.
The points are in [y, x] format normalized to 0-1000, NOT [x, y].
If an object appears in multiple poses, return the best detection with the clearest view.
Omit objects that are not visible; return an empty detections list if none are visible.
"""
    response = _astra_parse(_openai_content([Image.fromarray(image) for image in images] + [prompt]), _Detections)
    detections = [d.model_dump() for d in response.output_parsed.detections if d.pose_index <= n]
    print(f"Pointing response from {POINTING_MODEL}: {detections}")
    return detections


# ---------- part selection ----------

class _Selection(BaseModel):
    index: int
    label: str


class _Selections(BaseModel):
    points: list[_Selection]


SELECTION_PROMPT = ("You are given an image with some numbered points and a language instruction. "
                    "Based on the instruction and the image, return a list of indices of the points. "
                    "Please only output a list of points in format of "
                    "[{\"index\": 0, \"label\": \"left top corner of the cube\"}].")


def select_points(image, points, instruction):
    """The numbered points on an annotated image (PIL) that match `instruction`.

    `points` describes the candidates; returns [{'index': int, 'label': str}].
    """
    if PROVIDER == "gemini":
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        response = _openai().responses.create(model=SETTINGS['part_selection'], input=[{"role": "user", "content": [
            {"type": "input_text", "text": SELECTION_PROMPT},
            {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()},
            {"type": "input_text", "text": str(points)},
            {"type": "input_text", "text": instruction},
        ]}])
        print(response.output_text)
        text = response.output_text
        start, end = text.find('['), text.rfind(']')
        return json.loads(text[start:end + 1]) if start != -1 and end != -1 else []

    response = _astra_parse(_openai_content([SELECTION_PROMPT, image, str(points), instruction]), _Selections)
    print(response.output_text)
    return [s.model_dump() for s in response.output_parsed.points]
