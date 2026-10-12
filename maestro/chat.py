# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
from typing import Optional, Union, get_args

from google.genai import _transformers as t
from google.genai import types
from google.genai.models import Models
from google.genai.types import Content, ContentOrDict, GenerateContentConfigOrDict, GenerateContentResponse, Part, PartUnionDict


if sys.version_info >= (3, 10):
  from typing import TypeGuard
else:
  from typing_extensions import TypeGuard


def _validate_content(content: Content) -> bool:
  if not content.parts:
    return False
  for part in content.parts:
    if part == Part():
      return False
    if part.text is not None and part.text == "":
      return False
  return True


def _validate_response(response: GenerateContentResponse) -> bool:
  if not response.candidates:
    return False
  if not response.candidates[0].content:
    return False
  return _validate_content(response.candidates[0].content)


def _extract_curated_history(
    comprehensive_history: list[Content],
) -> list[Content]:
  """Extracts the curated (valid) history from a comprehensive history.

  The comprehensive history contains all turns (user input and model responses),
  including any invalid or rejected model outputs.  This function filters
  that history to return only the valid turns.

  A "turn" starts with one user input (a single content) and then follows by
  corresponding model response (which may consist of multiple contents).
  Turns are assumed to alternate: user input, model output, user input, model
  output, etc.

  Args:
      comprehensive_history: A list representing the complete chat history.
        Including invalid turns.

  Returns:
      curated history, which is a list of valid turns.
  """
  if not comprehensive_history:
    return []
  curated_history = []
  length = len(comprehensive_history)
  i = 0
  current_input = comprehensive_history[i]
  if current_input.role != "user":
    raise ValueError("History must start with a user turn.")
  while i < length:
    if comprehensive_history[i].role not in ["user", "model"]:
      raise ValueError(
          f"Role must be user or model, but got {comprehensive_history[i].role}"
      )

    if comprehensive_history[i].role == "user":
      current_input = comprehensive_history[i]
      i += 1
    else:
      current_output = []
      is_valid = True
      while i < length and comprehensive_history[i].role == "model":
        current_output.append(comprehensive_history[i])
        if is_valid and not _validate_content(comprehensive_history[i]):
          is_valid = False
        i += 1
      if is_valid:
        curated_history.append(current_input)
        curated_history.extend(current_output)
  return curated_history


class _BaseChat:
  """Base chat session."""

  def __init__(
      self,
      *,
      model: str,
      config: Optional[GenerateContentConfigOrDict] = None,
      history: list[ContentOrDict],
  ):
    self._model = model
    self._config = config
    content_models = []
    for content in history:
      if not isinstance(content, Content):
        content_model = Content.model_validate(content)
      else:
        content_model = content
      content_models.append(content_model)
    self._comprehensive_history = content_models
    """Comprehensive history is the full history of the chat, including turns of the invalid contents from the model and their associated inputs.
    """
    self._curated_history = _extract_curated_history(content_models)
    """Curated history is the set of valid turns that will be used in the subsequent send requests.
    """

  def record_history(
      self,
      user_input: Content,
      model_output: list[Content],
      automatic_function_calling_history: list[Content],
      is_valid: bool,
  ) -> None:
    """Records the chat history.

    Maintaining both comprehensive and curated histories.

    Args:
      user_input: The user's input content.
      model_output: A list of `Content` from the model's response. This can be
        an empty list if the model produced no output.
      automatic_function_calling_history: A list of `Content` representing the
        history of automatic function calls, including the user input as the
        first entry.
      is_valid: A boolean flag indicating whether the current model output is
        considered valid.
    """
    input_contents = (
        # Because the AFC input contains the entire curated chat history in
        # addition to the new user input, we need to truncate the AFC history
        # to deduplicate the existing chat history.
        automatic_function_calling_history[len(self._curated_history) :]
        if automatic_function_calling_history
        else [user_input]
    )
    # Appends an empty content when model returns empty response, so that the
    # history is always alternating between user and model.
    output_contents = (
        model_output if model_output else [Content(role="model", parts=[])]
    )
    for i in range(len(input_contents)):
        input_contents[i].parts = [part for part in input_contents[i].parts if part.text]
    for i in range(len(output_contents)):
        output_contents[i].parts = [part for part in output_contents[i].parts if part.text]
      
    self._comprehensive_history.extend(input_contents)
    self._comprehensive_history.extend(output_contents)
    if is_valid:
      self._curated_history.extend(input_contents)
      self._curated_history.extend(output_contents)


def _is_part_type(
    contents: Union[list[PartUnionDict], PartUnionDict],
) -> TypeGuard[t.ContentType]:
  if isinstance(contents, list):
    return all(_is_part_type(part) for part in contents)
  else:
    allowed_part_types = get_args(types.PartUnion)
    if type(contents) in allowed_part_types:
      return True
    else:
      # Some images don't pass isinstance(item, PIL.Image.Image)
      # For example <class 'PIL.JpegImagePlugin.JpegImageFile'>
      if types.PIL_Image is not None and isinstance(contents, types.PIL_Image):
        return True
    return False


class Chat(_BaseChat):
  """Chat session."""

  def __init__(
      self,
      *,
      modules: Models,
      model: str,
      config: Optional[GenerateContentConfigOrDict] = None,
      history: list[ContentOrDict],
  ):
    self._modules = modules
    super().__init__(
        model=model,
        config=config,
        history=history,
    )

  def send_message(
      self,
      message: Union[list[PartUnionDict], PartUnionDict],
      config: Optional[GenerateContentConfigOrDict] = None,
  ) -> GenerateContentResponse:
    """Sends the conversation history with the additional message and returns the model's response.

    Args:
      message: The message to send to the model.
      config:  Optional config to override the default Chat config for this
        request.

    Returns:
      The model's response.

    Usage:

    .. code-block:: python

      chat = client.chats.create(model='gemini-2.0-flash')
      response = chat.send_message('tell me a story')
    """

    if not _is_part_type(message):
      raise ValueError(
          f"Message must be a valid part type: {types.PartUnion} or"
          f" {types.PartUnionDict}, got {type(message)}"
      )
    input_content = t.t_content(self._modules._api_client, message)
    response = self._modules.generate_content(
        model=self._model,
        contents=self._curated_history + [input_content],  # type: ignore[arg-type]
        config=config if config else self._config,
    )
    model_output = (
        [response.candidates[0].content]
        if response.candidates and response.candidates[0].content
        else []
    )
    automatic_function_calling_history = (
        response.automatic_function_calling_history
        if response.automatic_function_calling_history
        else []
    )
    self.record_history(
        user_input=input_content,
        model_output=model_output,
        automatic_function_calling_history=automatic_function_calling_history,
        is_valid=_validate_response(response),
    )
    return response


class Chats:
  """A util class to create chat sessions."""

  def __init__(self, modules: Models):
    self._modules = modules

  def create(
      self,
      *,
      model: str,
      config: Optional[GenerateContentConfigOrDict] = None,
      history: Optional[list[ContentOrDict]] = None,
  ) -> Chat:
    """Creates a new chat session.

    Args:
      model: The model to use for the chat.
      config: The configuration to use for the generate content request.
      history: The history to use for the chat.

    Returns:
      A new chat session.
    """
    return Chat(
        modules=self._modules,
        model=model,
        config=config,
        history=history if history else [],
    )
