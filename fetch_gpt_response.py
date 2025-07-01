import os
import httpx
from typing import List, Dict, Tuple, TypeVar, Type, Set
from pydantic import BaseModel
from openai import OpenAI

GPT_RESPONSE_TYPE = TypeVar("GPT_RESPONSE_TYPE", bound=BaseModel)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
proxy = os.getenv("OPEN_AI_PROXY_HTTP")
api_key = os.getenv("OPENi_API_KEY")

class Scene(BaseModel):
    Time: str
    Location: str
    Subject: str

class FullScene(BaseModel):
    SceneList: list[Scene]

class Dialogue(BaseModel):
    Costar: str
    Protagonist: str

class FullDialogue(BaseModel):
    Background: str
    DialogueList: list[Dialogue]

def get_gpt_answer(
        message_schema: Type[GPT_RESPONSE_TYPE] | None,
        input: List[dict] | None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024*10
) -> GPT_RESPONSE_TYPE:
    client = OpenAI(http_client=httpx.Client(proxies=proxy), api_key=api_key)
    response = client.responses.parse(
        model=model,
        input=input,
        text_format=message_schema,
        max_output_tokens=max_tokens
    )
    result = response.output_parsed
    return result

