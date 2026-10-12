
import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from app.schemas import ClassificationResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class PromptConfig:
    """Load the versioned YAML prompt configuration."""

    def __init__(self, prompt_path: Path):
        with prompt_path.open("r", encoding="utf-8") as file:
            self.data: dict[str, Any] = yaml.safe_load(file)

        self.version = self.data["version"]
        self.model = self.data["model"]
        self.system_prompt = self.data["system_prompt"]
        self.examples = self.data.get("few_shot_examples", [])


def mock_classify_email(email_text: str) -> ClassificationResult:
    """Return a predictable result without calling an API."""

    text = email_text.lower()

    if any(word in text for word in ["charged", "payment", "invoice", "refund", "bill"]):
        category = "billing"
        summary = "The customer has a billing or payment-related enquiry."
    elif any(word in text for word in ["crash", "error", "bug", "website", "upload"]):
        category = "technical"
        summary = "The customer reports a technical issue."
    elif any(word in text for word in ["login", "log in", "password", "account", "sign in"]):
        category = "account"
        summary = "The customer has an account access-related enquiry."
    else:
        category = "general"
        summary = "The customer has a general enquiry."

    return ClassificationResult(category=category, summary=summary)


def classify_email(
    email_text: str,
    prompt_path: Path | None = None,
) -> ClassificationResult:
    """Classify an email using mock mode or the OpenAI API."""

    if not email_text.strip():
        raise ValueError("Email text cannot be empty.")

  
    if os.getenv("CLASSIFIER_MODE", "mock").lower() == "mock":
        return mock_classify_email(email_text)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Check your .env file.")

    if prompt_path is None:
        prompt_path = PROJECT_ROOT / "prompts" / "support_v1.yaml"

    config = PromptConfig(prompt_path)
    client = OpenAI(api_key=api_key)

    messages = [{"role": "system", "content": config.system_prompt}]

    for example in config.examples:
        expected = ClassificationResult(
            category=example["category"],
            summary=example["summary"],
        )
        messages.append({"role": "user", "content": example["input"]})
        messages.append({
            "role": "assistant",
            "content": expected.model_dump_json(),
        })

    messages.append({"role": "user", "content": email_text})

    response = client.chat.completions.parse(
        model=config.model,
        messages=messages,
        response_format=ClassificationResult,
        temperature=0,
    )

    message = response.choices[0].message

    if message.refusal:
        raise RuntimeError(f"Model refused the request: {message.refusal}")

    if message.parsed is None:
        raise RuntimeError("No validated classification was returned.")

    return message.parsed
