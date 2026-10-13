
import json
import os
import urllib.error
import urllib.request
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

    if any(word in text for word in [
        "charged", "payment", "invoice", "refund", "bill"
    ]):
        category = "billing"
        summary = "The customer has a billing or payment-related enquiry."
    elif any(word in text for word in [
        "crash", "error", "bug", "website", "upload"
    ]):
        category = "technical"
        summary = "The customer reports a technical issue."
    elif any(word in text for word in [
        "login", "log in", "password", "account", "sign in"
    ]):
        category = "account"
        summary = "The customer has an account access-related enquiry."
    else:
        category = "general"
        summary = "The customer has a general enquiry."

    return ClassificationResult(category=category, summary=summary)


def ollama_classify_email(
    email_text: str,
    prompt_path: Path | None = None,
) -> ClassificationResult:
    """Classify an email using a locally running Ollama model."""

    if prompt_path is None:
        prompt_path = PROJECT_ROOT / "prompts" / "support_v1.yaml"

    config = PromptConfig(prompt_path)

    system_prompt = (
        config.system_prompt
        + "\nReturn only valid JSON with exactly these fields: "
        ' "category" and "summary". '
        'The category must be one of "billing", "technical", '
        '"account", or "general".'
    )

    messages = [{"role": "system", "content": system_prompt}]

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

    payload = {
        "model": os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct"),
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }

    request = urllib.request.Request(
        os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. Ensure Ollama is running "
            "and the model is installed."
        ) from exc

    content = result.get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
        return ClassificationResult.model_validate(parsed)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"Ollama returned an invalid classification: {content}"
        ) from exc


def classify_email(
    email_text: str,
    prompt_path: Path | None = None,
) -> ClassificationResult:
    """Classify using mock, local Ollama, or OpenAI mode."""

    if not email_text.strip():
        raise ValueError("Email text cannot be empty.")

    mode = os.getenv("CLASSIFIER_MODE", "mock").strip().lower()

    if mode == "mock":
        return mock_classify_email(email_text)

    if mode in ("ollama", "local"):
        return ollama_classify_email(email_text, prompt_path)

    if mode != "openai":
        raise ValueError(
            "Invalid CLASSIFIER_MODE. Use mock, ollama, or openai."
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Check your .env file."
        )

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
