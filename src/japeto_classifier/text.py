from __future__ import annotations


def normalize_natural_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\x00", " ").split())


def build_model_text(
    chatbot_response: object,
    user_message: object | None = None,
    mode: str = "response_only",
) -> str:
    response = normalize_natural_text(chatbot_response)
    if not response:
        raise ValueError("chatbot_response must not be empty")
    if mode == "response_only":
        return response
    if mode != "context_enhanced":
        raise ValueError(f"Unsupported input mode: {mode}")
    user = normalize_natural_text(user_message)
    if not user:
        raise ValueError("user_message is required for context_enhanced models")
    return f"User message:\n{user}\n\nChatbot response:\n{response}"

