#!/usr/bin/env python3
"""
Multi-provider LLM abstraction layer.

Environment Variables:
    LLM_PROVIDER: Provider name (gemini, openai, anthropic) - required
    LLM_MODEL: Model name (optional, uses provider default or auto-selects)
    GOOGLE_API_KEY or GEMINI_API_KEY: API key for Gemini
    OPENAI_API_KEY: API key for OpenAI
    ANTHROPIC_API_KEY: API key for Anthropic
"""

import os
from abc import ABC, abstractmethod
from typing import Optional


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class ProviderNotFoundError(LLMError):
    """Raised when the specified provider is not found."""
    pass


class ModelNotFoundError(LLMError):
    """Raised when the specified model is not found or not supported."""
    pass


class ConfigurationError(LLMError):
    """Raised when there's a configuration issue."""
    pass


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model: Optional[str] = None):
        self._requested_model = model
        self._model: Optional[str] = None
        self._client = None
        self._setup()
        self._select_model()

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the default model for this provider."""
        pass

    @property
    def model(self) -> str:
        """Return the currently selected model."""
        return self._model

    @abstractmethod
    def _setup(self) -> None:
        """Initialize the provider client."""
        pass

    @abstractmethod
    def _select_model(self) -> None:
        """Select and validate the model."""
        pass

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Generate text from a prompt."""
        pass


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider with dynamic model discovery."""

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def default_model(self) -> str:
        return "gemini-1.5-flash"

    def _setup(self) -> None:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "GOOGLE_API_KEY or GEMINI_API_KEY environment variable is required for Gemini"
            )

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._genai = genai
        except ImportError:
            raise ConfigurationError(
                "google-generativeai package is required. Install with: pip install google-generativeai"
            )

    def _get_supported_models(self) -> list[str]:
        """Dynamically list models that support generateContent."""
        supported = []
        try:
            for model in self._genai.list_models():
                if "generateContent" in model.supported_generation_methods:
                    model_name = model.name.replace("models/", "")
                    supported.append(model_name)
        except Exception as e:
            raise LLMError(f"Failed to list Gemini models: {e}")
        return supported

    def _select_model(self) -> None:
        supported_models = self._get_supported_models()

        if not supported_models:
            raise ModelNotFoundError("No Gemini models available that support generateContent")

        if self._requested_model is None:
            if self.default_model in supported_models:
                self._model = self.default_model
            else:
                self._model = supported_models[0]
        else:
            matching = [m for m in supported_models if self._requested_model in m]
            if matching:
                self._model = matching[0]
            elif self._requested_model in supported_models:
                self._model = self._requested_model
            else:
                raise ModelNotFoundError(
                    f"Model '{self._requested_model}' not found or doesn't support generateContent. "
                    f"Available models: {', '.join(supported_models[:10])}"
                )

        self._client = self._genai.GenerativeModel(self._model)

    def generate_text(self, prompt: str) -> str:
        try:
            response = self._client.generate_content(prompt)
            return response.text
        except Exception as e:
            raise LLMError(f"Gemini generation failed: {e}")


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation."""

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return "gpt-4o-mini"

    def _setup(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY environment variable is required for OpenAI")

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
        except ImportError:
            raise ConfigurationError(
                "openai package is required. Install with: pip install openai"
            )

    def _select_model(self) -> None:
        if self._requested_model is None:
            self._model = self.default_model
            return

        try:
            models = self._client.models.list()
            available_ids = [m.id for m in models.data]
            if self._requested_model in available_ids:
                self._model = self._requested_model
            else:
                matching = [m for m in available_ids if self._requested_model in m]
                if matching:
                    self._model = matching[0]
                else:
                    self._model = self.default_model
        except Exception:
            self._model = self._requested_model or self.default_model

    def generate_text(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            if "model" in str(e).lower():
                raise ModelNotFoundError(f"Model '{self._model}' error: {e}")
            raise LLMError(f"OpenAI generation failed: {e}")


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation."""

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return "claude-3-5-sonnet-20241022"

    def _setup(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "ANTHROPIC_API_KEY environment variable is required for Anthropic"
            )

        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ConfigurationError(
                "anthropic package is required. Install with: pip install anthropic"
            )

    def _select_model(self) -> None:
        self._model = self._requested_model or self.default_model

    def generate_text(self, prompt: str) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            error_str = str(e).lower()
            if "model" in error_str and ("not found" in error_str or "invalid" in error_str):
                raise ModelNotFoundError(f"Model '{self._model}' not found: {e}")
            raise LLMError(f"Anthropic generation failed: {e}")


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""

    _providers: dict[str, type[BaseLLMProvider]] = {
        "gemini": GeminiProvider,
        "google": GeminiProvider,
        "openai": OpenAIProvider,
        "gpt": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "claude": AnthropicProvider,
    }

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseLLMProvider]) -> None:
        """Register a custom provider."""
        cls._providers[name.lower()] = provider_class

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Return list of unique available provider names."""
        return list(set(cls._providers.values()))

    @classmethod
    def get_provider_names(cls) -> list[str]:
        """Return list of provider name aliases."""
        return ["gemini", "openai", "anthropic"]

    @classmethod
    def create(
        cls,
        provider_name: Optional[str] = None,
        model: Optional[str] = None
    ) -> BaseLLMProvider:
        """Create a provider instance from env vars or explicit params."""
        if provider_name is None:
            provider_name = os.environ.get("LLM_PROVIDER")

        if not provider_name:
            raise ConfigurationError(
                f"LLM_PROVIDER environment variable is required. "
                f"Available providers: {', '.join(cls.get_provider_names())}"
            )

        provider_name = provider_name.lower().strip()
        if provider_name not in cls._providers:
            raise ProviderNotFoundError(
                f"Unknown provider '{provider_name}'. "
                f"Available providers: {', '.join(cls.get_provider_names())}"
            )

        if model is None:
            model = os.environ.get("LLM_MODEL")

        return cls._providers[provider_name](model=model)


_provider_instance: Optional[BaseLLMProvider] = None


def _get_provider() -> BaseLLMProvider:
    """Get or create the singleton provider instance."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = LLMProviderFactory.create()
    return _provider_instance


def generate(prompt: str) -> str:
    """
    Generate text from a prompt using the configured LLM provider.

    Configure via environment variables:
        LLM_PROVIDER: gemini, openai, or anthropic
        LLM_MODEL: (optional) specific model name
        + appropriate API key for the provider

    Args:
        prompt: The input prompt for text generation.

    Returns:
        The generated text response.

    Raises:
        ConfigurationError: If provider is not configured properly.
        ModelNotFoundError: If the specified model is not available.
        LLMError: If text generation fails.
    """
    provider = _get_provider()
    return provider.generate_text(prompt)


def reset_provider() -> None:
    """Reset the provider instance (useful for reconfiguration)."""
    global _provider_instance
    _provider_instance = None


def get_provider_info() -> dict:
    """Get information about the current provider configuration."""
    try:
        provider = _get_provider()
        return {
            "provider": provider.provider_name,
            "model": provider.model,
            "status": "ready"
        }
    except LLMError as e:
        return {
            "provider": os.environ.get("LLM_PROVIDER", "not set"),
            "model": os.environ.get("LLM_MODEL", "not set"),
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Multi-Provider LLM Module")
        print("\nUsage: python llm_provider.py 'your prompt here'")
        print("\nEnvironment variables:")
        print("  LLM_PROVIDER: gemini, openai, or anthropic (required)")
        print("  LLM_MODEL: model name (optional, auto-selects if not set)")
        print("  GOOGLE_API_KEY or GEMINI_API_KEY: for Gemini")
        print("  OPENAI_API_KEY: for OpenAI")
        print("  ANTHROPIC_API_KEY: for Anthropic")
        print("\nExample:")
        print("  LLM_PROVIDER=gemini python llm_provider.py 'Hello, world!'")
        sys.exit(0)

    prompt = " ".join(sys.argv[1:])

    try:
        info = get_provider_info()
        print(f"Provider: {info['provider']} | Model: {info['model']}")
        print("-" * 50)
        response = generate(prompt)
        print(response)
    except LLMError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
