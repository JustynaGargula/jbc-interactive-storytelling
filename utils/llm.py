from typing import Optional
from google import genai
import requests
import streamlit as st
from .general import display_error, print_llm_usage

DEFAULT_LLM_PROVIDER = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

def get_llm_config() -> dict:
    """
    Odczytuje konfigurację LLM ze Streamlit secrets.

    Domyślnie używa Gemini, więc brak sekcji [llm] zachowuje stare działanie aplikacji.
    """
    llm_config = st.secrets.get("llm", {})

    return {
        "provider": llm_config.get("provider", DEFAULT_LLM_PROVIDER).lower(),
        "gemini_model": llm_config.get("gemini_model", DEFAULT_GEMINI_MODEL),
        "gemini_api_key": llm_config.get("gemini_api_key"),
        "openrouter_model": llm_config.get("openrouter_model", DEFAULT_OPENROUTER_MODEL),
        "openrouter_api_key": llm_config.get("openrouter_api_key"),
    }

def call_gemini(prompt: str, model: str, api_key: Optional[str] = None) -> Optional[str]:
    """
    Obsługuje komunikację z Gemini.
    """
    page_text_part = st.session_state["page_text"].get("utils_handle_llm")
    try:
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            # The client gets the API key from the environment variable `GEMINI_API_KEY`.
            client = genai.Client()

        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )

        usage_metadata = getattr(response, "usage_metadata", None)

        prompt_tokens = getattr(usage_metadata, "prompt_token_count", None)
        completion_tokens = getattr(usage_metadata, "candidates_token_count", None)
        total_tokens = getattr(usage_metadata, "total_token_count", None)

        print_llm_usage(
            provider="gemini",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        client.close()
        return response.text

    except genai.errors.APIError as e:
        if e.code == 400:
            # 400 - błędne dane wejściowe
            display_error(400, page_text_part, e)
            return None
        elif e.code == 403:
            # 403 - problem z kluczem API
            display_error(403, page_text_part, e)
            return None
        elif e.code == 404:
            # 404 - model nie znaleziony
            display_error(404, page_text_part, e)
            return None
        elif e.code == 429:
            # 429 - przekroczono limit requestów
            display_error(429, page_text_part, e)
            return None
        elif e.code == 503:
            # 503 - serwer przeciążony
            display_error(503, page_text_part, e)
            return None

    except Exception as e:
        # Inne nieoczekiwane błędy
        st.error(f"{page_text_part.get('other_error')}{type(e).__name__}")
        with st.expander(page_text_part.get("other_error_details")):
            st.code(str(e))
        return None


def call_openrouter(prompt: str, model: str, api_key: Optional[str]) -> Optional[str]:
    """
    Obsługuje komunikację z OpenRouter.
    """
    if not api_key:
        st.error(
            "⚠️ **Brakuje klucza OpenRouter.**\n\n"
            "Ustaw `openrouter_api_key` w sekcji `[llm]` pliku `.streamlit/secrets.toml`."
        )
        return None

    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501",
                "X-OpenRouter-Title": "JBC Interactive Storytelling",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            },
            timeout=60,
        )

        if response.status_code in (401, 403):
            st.error("⚠️ **Problem z autoryzacją OpenRouter.**\n\nSprawdź `openrouter_api_key`.")
            return None

        if response.status_code == 402:
            st.error("⚠️ **Brak środków lub limitu kredytów na koncie OpenRouter.**")
            return None

        if response.status_code == 404:
            st.error("⚠️ **Nie znaleziono modelu OpenRouter.**\n\nSprawdź `openrouter_model`.")
            return None

        if response.status_code == 429:
            st.error("⚠️ **Przekroczono limit zapytań do OpenRouter.**\n\nSpróbuj ponownie później.")
            return None

        if response.status_code >= 500:
            st.error("⚠️ **OpenRouter albo dostawca modelu jest chwilowo niedostępny.**")
            with st.expander("Szczegóły błędu (dla deweloperów)"):
                st.code(response.text)
            return None

        response.raise_for_status()
        response_data = response.json()

        usage = response_data.get("usage", {})

        print_llm_usage(
            provider="openrouter",
            model=model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            cost=usage.get("cost"),
        )

        return response_data["choices"][0]["message"]["content"]

    except requests.Timeout:
        st.error("⚠️ **Zapytanie do OpenRouter przekroczyło limit czasu.**")
        return None

    except requests.RequestException as e:
        st.error("⚠️ **Błąd połączenia z OpenRouter.**")
        with st.expander("Szczegóły błędu (dla deweloperów)"):
            st.code(str(e))
        return None

    except (KeyError, IndexError, TypeError) as e:
        st.error("⚠️ **Nieoczekiwany format odpowiedzi z OpenRouter.**")
        with st.expander("Szczegóły odpowiedzi API"):
            st.code(str(e))
        return None

    except Exception as e:
        st.error(f"⚠️ **Wystąpił nieoczekiwany błąd OpenRouter:**\n\n{type(e).__name__}")
        with st.expander("Szczegóły błędu (dla deweloperów)"):
            st.code(str(e))
        return None


def handle_llm(prompt: str) -> Optional[str]:
    """
    Obsługuje komunikację z wybranym backendem LLM.

    Domyślnie działa jak wcześniej, czyli używa Gemini.
    OpenRouter jest używany tylko po ustawieniu provider = "openrouter" w sekcji [llm].
    """
    config = get_llm_config()
    provider = config["provider"]

    if provider == "gemini":
        return call_gemini(
            prompt=prompt,
            model=config["gemini_model"],
            api_key=config["gemini_api_key"],
        )

    if provider == "openrouter":
        return call_openrouter(
            prompt=prompt,
            model=config["openrouter_model"],
            api_key=config["openrouter_api_key"],
        )

    st.error(
        "⚠️ **Nieznany provider LLM.**\n\n"
        "Dozwolone wartości w `.streamlit/secrets.toml`: `gemini` albo `openrouter`."
    )
    return None
