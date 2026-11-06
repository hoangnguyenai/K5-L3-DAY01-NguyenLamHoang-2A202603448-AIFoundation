"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to src/optional_llm_api/solution/solution.py when done.
    4. Run from the repo root: pytest tests/test_solution.py -v

Models (11/2026): gpt-4o / gpt-4o-mini are removed from the OpenAI API on 2027-01-14
(https://developers.openai.com/api/docs/deprecations), so this lab uses gpt-6.1-terra / gpt-6-luna.
The result keys and table headers below still say "gpt4o" / "GPT-4o" / "Mini": they are a test
contract kept for compatibility. Read "gpt4o" as "the large model" and "mini" as "the small model".
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1K OUTPUT tokens (USD) — OpenAI pricing page, 11/2026. Update if pricing changes.
# ---------------------------------------------------------------------------
COST_PER_1K_OUTPUT_TOKENS = {
    "gpt-6.1-terra": 0.005,   # $5.00 / 1M output tokens
    "gpt-6-luna": 0.0005,     # $0.50 / 1M output tokens
}

OPENAI_MODEL = "gpt-6.1-terra"
OPENAI_MINI_MODEL = "gpt-6-luna"

# Select before starting Python; the template defaults remain for grading.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
if LLM_PROVIDER not in {"openai", "groq"}:
    raise ValueError("LLM_PROVIDER must be openai or groq")
if LLM_PROVIDER == "groq":
    OPENAI_MODEL = "openai/gpt-oss-120b"
    OPENAI_MINI_MODEL = "openai/gpt-oss-20b"


def _client_options() -> dict:
    if LLM_PROVIDER == "groq":
        return {"api_key": os.getenv("GROQ_API_KEY"),
                "base_url": "https://api.groq.com/openai/v1"}
    return {"api_key": os.getenv("OPENAI_API_KEY")}


def _generation_options(max_tokens: int) -> dict:
    return {
        "reasoning_effort": "low" if LLM_PROVIDER == "groq" else "none",
        # Groq GPT-OSS also needs room for reasoning tokens.
        "max_completion_tokens": max_tokens,
    }


# ---------------------------------------------------------------------------
# Task 1 — Call the large model (OPENAI_MODEL)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Call the OpenAI Chat Completions API and return the response text + latency.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: OPENAI_MODEL).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of (response_text: str, latency_seconds: float).

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            reasoning_effort="none",          # GPT-6 models only accept temperature/top_p with effort "none"
            temperature=temperature, top_p=top_p,
            max_completion_tokens=max_tokens,  # GPT-6 models reject max_tokens
        )
    """
    from openai import OpenAI

    client = OpenAI(**_client_options())
    try:
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            **_generation_options(max_tokens),
        )
        latency = time.perf_counter() - start
    finally:
        client.close()
    return response.choices[0].message.content or "", latency


# ---------------------------------------------------------------------------
# Task 2 — Call the small model (OPENAI_MINI_MODEL)
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Call the OpenAI Chat Completions API using OPENAI_MINI_MODEL and return the
    response text + latency.

    Args:
        prompt:      The user message to send.
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of (response_text: str, latency_seconds: float).

    Hint:
        Reuse call_openai() by passing model=OPENAI_MINI_MODEL.
    """
    return call_openai(
        prompt, model=OPENAI_MINI_MODEL, temperature=temperature,
        top_p=top_p, max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Task 3 — Compare the large vs the small model
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call both OPENAI_MODEL and OPENAI_MINI_MODEL with the same prompt and return a
    comparison dictionary.

    Args:
        prompt: The user message to send to both models.

    Returns:
        A dict with keys:
            - "gpt4o_response":      str
            - "mini_response":       str
            - "gpt4o_latency":       float
            - "mini_latency":        float
            - "gpt4o_cost_estimate": float  (estimated USD for the response)

    Hint:
        Cost estimate = (len(response.split()) / 0.75) / 1000 * COST_PER_1K_OUTPUT_TOKENS[OPENAI_MODEL]
        (0.75 words ≈ 1 token is a rough approximation)
    """
    if LLM_PROVIDER == "groq":
        large_response, large_latency = call_openai(prompt, max_tokens=1024)
        mini_response, mini_latency = call_openai_mini(prompt, max_tokens=1024)
    else:
        large_response, large_latency = call_openai(prompt)
        mini_response, mini_latency = call_openai_mini(prompt)
    # Template approximation: output only; excludes input and reasoning tokens.
    output_tokens_estimate = len(large_response.split()) / 0.75
    return {
        "gpt4o_response": large_response,
        "mini_response": mini_response,
        "gpt4o_latency": large_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": (
            output_tokens_estimate / 1000
            * COST_PER_1K_OUTPUT_TOKENS.get(OPENAI_MODEL, 0.0)
        ),
    }


# ---------------------------------------------------------------------------
# Task 4 — Streaming chatbot with conversation history
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal.

    Behaviour:
        - Streams tokens from OpenAI as they arrive (print each chunk).
        - Maintains the last 3 conversation turns in history.
        - Typing 'quit' or 'exit' ends the loop.

    Hints:
        - Keep a list `history` of {"role": ..., "content": ...} dicts.
        - Use stream=True in client.chat.completions.create() and iterate:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
        - After each turn, append the assistant reply to history.
        - Trim history to the last 3 turns (= 6 messages): history = history[-6:]
    """
    from openai import OpenAI

    history: list[dict[str, str]] = []
    # Create the client only after a message, so quitting needs no API key.
    client = None
    try:
        while True:
            try:
                text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text.lower() in {"quit", "exit"}:
                break
            if not text:
                continue
            if client is None:
                client = OpenAI(**_client_options())
            messages = history + [{"role": "user", "content": text}]
            parts = []
            print("Assistant: ", end="", flush=True)
            stream = client.chat.completions.create(
                model=OPENAI_MODEL, messages=messages, stream=True,
                **_generation_options(1024 if LLM_PROVIDER == "groq" else 256),
            )
            try:
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content or ""
                    parts.append(delta)
                    print(delta, end="", flush=True)
            finally:
                stream.close()
                print()
            # Retain three complete user/assistant pairs for the next request.
            history = (messages + [
                {"role": "assistant", "content": "".join(parts)}
            ])[-6:]
    finally:
        if client is not None:
            client.close()


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    if max_retries < 0 or base_delay < 0:
        raise ValueError("max_retries and base_delay must be non-negative")
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * 2 ** attempt)


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    return [{**compare_models(prompt), "prompt": prompt} for prompt in prompts]


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of compare_models results as a readable text table.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A formatted string table with columns:
        Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency

    Hint:
        Truncate long text to 40 characters for readability.
    """
    def shorten(value: str) -> str:
        text = " ".join(value.split())
        return text if len(text) <= 40 else text[:37] + "..."

    headers = ["Prompt", "GPT-4o Response", "Mini Response",
               "GPT-4o Latency", "Mini Latency"]
    rows = [headers]
    for result in results:
        rows.append([
            shorten(result.get("prompt", "")),
            shorten(result["gpt4o_response"]),
            shorten(result["mini_response"]),
            f"{result['gpt4o_latency']:.3f}s",
            f"{result['mini_latency']:.3f}s",
        ])
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    lines = [" | ".join(cell.ljust(width) for cell, width in zip(row, widths))
             for row in rows]
    lines.insert(1, "-+-".join("-" * width for width in widths))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Provider: {LLM_PROVIDER}; large: {OPENAI_MODEL}; small: {OPENAI_MINI_MODEL}")
    if LLM_PROVIDER == "groq":
        print("Groq cost estimate is unavailable; displayed 0 is not a billing estimate.")
    test_prompt = "Explain the difference between temperature and top_p in one sentence."
    print("=== Comparing models ===")
    result = compare_models(test_prompt)
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Starting chatbot (type 'quit' to exit) ===")
    streaming_chatbot()
