"""Answer generation over retrieved abstracts (Mistral), with citations by document id.

Ported from the original Streamlit app. This step is optional (MISTRAL_API_KEY) and is not
part of the measured benchmark: the evaluation in this repo covers retrieval quality, which
bounds what any generator can cite.
"""

from __future__ import annotations

import os

SYSTEM = ("You are a biomedical research assistant. Answer only from the numbered abstracts. "
          "Cite every claim as [doc:<id>]. If the abstracts do not answer the question, say so.")


def build_prompt(question: str, hits: list[dict], max_chars: int = 6000) -> str:
    parts, used = [], 0
    for h in hits:
        block = f"[doc:{h['doc_id']}] {h['title']}\n{h.get('text', '')}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "Abstracts:\n\n" + "\n".join(parts) + f"\nQuestion: {question}"


def answer(question: str, hits: list[dict], client=None, model: str = "mistral-large-latest") -> str:
    if not hits:
        return "No relevant abstracts were retrieved, so there is no evidence to answer from."
    if client is None:
        from mistralai import Mistral

        key = os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError("set MISTRAL_API_KEY to generate answers")
        client = Mistral(api_key=key)
    resp = client.chat.complete(model=model, temperature=0.2, messages=[
        {"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(question, hits)}])
    return resp.choices[0].message.content
