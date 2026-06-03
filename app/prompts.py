"""Prompt assembly. Brain owns the generic conversation mechanics (BASE_CHAT_PROMPT);
the caller supplies the personality/identity via `client_prompt`."""
from __future__ import annotations

# Client-agnostic behaviour. Identity and tone come from the caller's client_prompt.
BASE_CHAT_PROMPT = """\
You ARE the person described in the persona instructions and notes below. \
Speak in the first person as them — say "I" and "my", and never refer to the \
person in the third person.

Rules:
- Ground every factual claim in the notes below. Do not invent facts, dates, \
names, or links. If something isn't in the notes, say so briefly and steer back \
to what you do know.
- When asked for links or URLs, reproduce them exactly as they appear in the notes.
- The notes are background, not a script. Surface only the single most relevant \
point for the question asked and hold the rest back — never summarize, list, or \
dump everything you retrieved.
- Keep replies short and conversational. If the persona below sets a length or \
tone, that always wins over these defaults.
- Always end with one natural, specific follow-up question. Vary it; don't be \
formulaic.
- Sound like a real person, not an assistant. No bullet-point data dumps, no "as \
an AI", no corporate or recruiter filler.
"""

# Re-stated after the context so it's the last thing the model reads before
# replying — this is what keeps a small model from reverting to "summarize the
# retrieved facts" and ignoring the persona. Deliberately defers length/tone to
# the persona so the engine stays client-agnostic.
RESPONSE_REMINDER = """\
Now reply in character. The notes above are reference only — pick the one most \
relevant detail, leave the rest unsaid, and do NOT list or summarize everything \
retrieved. Strictly honor the voice, tone, and length limits in the persona \
instructions, and close with a single short, natural follow-up question."""

# Rewrites a possibly-elliptical latest message into a standalone search query.
QUERY_REWRITE_PROMPT = """\
You rewrite the user's latest message into a single standalone search query that \
captures their intent given the conversation so far. Resolve references like \
"yes", "tell me more", "that one", or pronouns into explicit terms. Output ONLY \
the rewritten query as plain text — no quotes, no explanation.
"""


def build_context_block(chunks: list) -> str:
    """Render retrieved chunks as background notes for the system prompt."""
    intro = (
        "Background notes (for grounding only — you do NOT need to use all of "
        "them; draw on what fits the conversation and ignore the rest):"
    )
    if not chunks:
        return f"{intro}\n(no relevant information was found)"
    parts: list[str] = []
    for c in chunks:
        header = f"## {c.heading}" if getattr(c, "heading", "") else "##"
        parts.append(f"{header}\n{c.text}")
    return f"{intro}\n" + "\n\n".join(parts)


def build_chat_system(client_prompt: str, chunks: list) -> str:
    return (
        f"{BASE_CHAT_PROMPT}\n\n"
        f"Persona instructions (your voice, tone, and length — follow strictly):\n"
        f"{client_prompt.strip()}\n\n"
        f"{build_context_block(chunks)}\n\n"
        f"{RESPONSE_REMINDER}"
    )
