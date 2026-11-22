"""Element-name patterns: the accelerators write Lua patterns, other backends match them as regex."""

from __future__ import annotations

import re

#: Lua ``%x`` character classes and their regex equivalents (ASCII).
_CLASSES = {
    "a": "A-Za-z",
    "d": "0-9",
    "l": "a-z",
    "s": r"\s",
    "u": "A-Z",
    "w": "A-Za-z0-9",
    "x": "0-9A-Fa-f",
    "p": re.escape("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"),
}


def _class(letter: str) -> str:
    if letter.lower() not in _CLASSES:
        raise ValueError(f"Unsupported Lua class %{letter}")
    body = _CLASSES[letter.lower()]
    return f"[^{body}]" if letter.isupper() else f"[{body}]"


def lua_to_regex(pattern: str) -> re.Pattern[str]:
    """Compile a Lua pattern as a regex with the same matches (use ``.search``, as ``string.match``).

    Supports anchors, ``.``, ``%`` classes and escapes, sets ``[...]``/``[^...]``, the
    quantifiers ``* + ? -`` and captures. Raises ``ValueError`` on ``%b``, ``%f`` and
    back-references.
    """
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        char = pattern[i]
        if char == "%":
            nxt = pattern[i + 1]
            if nxt.isalpha():
                out.append(_class(nxt))
            elif nxt.isdigit():
                raise ValueError(f"Back-references are not supported: {pattern!r}")
            else:
                out.append(re.escape(nxt))
            i += 2
        elif char == "[":
            j = i + 1
            body = ["["]
            if j < n and pattern[j] == "^":
                body.append("^")
                j += 1
            first = True
            while j < n and (pattern[j] != "]" or first):
                first = False
                if pattern[j] == "%":
                    letter = pattern[j + 1]
                    body.append(
                        _class(letter)[1:-1]
                        if letter.isalpha() and letter.islower()
                        else re.escape(letter)
                        if not letter.isalpha()
                        else _unsupported(pattern)
                    )
                    j += 2
                    continue
                body.append("\\" + pattern[j] if pattern[j] in "\\[]" else pattern[j])
                j += 1
            if j >= n:
                raise ValueError(f"Unterminated set in {pattern!r}")
            out.append("".join(body) + "]")
            i = j + 1
        elif char == "-":
            out.append("*?" if out else re.escape(char))
            i += 1
        elif char in "*+?().":
            out.append(char)
            i += 1
        elif char == "^" and i == 0:
            out.append("^")
            i += 1
        elif char == "$" and i == n - 1:
            out.append(r"\Z")
            i += 1
        else:
            out.append(re.escape(char))
            i += 1
    return re.compile("".join(out))


def _unsupported(pattern: str) -> str:
    raise ValueError(f"Negated class inside a set is not supported: {pattern!r}")
