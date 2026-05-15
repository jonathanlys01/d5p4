"""Extract number from text string"""

import re

from sympy import simplify
from sympy.parsing.latex import parse_latex


class MathParser:
    """Robust math / LaTeX answer extractor and evaluator.

    Pre-compiles all regex patterns once and exposes a clean API that
    covers the main extraction strategies:
      • ``extract_universal_numeric`` — best general-purpose pipeline
      • ``parse_latex`` — symbolic parsing via SymPy (optionally evaluated)
    """

    # Pre-compiled regexes ───────────────────────────────────────────────────
    _BOXED_RE = re.compile(r"\\boxed{((?:[^{}]|{[^{}]*})*)}")
    _GSM8K_FINAL_RE = re.compile(r"####\s*([\-\d,\.]+)")
    _INLINE_MATH_RE = re.compile(r"\$(.*?)\$|\\\((.*?)\\\)|\\\[(.*?)\\\]", re.DOTALL)
    _ANSWER_LINE_RE = re.compile(r"(?:^|\b)(?:final answer|answer)\s*[:=]\s*(.+)$", re.IGNORECASE)
    _SCIENTIFIC_NUMBER_RE = re.compile(r"-?(?:\d+(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?%?")
    _FRACTION_RE = re.compile(r"-?\d+/\d+")
    _MIXED_NUMBER_RE = re.compile(r"(\d+)\s+(\d+/\d+)")
    _LATEX_FORMAT_RE = re.compile(r"\\(mathrm|mathbf|mathit|mathsf|mathtt|mathcal|bm|bold|textbf|textit){(.*?)}")
    _LATEX_TEXT_RE = re.compile(r"\\text{.*?}")
    _ARITHMETIC_RE = re.compile(r"-?\d+(?:\s*[\+\-\*\/]\s*-?\d+)+")

    # ── normalisation ──────────────────────────────────────────────────────

    @staticmethod
    def normalize(answer: str) -> str:
        """Strip LaTeX delimiters, formatting commands, and trailing punctuation."""
        # Handle degree symbol
        answer = answer.replace(r"^\circ", "").replace(r"\degree", "")

        # Shelter escaped chars with sentinels so the bare-$ strip doesn't eat them
        answer = answer.replace("\\%", "\x00PCT").replace("\\$", "\x00DLR")
        answer = answer.replace("$", "").replace("\\[", "").replace("\\]", "")
        answer = answer.replace("\\(", "").replace("\\)", "")

        # Remove common LaTeX formatting commands while keeping their content
        answer = MathParser._LATEX_FORMAT_RE.sub(r"\2", answer)

        # Remove LaTeX text commands completely
        answer = MathParser._LATEX_TEXT_RE.sub("", answer)

        # Restore escaped chars
        answer = answer.replace("\x00PCT", "%").replace("\x00DLR", "$")
        answer = answer.replace(" ", "").rstrip(".")
        return answer.strip()

    # ── text → str extraction ──────────────────────────────────────────────

    def _extract_numeric_candidates(self, text: str) -> list[str]:
        """Return candidates ordered from most to least likely."""
        candidates: list[str] = []

        gsm8k_final = self._GSM8K_FINAL_RE.findall(text)
        if gsm8k_final:
            candidates.extend(reversed(gsm8k_final))

        boxed = self._BOXED_RE.findall(text)
        if boxed:
            candidates.extend(reversed(boxed))

        for line in reversed(text.splitlines()):
            match = self._ANSWER_LINE_RE.search(line)
            if match:
                candidates.append(match.group(1).strip())

        for math_match in self._INLINE_MATH_RE.findall(text):
            expr = next((m for m in math_match if m), "")
            if expr:
                candidates.append(expr.strip())

        # Prioritize more complex forms (mixed numbers, arithmetic, fractions)
        mixed = self._MIXED_NUMBER_RE.findall(text)
        if mixed:
            candidates.extend(reversed([f"({w}+{f})" for w, f in mixed]))

        arithmetic = self._ARITHMETIC_RE.findall(text)
        if arithmetic:
            candidates.extend(reversed(arithmetic))

        fractions = self._FRACTION_RE.findall(text)
        if fractions:
            candidates.extend(reversed(fractions))

        candidates.append(text)

        sci_numbers = self._SCIENTIFIC_NUMBER_RE.findall(text)
        if sci_numbers:
            # Filter matches to avoid picking up single digits from fractions already found
            last_num = sci_numbers[-1].replace(",", "")
            if last_num not in candidates:
                candidates.append(last_num)

        return candidates

    def _canonicalize_numeric(self, value: str) -> str | None:  # noqa: PLR0911
        """Convert input into a canonical numeric string, if possible."""
        if not value:
            return None

        candidate = self.normalize(value).replace(",", "")
        if not candidate:
            return None

        percent = False
        if candidate.endswith("%"):
            percent = True
            candidate = candidate[:-1]
            if not candidate:
                return None

        expr = None
        if "\\" in candidate:
            expr = self.parse_latex(candidate)
        if expr is None:
            try:
                expr = simplify(candidate)
            except Exception:
                return None

        if expr is None or getattr(expr, "free_symbols", None):
            return None
        if not getattr(expr, "is_number", False):
            return None

        if percent:
            expr = simplify(expr / 100)

        expr = simplify(expr)
        if getattr(expr, "is_Integer", False):
            return str(int(expr))
        if getattr(expr, "is_Rational", False):
            p = getattr(expr, "p", None)
            q = getattr(expr, "q", None)
            if q == 1 and p is not None:
                return str(int(p))
            return f"{p}/{q}"

        try:
            as_float = float(expr.evalf())
        except Exception:
            return None

        if as_float.is_integer():
            return str(int(as_float))
        return format(as_float, ".12g")

    def extract_universal_numeric(self, text: str) -> str:
        """Best-effort universal extractor for math/LaTeX numeric answers."""
        if not text:
            return "NULL"

        for candidate in self._extract_numeric_candidates(text):
            parsed = self._canonicalize_numeric(candidate)
            if parsed is not None:
                return parsed
        return "NULL"

    # ── symbolic parsing ───────────────────────────────────────────────────

    def parse_latex(self, latex_str: str, *, evaluate: bool = False):
        """Parse a LaTeX string into a SymPy expression.

        Parameters
        ----------
        latex_str:
            Raw LaTeX math string (e.g. ``\\frac{1}{2} + \\frac{1}{4}``).
        evaluate:
            If *True*, call ``.evalf()`` on the result to obtain a float.

        Returns
        -------
        sympy.Expr | None
            The parsed (and optionally evaluated) expression, or *None*
            if parsing fails.
        """
        if not latex_str:
            return None

        try:
            expr = parse_latex(latex_str, backend="lark")
            if expr is None:
                return None
            if evaluate and hasattr(expr, "evalf"):
                return expr.evalf()
            return expr
        except Exception:
            return None


_DEFAULT_MATH_PARSER = MathParser()


def universal_math_postprocess(text: str) -> str:
    """Extract a canonical numeric answer from mixed text/LaTeX output."""
    return _DEFAULT_MATH_PARSER.extract_universal_numeric(text)


if __name__ == "__main__":
    mp = MathParser()

    print(mp.extract_universal_numeric(r"The answer is \boxed{42}."))
    print(mp.extract_universal_numeric(r"Therefore, the radius is $3.14$ meters."))
    print(mp.extract_universal_numeric(r"I calculate the fraction to be \boxed{\frac{1}{2}}"))
    print(mp.extract_universal_numeric("The cost of the 5 items is 1,234.50 dollars."))

    expr = mp.parse_latex(r"\frac{1}{2} + \frac{1}{4}")
    if expr is not None:
        print(f"Symbolic: {expr}, Simplified: {simplify(expr)}, Float: {expr.evalf()}")

    print(mp.parse_latex(r"\sqrt{16} * 2", evaluate=True))
    print(mp.parse_latex(r"\pi \approx"))  # None
