"""
NEC file parser.

Two-pass parser: resolves SY symbol variables first, then parses all cards
with type-safe parameter handling. Based on the NEC-domain ontology specs
from antenna-agent-model.
"""

import ast
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# NEC card parameter specs: (name, python_type, required)
CARD_SPECS: dict[str, list[tuple[str, type, bool]]] = {
    "GW": [  # Wire geometry
        ("tag", int, True),
        ("segments", int, True),
        ("x1", float, True), ("y1", float, True), ("z1", float, True),
        ("x2", float, True), ("y2", float, True), ("z2", float, True),
        ("radius", float, False),
    ],
    "GA": [  # Wire arc
        ("tag", int, True), ("segments", int, True),
        ("arcRadius", float, True), ("startAngle", float, True), ("endAngle", float, True),
        ("wireRadius", float, False),
    ],
    "GH": [  # Helix / spiral
        ("tag", int, True), ("segments", int, True),
        ("spacing", float, True), ("hl", float, True),
        ("r1", float, True), ("r2", float, True),
        ("r3", float, True), ("r4", float, True),
        ("wireRadius", float, False),
    ],
    "GM": [  # Coordinate transformation / move
        ("i1", int, True), ("i2", int, True),
        ("roX", float, True), ("roY", float, True), ("roZ", float, True),
        ("trX", float, True), ("trY", float, True), ("trZ", float, True),
        ("its", float, False),
    ],
    "GX": [  # Reflection
        ("i1", int, True), ("i2", int, True),
    ],
    "GR": [  # Cylindrical structure generation
        ("i1", int, True), ("i2", int, True),
    ],
    "GS": [  # Scale structure dimensions
        ("i1", int, True), ("i2", int, True), ("factor", float, True),
    ],
    "GC": [  # Wire geometry continuation (tapered wire)
        ("rdel", float, True), ("rad1", float, True), ("rad2", float, True),
    ],
    "SP": [  # Surface patch
        ("patchType", int, True),
        ("x1", float, True), ("y1", float, True), ("z1", float, True),
        ("elev", float, True), ("azim", float, True), ("area", float, False),
    ],
    "SM": [  # Multiple surface patches
        ("nx", int, True), ("ny", int, True),
        ("x1", float, True), ("y1", float, True), ("z1", float, True),
        ("x2", float, True), ("y2", float, True), ("z2", float, True),
    ],
    "SC": [  # Surface patch continuation
        ("x3", float, True), ("y3", float, True), ("z3", float, True),
        ("x4", float, False), ("y4", float, False), ("z4", float, False),
    ],
    "GE": [  # Geometry end
        ("ground", int, False),
    ],
    "FR": [  # Frequency
        ("freqType", int, True), ("nFreq", int, True),
        ("i1", int, True), ("i2", int, True),
        ("freq", float, True), ("freqStep", float, False),
    ],
    "EX": [  # Excitation
        ("exType", int, True), ("tag", int, True), ("segment", int, True),
        ("i3", int, True), ("vReal", float, True), ("vImag", float, False),
        ("f3", float, False), ("f4", float, False),
    ],
    "LD": [  # Loading
        ("ldType", int, True), ("tag", int, True),
        ("segStart", int, True), ("segEnd", int, True),
        ("zlr", float, True), ("zli", float, False), ("zlc", float, False),
    ],
    "GN": [  # Ground
        ("groundType", int, True), ("nRadials", int, False),
        ("epsr", float, False), ("sigma", float, False),
        ("epsr2", float, False), ("sigma2", float, False),
    ],
    "RP": [  # Radiation pattern
        ("mode", int, True), ("nTheta", int, True), ("nPhi", int, True),
        ("output", int, True),
        ("thetaStart", float, True), ("thetaStep", float, True),
        ("phiStart", float, False), ("phiStep", float, False),
        ("rNorm", float, False),
    ],
    "TL": [  # Transmission line
        ("tag1", int, True), ("seg1", int, True),
        ("tag2", int, True), ("seg2", int, True),
        ("z0", float, True), ("length", float, False),
        ("shuntR1", float, False), ("shuntI1", float, False),
        ("shuntR2", float, False), ("shuntI2", float, False),
    ],
    "NT": [  # Networks (2-port)
        ("tag1", int, True), ("seg1", int, True),
        ("tag2", int, True), ("seg2", int, True),
        ("r11", float, True), ("i11", float, True),
        ("r12", float, True), ("i12", float, True),
        ("r22", float, True), ("i22", float, True),
    ],
    "NE": [  # Near E-field
        ("nearType", int, True), ("nX", int, True), ("nY", int, True), ("nZ", int, True),
        ("x", float, True), ("y", float, True), ("z", float, True),
        ("dx", float, False), ("dy", float, False), ("dz", float, False),
    ],
    "NH": [  # Near H-field
        ("nearType", int, True), ("nX", int, True), ("nY", int, True), ("nZ", int, True),
        ("x", float, True), ("y", float, True), ("z", float, True),
        ("dx", float, False), ("dy", float, False), ("dz", float, False),
    ],
    "XQ": [  # Execute
        ("i1", int, False),
    ],
    "EN": [  # End
    ],
    "EK": [  # Extended thin-wire kernel
        ("i1", int, False),
    ],
    "CP": [  # Couple calculation output
        ("tag1", int, True), ("seg1", int, True),
        ("tag2", int, True), ("seg2", int, True),
    ],
    "PQ": [  # Print control for charge
        ("i1", int, False), ("i2", int, False), ("i3", int, False), ("i4", int, False),
    ],
    "PT": [  # Print control for current
        ("i1", int, True), ("tag", int, False), ("segStart", int, False), ("segEnd", int, False),
    ],
    "WG": [  # Write NGF file
    ],
    "KH": [  # Interaction approximation
        ("approx", float, True),
    ],
}

_SY_RE = re.compile(r"^\s*SY\b", re.IGNORECASE)

# AWG wire gauge to diameter in METERS (radius = diameter/2)
AWG_LOOKUP: dict[str, float] = {
    "#4": 0.005189, "#6": 0.004115, "#8": 0.003264, "#10": 0.002588,
    "#12": 0.002053, "#14": 0.001628, "#16": 0.001291, "#18": 0.001024,
    "#20": 0.000812, "#22": 0.000643, "#24": 0.000511, "#26": 0.000404,
    "#28": 0.000320, "#30": 0.000254, "#32": 0.000203, "#34": 0.000160,
    "#36": 0.000127, "#38": 0.000102, "#40": 0.000079,
}


@dataclass
class NECCard:
    """A single parsed NEC card."""
    line_number: int
    raw: str
    card_type: str
    params: list[Any] = field(default_factory=list)
    labeled_params: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    errors: list[str] = field(default_factory=list)


@dataclass
class ParseResult:
    """Complete result of parsing a NEC file."""
    source: str
    cards: list[NECCard] = field(default_factory=list)
    symbol_table: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def geometry_cards(self) -> list[NECCard]:
        return [c for c in self.cards if c.card_type in ("GW", "GA", "GH", "GM", "GX", "GR", "GS", "GC", "SP", "SM", "SC")]

    @property
    def wire_cards(self) -> list[NECCard]:
        return [c for c in self.cards if c.card_type == "GW"]

    @property
    def comment_text(self) -> str:
        """Concatenated CM/CE comment text."""
        parts = []
        for c in self.cards:
            if c.card_type in ("CM", "CE") and c.text:
                parts.append(c.text)
        return " ".join(parts)


def _preprocess_awg(expr: str) -> str:
    """Replace AWG gauge notation like '#12/in' or bare '#12' with metric diameter."""
    # Replace '#XX/in' patterns first
    for gauge, diameter_m in AWG_LOOKUP.items():
        expr = expr.replace(f"{gauge}/in", str(diameter_m))
    # Then replace bare '#XX' patterns (won't double-match since /in forms are gone)
    for gauge, diameter_m in AWG_LOOKUP.items():
        if gauge in expr:
            expr = expr.replace(gauge, str(diameter_m))
    return expr


def _expand_scientific(expr: str) -> str:
    """Expand scientific notation so ast.parse doesn't choke on it."""
    m = re.match(r"^(-?\d+(?:\.\d*)?)[eE]([-+]?\d+)$", expr.strip())
    if m:
        try:
            val = float(expr)
            return str(int(val)) if val == int(val) else f"{val:.12g}"
        except (ValueError, OverflowError):
            pass
    return expr


def _safe_eval(expr: str, sym: dict[str, float]) -> int | float | str:
    """Safely evaluate a numeric expression with symbol substitution."""
    cleaned = _preprocess_awg(expr.strip())
    cleaned = _expand_scientific(cleaned)

    # Handle trailing percent (4NEC2 segment-percent notation)
    if cleaned.endswith("%"):
        try:
            return int(cleaned[:-1])
        except ValueError:
            pass

    # Quick path: plain number
    try:
        if "." in cleaned or "e" in cleaned.lower():
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        pass

    try:
        node = ast.parse(cleaned, mode="eval").body
        return _eval_node(node, sym)
    except Exception:
        return cleaned  # unresolvable — return as string


# Math functions available in SY expressions (NEC uses degrees for trig)
_MATH_FUNCS: dict[str, callable] = {
    "cos": lambda x: math.cos(math.radians(x)),
    "sin": lambda x: math.sin(math.radians(x)),
    "tan": lambda x: math.tan(math.radians(x)),
    "acos": lambda x: math.degrees(math.acos(x)),
    "asin": lambda x: math.degrees(math.asin(x)),
    "atan": lambda x: math.degrees(math.atan(x)),
    "atan2": lambda y, x: math.degrees(math.atan2(y, x)),
    "sqrt": math.sqrt,
    "abs": abs,
    "log": math.log10,
    "ln": math.log,
    "exp": math.exp,
    "int": lambda x: int(x),
    "pi": None,  # handled as constant
}


def _eval_node(n: ast.AST, sym: dict[str, float]) -> int | float:
    if isinstance(n, ast.BinOp):
        left = _eval_node(n.left, sym)
        right = _eval_node(n.right, sym)
        ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
               ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
               ast.Pow: lambda a, b: a ** b,
               ast.Mod: lambda a, b: a % b, ast.FloorDiv: lambda a, b: a // b}
        op_fn = ops.get(type(n.op))
        if op_fn is None:
            raise ValueError(f"Unsupported op {n.op}")
        return op_fn(left, right)
    if isinstance(n, ast.UnaryOp):
        val = _eval_node(n.operand, sym)
        if isinstance(n.op, ast.UAdd):
            return +val
        if isinstance(n.op, ast.USub):
            return -val
        raise ValueError(f"Unsupported unary {n.op}")
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return n.value
    if isinstance(n, ast.Name):
        key = n.id.lower()  # case-insensitive variable lookup
        if key in sym:
            return sym[key]
        if key == "pi":
            return math.pi
        raise ValueError(f"Unknown variable '{n.id}'")
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
        fname = n.func.id.lower()
        fn = _MATH_FUNCS.get(fname)
        if fn is not None:
            args = [_eval_node(a, sym) for a in n.args]
            return fn(*args)
        # Check if it's a variable call (e.g., max(a,b))
        raise ValueError(f"Unknown function '{fname}'")
    raise ValueError(f"Cannot evaluate node {ast.dump(n)}")


def _split_params(raw_tail: str) -> list[str]:
    """Split the parameter portion of a NEC card line, handling commas, spaces, and inline comments."""
    # Strip inline comments (everything after an unquoted apostrophe)
    comment_idx = raw_tail.find("'")
    if comment_idx >= 0:
        raw_tail = raw_tail[:comment_idx]
    # Also strip C-style comments
    comment_idx = raw_tail.find("!")
    if comment_idx >= 0:
        raw_tail = raw_tail[:comment_idx]
    # Split on commas and/or whitespace
    return [p.strip() for p in re.split(r"[,\s]\s*", raw_tail.strip()) if p.strip()]


def parse_file(path: str | Path) -> ParseResult:
    """Parse a NEC file from disk."""
    path = Path(path)
    text = path.read_text(errors="replace")
    return parse_text(text, source=str(path))


def parse_text(text: str, source: str = "<string>") -> ParseResult:
    """Parse NEC content from a string."""
    lines = text.splitlines()
    return _parse_lines(lines, source)


def _parse_lines(lines: list[str], source: str) -> ParseResult:
    result = ParseResult(source=source)

    # Pass 1: build symbol table from SY cards
    for line in lines:
        stripped = line.strip()
        if _SY_RE.match(stripped):
            _resolve_sy(stripped, result.symbol_table, result.warnings)

    # Pass 2: parse all cards
    for idx, line in enumerate(lines, start=1):
        raw = line.rstrip("\r\n")
        stripped = raw.strip()

        if not stripped:
            continue

        # Apostrophe-only comment lines
        if stripped.startswith("'"):
            result.cards.append(NECCard(
                line_number=idx, raw=raw, card_type="CM",
                text=stripped[1:].strip(),
            ))
            continue

        # SY cards — keep as-is
        if _SY_RE.match(stripped):
            text = stripped.split(None, 1)[1].strip() if " " in stripped else ""
            result.cards.append(NECCard(
                line_number=idx, raw=raw, card_type="SY", text=text,
            ))
            continue

        # Split card type from parameters.
        # Strip inline comments first, then extract card type and params.
        clean = _strip_inline_comment(stripped)
        if not clean:
            continue
        if "\t" in clean:
            parts = clean.split("\t")
            first_part = parts[0].strip()
            # Handle mixed format: spaces within first tab field
            # e.g. "GW 1 9 0 ... \t' comment" where GW and params are
            # space-separated but tab separates data from comment
            if " " in first_part:
                sub = first_part.split(None, 1)
                ctype = sub[0].upper()
                param_tokens = _split_params(sub[1]) if len(sub) > 1 else []
            else:
                ctype = first_part.upper()
                param_tokens = []
            for p in parts[1:]:
                param_tokens.extend(_split_params(p))
        else:
            parts = clean.split(None, 1)
            ctype = parts[0].upper()
            param_tokens = _split_params(parts[1]) if len(parts) > 1 else []

        # Comment cards — preserve original text (before comment stripping)
        if ctype in ("CM", "CE"):
            # Extract text after card type from the original stripped line
            cm_match = re.match(r'(?:CM|CE)\s*(.*)', stripped, re.IGNORECASE)
            text = cm_match.group(1).strip() if cm_match else ""
            result.cards.append(NECCard(
                line_number=idx, raw=raw, card_type=ctype, text=text,
            ))
            continue

        # Known card types: parse with spec
        if ctype in CARD_SPECS:
            card = _parse_card(ctype, param_tokens, result.symbol_table, idx, raw)
            result.cards.append(card)
            continue

        # Unknown card type — preserve raw params
        result.cards.append(NECCard(
            line_number=idx, raw=raw, card_type=ctype, params=param_tokens,
        ))

    return result


def _strip_inline_comment(text: str) -> str:
    """Strip inline comments (apostrophe or exclamation) from a line."""
    for marker in ("'", "!"):
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx]
    return text.rstrip()


def _resolve_sy(line: str, sym: dict[str, float], warnings: list[str]) -> None:
    """Resolve one SY line, potentially with multiple comma-separated assignments."""
    parts = line.split(None, 1)
    if len(parts) < 2:
        return
    # Strip inline comments before splitting assignments
    body = _strip_inline_comment(parts[1])
    # SY cards can have: SY N=12, D=0.05, H=1.2
    assignments = body.split(",")
    for assignment in assignments:
        assignment = assignment.strip()
        if "=" not in assignment:
            continue
        name, expr = assignment.split("=", 1)
        name = name.strip().lower()  # NEC variable names are case-insensitive
        expr = expr.strip()
        try:
            val = _safe_eval(expr, sym)
            if isinstance(val, (int, float)):
                sym[name] = val
            else:
                warnings.append(f"SY '{name}' resolved to non-numeric: {val}")
        except Exception as e:
            warnings.append(f"Failed to resolve SY '{name}': {e}")


def _parse_card(ctype: str, tokens: list[str], sym: dict[str, float],
                line_num: int, raw: str) -> NECCard:
    """Parse a card with known spec."""
    spec = CARD_SPECS[ctype]
    params: list[Any] = []
    labeled: dict[str, Any] = {}
    errors: list[str] = []

    for i, token in enumerate(tokens):
        if i < len(spec):
            pname, ptype, _required = spec[i]
            try:
                val = _safe_eval(token, sym)
                if ptype is int:
                    val = int(float(val)) if isinstance(val, (int, float)) else val
                elif ptype is float:
                    val = float(val) if isinstance(val, (int, float, str)) else val
                params.append(val)
                labeled[pname] = val
            except Exception as e:
                errors.append(f"param '{pname}': {e}")
                params.append(token)
                labeled[pname] = token
        else:
            params.append(token)

    # Flag missing required params
    min_required = sum(1 for _, _, req in spec if req)
    if len(params) < min_required:
        errors.append(f"Expected at least {min_required} params, got {len(params)}")

    # Fill labeled_params for missing optional params
    for i in range(len(params), len(spec)):
        pname, _, _ = spec[i]
        labeled[pname] = None

    return NECCard(
        line_number=line_num, raw=raw, card_type=ctype,
        params=params, labeled_params=labeled, errors=errors,
    )
