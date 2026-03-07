from fastapi import FastAPI, Body
import ast as _ast
import tempfile, subprocess, shlex, re, json, os, time, hashlib, math, threading

# Allow selecting solver binary via environment (nec2c, nec2++, xnec2c). Default nec2c.
NEC_BIN = os.getenv("NEC_BIN", "nec2c")

# Engineering notation suffixes used by xnec2c / 4nec2
_ENG_SUFFIXES = {
    "T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3, "K": 1e3,
    "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15,
}
# Unit labels that follow a suffix (optional, stripped after multiplier applied)
_ENG_UNITS = {"H", "F", "Hz", "Ohm", "V", "A", "W", "S", "m"}

_ENG_RE = re.compile(
    r'^([\d.Ee+\-]+)\s*([TGMkKmuµnpf])([A-Za-z]*)$'
)

def _parse_eng(token: str) -> str | None:
    """Parse engineering notation like 5.8pF, 1.22uH, 26.3pF -> plain float string.
    Returns None if not engineering notation."""
    m = _ENG_RE.match(token.strip())
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    prefix = m.group(2)
    if prefix not in _ENG_SUFFIXES:
        return None
    return _fmt_float(value * _ENG_SUFFIXES[prefix])

def _fmt_float(v) -> str:
    """Format a numeric value for NEC deck output.
    
    nec2c uses fixed-width column parsing; long decimal strings overflow fields.
    Use 7 significant digits (sub-micron accuracy, fits in nec2c columns).
    """
    if isinstance(v, int):
        return str(v)
    # Use 7 significant digits, strip trailing zeros
    return f"{v:.7g}"

app = FastAPI(title="NEC Solver API", version="1.0")

# ---------------------------------------------------------------------------
# Simulation cache — deterministic NEC output cached by content hash
# ---------------------------------------------------------------------------
CACHE_DIR = os.getenv("CACHE_DIR", "/cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Concurrency limiter — prevent CPU saturation from parallel nec2c processes
_MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_SIMS", "3"))
_sim_semaphore = threading.Semaphore(_MAX_CONCURRENT)

# Cache stats (in-memory, reset on restart)
_cache_hits = 0
_cache_misses = 0
_cache_lock = threading.Lock()


def _cache_key(sanitized_deck: str, endpoint: str, z0: float = 50.0) -> str:
    """Cache key = SHA-256 of sanitized NEC content + endpoint + z0."""
    blob = f"{sanitized_deck}|{endpoint}|{z0}"
    return hashlib.sha256(blob.encode()).hexdigest()


def _cache_get(key: str) -> dict | None:
    """Return cached JSON result or None."""
    global _cache_hits, _cache_misses
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        with open(path) as f:
            data = json.load(f)
        with _cache_lock:
            _cache_hits += 1
        return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        with _cache_lock:
            _cache_misses += 1
        return None


def _cache_put(key: str, result: dict) -> None:
    """Write result JSON to cache (non-fatal on error)."""
    path = os.path.join(CACHE_DIR, f"{key}.json")
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(result, f, separators=(",", ":"))
        os.replace(tmp, path)  # atomic on POSIX
    except OSError:
        pass


# ---------------------------------------------------------------------------
# SY variable resolution (ported from antenna-agent-model nec_parser.py)
# ---------------------------------------------------------------------------
_SY_RE = re.compile(r'^\s*SY\b', re.IGNORECASE)

def _safe_eval(expr: str, vars_dict: dict) -> float | int:
    """Evaluate a numeric expression using AST.

    Supports: +, -, *, /, **, ^ (as power), parens, symbol refs,
    function calls (LOG, LN, SQRT, SIN, COS, TAN, ASIN, ACOS, ATAN, ATAN2, ABS, EXP, LOG10),
    and xnec2c built-in constants (mm, in, ft, pi).
    """
    # xnec2c built-in constants
    _BUILTINS = {
        "mm": 0.001,
        "in": 0.0254,
        "ft": 0.3048,
        "pi": math.pi,
        "PI": math.pi,
    }

    # NEC uses ^ for exponentiation; Python uses ** — rewrite before AST parse.
    prepared = expr.strip().replace("^", "**")
    node = _ast.parse(prepared, mode='eval').body

    # Allowed math functions
    _FUNCS = {
        "LOG": math.log,
        "LN": math.log,
        "LOG10": math.log10,
        "SQRT": math.sqrt,
        "SIN": math.sin,
        "COS": math.cos,
        "TAN": math.tan,
        "ASIN": math.asin,
        "ACOS": math.acos,
        "ATAN": math.atan,
        "ABS": abs,
        "EXP": math.exp,
        "INT": lambda x: int(x),
        "SQR": math.sqrt,
    }
    _FUNCS2 = {
        "ATAN2": math.atan2,
    }

    def _ev(n):
        if isinstance(n, _ast.BinOp):
            l, r = _ev(n.left), _ev(n.right)
            if isinstance(n.op, _ast.Add):    return l + r
            if isinstance(n.op, _ast.Sub):    return l - r
            if isinstance(n.op, _ast.Mult):   return l * r
            if isinstance(n.op, _ast.Div):    return l / r
            if isinstance(n.op, _ast.Pow):    return l ** r
            if isinstance(n.op, _ast.BitXor): return l ** r  # fallback for ^
            raise ValueError(f"unsupported op {n.op}")
        if isinstance(n, _ast.UnaryOp):
            v = _ev(n.operand)
            if isinstance(n.op, _ast.UAdd): return +v
            if isinstance(n.op, _ast.USub): return -v
            raise ValueError(f"unsupported unary {n.op}")
        if isinstance(n, _ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, _ast.Name):
            # Case-insensitive lookup: vars_dict keys are lowercased
            low = n.id.lower()
            if low in vars_dict:
                return vars_dict[low]
            if n.id in vars_dict:
                return vars_dict[n.id]
            if n.id in _BUILTINS:
                return _BUILTINS[n.id]
            upper = n.id.upper()
            if upper in _BUILTINS:
                return _BUILTINS[upper]
            raise ValueError(f"unknown var {n.id!r}")
        if isinstance(n, _ast.Call):
            fname = n.func.id.upper() if isinstance(n.func, _ast.Name) else ""
            args = [_ev(a) for a in n.args]
            if fname in _FUNCS and len(args) == 1:
                return _FUNCS[fname](args[0])
            if fname in _FUNCS2 and len(args) == 2:
                return _FUNCS2[fname](args[0], args[1])
            raise ValueError(f"unsupported func {fname}")
        raise ValueError(f"unsupported node {n!r}")

    return _ev(node)


# AWG wire gauge lookup (diameter in inches) — from enhanced_nec_parser.py
_AWG = {
    "#0": 0.3249, "#1": 0.2893, "#2": 0.2576, "#3": 0.2294,
    "#4": 0.2043, "#6": 0.1620, "#8": 0.1285, "#10": 0.1019,
    "#12": 0.0808, "#14": 0.0641, "#16": 0.0508, "#18": 0.0403,
    "#20": 0.0320, "#22": 0.0253, "#24": 0.0201, "#26": 0.0159,
    "#28": 0.0126, "#30": 0.0100, "#32": 0.0080, "#34": 0.0063,
    "#36": 0.0050, "#38": 0.0040, "#40": 0.0031,
}


def _preprocess_expr(expr: str) -> str:
    """Preprocess SY expressions for AST evaluation.

    Handles xnec2c conventions:
    - AWG wire gauge: #12 -> 0.0808, #12/in -> 0.0808
    - Implicit multiplication: '135 ft' -> '135*ft', '2 pi' -> '2*pi'
    - Builtin constants that are Python keywords: in -> 0.0254
    """
    e = expr.strip()
    # Replace AWG gauges (e.g. #12/in -> diameter_in_inches)
    for gauge, diam in _AWG.items():
        e = e.replace(f"{gauge}/in", str(diam))
        e = e.replace(gauge, str(diam))
    # Insert implicit * BEFORE replacing builtins, so "73.04328 in" becomes
    # "73.04328*in" first, then "73.04328*0.0254" (not "73.04328 0.0254").
    e = re.sub(r'(\d)\s+([a-zA-Z])', r'\1*\2', e)
    # "2pi" -> "2*pi" (no space), but NOT scientific notation "2.67e-3"
    e = re.sub(r'(\d)([a-df-zA-DF-Z])', r'\1*\2', e)
    e = re.sub(r'(\d)([eE])(?![+-]?\d)', r'\1*\2', e)
    # NOW replace xnec2c builtin constants (after implicit * is in place)
    e = re.sub(r'\bin\b', '0.0254', e)
    e = re.sub(r'\bft\b', '0.3048', e)
    e = re.sub(r'\bmm\b', '0.001', e)
    e = re.sub(r'\bcm\b', '0.01', e)
    return e


def _resolve_sy(lines: list[str]) -> list[str]:
    """Resolve SY variable definitions and substitute into subsequent lines.

    Variable names are matched case-insensitively (xnec2c convention).
    """
    # Case-insensitive variable storage: keys are lowercased
    vars_ci: dict[str, float | int] = {}

    # Pass 1 — build symbol table from SY cards
    for line in lines:
        s = line.strip()
        if not _SY_RE.match(s):
            continue
        # Strip inline comments before parsing (xnec2c SY lines often have ' or ! comments)
        s = re.sub(r"\s*['\!].*$", "", s)
        # SY may have multiple comma-separated assignments: SY X=1, Y=2
        _, rest = s.split(None, 1)
        for assignment in rest.split(','):
            assignment = assignment.strip()
            if '=' not in assignment:
                continue
            name, expr = (p.strip() for p in assignment.split('=', 1))
            if not name or not expr:
                continue
            # Try engineering suffix first (e.g. 5.8pF → 5.8e-12)
            eng_val = _parse_eng(expr)
            if eng_val is not None:
                try:
                    vars_ci[name.lower()] = float(eng_val)
                except ValueError:
                    pass
            else:
                try:
                    vars_ci[name.lower()] = _safe_eval(_preprocess_expr(expr), vars_ci)
                except Exception:
                    pass

    # Build regex matching any defined symbol (longest first, case-insensitive)
    if vars_ci:
        sorted_names = sorted(vars_ci.keys(), key=len, reverse=True)
        sym_pat = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_names)) + r')\b', re.IGNORECASE)
    else:
        sym_pat = None  # no symbol substitution, but still run field evaluation

    # Pass 2 — substitute symbols, drop SY lines, evaluate field expressions
    output = []
    for line in lines:
        if _SY_RE.match(line.strip()):
            continue  # remove SY lines — nec2c doesn't understand them
        # Also skip commented-out SY lines (e.g. 'SY Inp=in)
        stripped = line.strip()
        if stripped.startswith("'"):
            output.append(line)
            continue
        def _repl(mo):
            return _fmt_float(vars_ci[mo.group(1).lower()])
        # Protect the 2-char card tag from variable substitution (e.g. FR card vs fr variable)
        parts = line.split(None, 1)
        if sym_pat is not None:
            if len(parts) == 2 and len(parts[0]) >= 2 and parts[0][:2].isalpha():
                card_tag = parts[0]
                rest = sym_pat.sub(_repl, parts[1])
                subst = card_tag + '\t' + rest
            elif len(parts) == 1 and parts[0][:2].isalpha():
                subst = line  # tag-only line, nothing to substitute
            else:
                subst = sym_pat.sub(_repl, line)
        else:
            subst = line

        # Evaluate expressions remaining in card fields (e.g. "20-3.45" from "HGH-Lv")
        # nec2c integer fields MUST NOT contain a decimal point — use CARD_SPECS to cast.
        # Replace commas with spaces first — some NEC files use comma-separated fields
        # and comma-joined tokens cause wrong field-index computation for int casting.
        subst_clean = subst
        tag2 = subst.strip()[:2].upper() if len(subst.strip()) >= 2 else ""
        if tag2 not in ("CM", "CE", "SY") and not subst.strip().startswith("'"):
            subst_clean = subst.replace(',', ' ')
        tokens = subst_clean.split()
        if tokens and len(tokens[0]) >= 2 and tokens[0][:2].isalpha():
            card_type = tokens[0].upper()
            int_indices = _CARD_INT_FIELDS.get(card_type, set())
            new_tokens = [tokens[0]]
            for fi, tok in enumerate(tokens[1:]):
                # Skip tokens that are already plain numbers
                try:
                    v = float(tok)
                    # Format according to expected type
                    if fi in int_indices:
                        new_tokens.append(str(int(round(v))))
                    else:
                        new_tokens.append(tok)  # keep original string repr
                    continue
                except ValueError:
                    pass
                # Try evaluating as expression
                try:
                    val = _safe_eval(tok, vars_ci)
                    if fi in int_indices:
                        new_tokens.append(str(int(round(val))))
                    else:
                        new_tokens.append(_fmt_float(val))
                except Exception:
                    # Check for bare builtin constants (in, ft, mm, pi) which
                    # are Python keywords and can't be parsed by AST.
                    _BUILTINS_MAP = {"mm": 0.001, "cm": 0.01, "in": 0.0254, "ft": 0.3048, "pi": math.pi}
                    low = tok.lower()
                    if low in _BUILTINS_MAP:
                        v_b = _BUILTINS_MAP[low]
                        if fi in int_indices:
                            new_tokens.append(str(int(round(v_b))))
                        else:
                            new_tokens.append(_fmt_float(v_b))
                    else:
                        # Try engineering suffix as last resort
                        eng = _parse_eng(tok)
                        if eng is not None:
                            new_tokens.append(eng)
                        else:
                            new_tokens.append(tok)
            output.append('\t'.join(new_tokens))
        else:
            output.append(subst)
    return output


# ---------------------------------------------------------------------------
# NEC card integer field indices (0-based, after the card tag).
# nec2c Fortran parser rejects '.' in integer columns — these fields
# MUST be formatted as integers even when the SY evaluator returns float.
# Derived from CARD_SPECS in antenna-agent-model enhanced_nec_parser.py.
# ---------------------------------------------------------------------------
_CARD_INT_FIELDS: dict[str, set[int]] = {
    "GW": {0, 1},          # tag, segments
    "GA": {0, 1},          # tag, segments
    "GH": {0, 1},          # tag, segments
    "GC": {0, 1, 2},       # i, j, k
    "GM": {0, 1},          # tag increment, new structures
    "GR": {0, 1},          # tag increment, new structures
    "GS": {0, 1},          # i1, i2 (scale applies to floats after)
    "GX": {0, 1},          # tag increment, reflection flags
    "SP": {0},             # patch shape
    "SM": {0, 1},          # i1, i2
    "GE": {0},             # ground flag
    "EX": {0, 1, 2, 3},    # excitation type, tag, segment, i3
    "FR": {0, 1, 2, 3},    # type, steps, i1, i2
    "GN": {0, 1},          # ground type, n_radials
    "RP": {0, 1, 2, 3},    # mode, theta_count, phi_count, xnda
    "LD": {0, 1, 2, 3},    # type, tag, from_seg, to_seg
    "TL": {0, 1, 2, 3},    # tag1, seg1, tag2, seg2
    "NT": {0, 1, 2, 3},    # tag1, seg1, tag2, seg2
    "NE": {0, 1, 2, 3},    # type, x_count, y_count, z_count
    "NH": {0, 1, 2, 3},    # type, x_count, y_count, z_count
    "EK": {0},             # i1
    "CP": {0, 1, 2, 3},    # tag1, seg1, tag2, seg2
    "PQ": {0, 1, 2, 3},    # print control
    "XQ": {0},             # i1
    "PT": {0, 1, 2, 3},    # print flags
}


# ---------------------------------------------------------------------------
# NEC-4 only cards that nec2c does not support
# ---------------------------------------------------------------------------
_NEC4_CARDS = {"CW", "IS", "VC", "UM", "JN", "LE", "LH", "PS"}

def sanitize_nec(nec_text: str) -> str:
    """
    Prepare a NEC deck for nec2c:
    1. Resolve and remove SY variable definitions.
    2. Remove CM/CE comment cards.
    3. Strip inline comments (' … and ! …).
    4. Remove full-line quote-comments ('...).
    5. Remove NEC-4 only cards that nec2c rejects.
    6. Remove blank / whitespace-only lines.
    7. Enforce deck order: geometry cards, GE, control cards, EN.
    8. Insert GE before first control card if missing.
    9. Append EN if missing.
    """
    geometry_cards = {"GW", "GA", "GH", "GM", "GR", "GS", "GX", "SP", "SM", "SC", "GC"}
    control_cards = {"EX", "FR", "GN", "RP", "LD", "TL", "NT", "NE", "NH", "PQ", "KH",
                     "XQ", "PT", "NX", "WG", "CP", "PL", "EK", "GD"}

    # Step 1: resolve SY variables
    raw_lines = nec_text.splitlines()
    resolved = _resolve_sy(raw_lines)

    # Step 1b: preprocess card fields
    # - Replace commas with spaces (some NEC files use comma-separated fields)
    # - Replace AWG wire gauge notation (#12 -> 0.0808 inches, converted to radius)
    # - Handle percentage notation (50% in segment fields)
    preprocessed = []
    for line in resolved:
        s = line
        # Replace AWG gauges in card fields (e.g. #12 -> half diameter in meters for radius)
        for gauge, diam_in in _AWG.items():
            if gauge in s:
                # #12/in -> diameter in inches, #12 alone -> radius in meters (diam/2 * 0.0254)
                s = s.replace(f"{gauge}/in", str(diam_in))
                s = s.replace(gauge, str(diam_in / 2 * 0.0254))  # radius in meters
        # Replace commas with spaces in card lines (not SY or CM lines)
        stripped = s.strip()
        if stripped and not _SY_RE.match(stripped) and not stripped.startswith("'"):
            tag2 = stripped[:2].upper() if len(stripped) >= 2 else ""
            if tag2 not in ("CM", "CE"):
                s = s.replace(',', ' ')
        # Handle percentage notation: "50%" -> just use the number.
        # xnec2c uses % for percent-of-segments; nec2c doesn't support it.
        # Stripping % is a best-effort approximation.
        s = re.sub(r'(\d+)%', r'\1', s)
        # Collapse multiple whitespace (tabs/spaces) into single space to avoid
        # empty fields that confuse nec2c's fixed-width parser.
        # Preserve leading tag field by splitting on first whitespace.
        parts = s.split()
        if len(parts) > 1 and parts[0][:2].isalpha():
            s = parts[0] + ' ' + ' '.join(parts[1:])
        elif parts:
            s = ' '.join(parts)
        # Resolve engineering suffixes and builtin expressions in card fields
        # (5.8pF -> 5.8e-12, 1.22uH -> 1.22e-6, .2in/ft -> 0.01667)
        if s.strip() and not _SY_RE.match(s.strip()) and not s.strip().startswith("'"):
            toks = s.split()
            if len(toks) > 1 and toks[0][:2].isalpha():
                new_toks = [toks[0]]
                for tok in toks[1:]:
                    # Already a plain number?
                    try:
                        float(tok)
                        new_toks.append(tok)
                        continue
                    except ValueError:
                        pass
                    # Engineering suffix (5.8pF -> 5.8e-12)
                    eng = _parse_eng(tok)
                    if eng is not None:
                        new_toks.append(eng)
                        continue
                    # Compound expression with builtins (.2in/ft, 0.00051181/ft)
                    try:
                        val = _safe_eval(_preprocess_expr(tok), {})
                        new_toks.append(_fmt_float(val))
                        continue
                    except Exception:
                        pass
                    new_toks.append(tok)
                s = ' '.join(new_toks)
        preprocessed.append(s)
    resolved = preprocessed

    geo_lines = []
    ctrl_lines = []
    has_ge = False
    has_en = False

    for line in resolved:
        # Strip inline comments (nec2c doesn't handle ' or ! mid-line)
        line = re.sub(r"\s*['\!].*$", "", line)
        s = line.strip()

        # Skip empty lines
        if not s:
            continue

        tag = s[:2].upper() if len(s) >= 2 else ""

        # Skip lines that don't start with a letter (binary junk, Ctrl-Z EOF marker)
        if not s[0].isalpha():
            continue

        # Skip comment cards, NEC-4 cards, and lines starting with '
        if tag in ("CM", "CE"):
            continue
        if s.startswith("'"):
            continue
        if tag in _NEC4_CARDS:
            continue
        # GF (NGF file load) — not supported by nec2c; skip
        if tag == "GF":
            continue
        if tag == "EN":
            has_en = True
            continue  # we'll add EN at the end
        if tag == "GE":
            has_ge = True
            continue  # we'll insert GE between geo and ctrl

        # Classify into geometry or control
        if tag in geometry_cards:
            geo_lines.append(s)
        elif tag in control_cards:
            ctrl_lines.append(s)
        else:
            # Unknown card — treat as geometry (safer, before GE)
            geo_lines.append(s)

    # nec2c only supports LD types 0-5.  xnec2c/4nec2 type 6 ("series RLC
    # auto-resonance") has explicit R,L,C values, so we can convert to LD 0.
    fixed_ctrl = []
    for cl in ctrl_lines:
        toks = cl.split()
        if len(toks) >= 2 and toks[0].upper() == "LD":
            try:
                ld_type = int(toks[1])
                if ld_type > 5:
                    toks[1] = "0"  # series RLC
                    cl = ' '.join(toks)
            except ValueError:
                pass
        fixed_ctrl.append(cl)
    ctrl_lines = fixed_ctrl

    # Build tag → max_segments map from GW cards so we can clamp
    # EX/LD segment references that exceed the actual segment count
    # (caused by xnec2c "50%" notation stripped to bare "50").
    tag_segs: dict[int, int] = {}
    for gl in geo_lines:
        gt = gl.split()
        if len(gt) >= 3 and gt[0].upper() == "GW":
            try:
                wtag, nsegs = int(gt[1]), int(gt[2])
                if wtag > 0:
                    tag_segs[wtag] = tag_segs.get(wtag, 0) + nsegs
            except ValueError:
                pass
    if tag_segs:
        clamped_ctrl = []
        for cl in ctrl_lines:
            toks = cl.split()
            card = toks[0].upper() if toks else ""
            # EX: fields are type, tag, segment, ...  (indices 1,2,3 after card name → toks[1..3])
            # LD: fields are type, tag, seg_from, seg_to
            if card in ("EX", "LD") and len(toks) >= 4:
                try:
                    wtag = int(toks[2])
                    seg = int(toks[3])
                    max_seg = tag_segs.get(wtag)
                    if max_seg is not None and seg > max_seg:
                        toks[3] = str(max(1, max_seg))
                        cl = ' '.join(toks)
                except ValueError:
                    pass
            clamped_ctrl.append(cl)
        ctrl_lines = clamped_ctrl

    # nec2c 1.3.1 only outputs impedance data when an RP card is present.
    # Inject a minimal RP card if the deck doesn't already contain one.
    has_rp = any(l.split()[0].upper() == "RP" for l in ctrl_lines if l.strip())
    if not has_rp:
        ctrl_lines.append("RP 0 1 1 1000 0 0 0 0")

    # Reassemble in correct nec2c order: geometry → GE → control → EN
    deck = geo_lines + ["GE 0"] + ctrl_lines + ["EN"]

    return "\n".join(deck) + "\n"


def parse_currents(txt: str) -> dict:
    """Parse 'CURRENTS AND LOCATION' section from nec2c output.

    Returns per-segment current data grouped by wire tag, with normalized
    magnitudes suitable for direct use as a heat-map color scale.
    """
    lines = txt.splitlines()

    # Extract frequency from output header
    freq_mhz = None
    pat_freq = re.compile(r"FREQUENCY\s*[=:]\s*([\d.Ee+\-]+)\s*MHZ", re.IGNORECASE)
    for line in lines:
        m = pat_freq.search(line)
        if m:
            try:
                freq_mhz = float(m.group(1))
            except ValueError:
                pass
            break

    # Find and parse the currents table
    in_section = False
    header_skip = 0
    segments: list[dict] = []

    for line in lines:
        upper = line.upper()
        if "CURRENTS AND LOCATION" in upper:
            in_section = True
            header_skip = 3  # skip "DISTANCES IN WAVELENGTHS", column header lines
            continue
        if in_section:
            if header_skip > 0:
                header_skip -= 1
                continue
            stripped = line.strip()
            if not stripped:
                break  # blank line ends the section
            # Detect next section header (all-caps with dashes)
            if stripped.startswith("---") or "RADIATION" in upper or "POWER" in upper:
                break
            tokens = stripped.split()
            if len(tokens) < 10:
                continue
            try:
                seg_no = int(tokens[0])
                tag_no = int(tokens[1])
                real_i = float(tokens[6])
                imag_i = float(tokens[7])
                mag = float(tokens[8])
                phase = float(tokens[9])
                segments.append({
                    "seg": seg_no, "tag": tag_no,
                    "real": real_i, "imag": imag_i,
                    "mag": mag, "phase": phase,
                })
            except (ValueError, IndexError):
                continue

    if not segments:
        return {"freq_mhz": freq_mhz, "max_magnitude": 0, "n_segments": 0, "by_tag": {}}

    max_mag = max(s["mag"] for s in segments)

    # Group by tag — each tag gets ordered arrays of normalized magnitude + phase
    by_tag: dict[str, dict] = {}
    for s in segments:
        key = str(s["tag"])
        if key not in by_tag:
            by_tag[key] = {"magnitudes": [], "phases": []}
        by_tag[key]["magnitudes"].append(s["mag"] / max_mag if max_mag > 0 else 0.0)
        by_tag[key]["phases"].append(s["phase"])

    return {
        "freq_mhz": freq_mhz,
        "max_magnitude": max_mag,
        "n_segments": len(segments),
        "by_tag": by_tag,
    }


def parse_nec_output(txt: str, z0: float = 50.0):
    """Parse NEC2/nec2c textual output to extract impedance (R+jX) vs frequency and compute SWR.

    Tries combined line formats, separate frequency/impedance lines, and (fallback) ANTENNA INPUT PARAMETERS table.
    """
    freqs: list[float] = []
    rs: list[float] = []
    xs: list[float] = []
    swr: list[float] = []

    pat_combined = re.compile(
        r"FREQ(?:UENCY)?\s*= ?\s*([\d.+\-Ee]+)\s*MHZ.*?(?:Z(?:IN|INP)?|IMPED(?:ANCE)?)" \
        r"[^=]*=?\s*\(?\s*([\-+\d.Ee]+)\s*(?:[,\s;+]+|\+\s*j)j?([\-+\d.Ee]+)",
        re.IGNORECASE,
    )
    pat_freq = re.compile(r"FREQ(?:UENCY)?\s*=?\s*([\d.+\-Ee]+)\s*MHZ", re.IGNORECASE)
    pat_zin = re.compile(
        r"Z(?:IN|INP)\s*=\s*\(?\s*([\-+\d.Ee]+)\s*(?:[,\s;+]+|\+\s*j)j?([\-+\d.Ee]+)",
        re.IGNORECASE,
    )
    pat_imped = re.compile(
        r"IMPED(?:ANCE)?[^=]*=\s*\(?\s*([\-+\d.Ee]+)\s*(?:[,\s;+]+|\+\s*j)j?([\-+\d.Ee]+)",
        re.IGNORECASE,
    )

    last_freq: float | None = None
    seen_freqs = set()
    lines = txt.splitlines()

    for line in lines:
        u = line.upper()
        if not any(k in u for k in ("FREQ", "MHZ", "ZIN", "IMPED", "INPUT")):
            continue
        m = pat_combined.search(line)
        if m:
            try:
                f_val = float(m.group(1)); r_val = float(m.group(2)); x_val = float(m.group(3))
            except Exception:
                f_val = r_val = x_val = None
            if f_val is not None and r_val is not None and x_val is not None and f_val not in seen_freqs:
                last_freq = f_val
                seen_freqs.add(f_val)
                freqs.append(f_val); rs.append(r_val); xs.append(x_val)
                try:
                    zin = complex(r_val, x_val)
                    gamma = abs((zin - z0) / (zin + z0))
                    swr_val = (1 + gamma) / (1 - gamma) if gamma < 1 else float("inf")
                except Exception:
                    swr_val = float("nan")
                swr.append(swr_val)
            continue
        f_match = pat_freq.search(line)
        if f_match:
            try:
                last_freq = float(f_match.group(1))
            except Exception:
                last_freq = None
            continue
        if last_freq is not None and last_freq not in seen_freqs:
            z_match = pat_zin.search(line) or pat_imped.search(line)
            if z_match:
                try:
                    r_val = float(z_match.group(1)); x_val = float(z_match.group(2))
                    freqs.append(last_freq); rs.append(r_val); xs.append(x_val)
                    seen_freqs.add(last_freq)
                    zin = complex(r_val, x_val)
                    gamma = abs((zin - z0) / (zin + z0))
                    swr_val = (1 + gamma) / (1 - gamma) if gamma < 1 else float("inf")
                    swr.append(swr_val)
                except Exception:
                    last_freq = None
                continue

    # Always attempt table-based augmentation to capture any additional frequencies (multiband) not matched above.
    pat_freq_line = re.compile(r"FREQUENCY\s*:\s*([\d.Ee+\-]+)\s*MHZ", re.IGNORECASE)
    current_freq = None
    in_input_table = False
    captured_for_freq = False
    for line in lines:
        u = line.upper()
        m = pat_freq_line.search(u)
        if m:
            try:
                current_freq = float(m.group(1))
            except Exception:
                current_freq = None
            captured_for_freq = False
        if 'ANTENNA INPUT PARAMETERS' in u:
            in_input_table = True
            captured_for_freq = False
            continue
        if in_input_table:
            if (not line.strip()) or ('CURRENTS AND LOCATION' in u):
                in_input_table = False
                continue
            if set(line.strip()) <= set('- '):
                continue
            if captured_for_freq:
                continue
            tokens = re.split(r"\s+", line.strip())
            numeric = []
            for t in tokens:
                try:
                    numeric.append(float(t))
                except ValueError:
                    numeric.append(None)
            if len(numeric) >= 8 and isinstance(numeric[6], float) and isinstance(numeric[7], float) and current_freq is not None:
                r_val, x_val = numeric[6], numeric[7]
                if current_freq not in seen_freqs:
                    freqs.append(current_freq); rs.append(r_val); xs.append(x_val); seen_freqs.add(current_freq)
                    try:
                        zin = complex(r_val, x_val)
                        gamma = abs((zin - z0) / (zin + z0))
                        swr_val = (1 + gamma) / (1 - gamma) if gamma < 1 else float('inf')
                    except Exception:
                        swr_val = float('nan')
                    swr.append(swr_val)
                captured_for_freq = True

    # Augment with capped SWR and quality classification (non-breaking additions)
    swr_cap_value = 1000.0
    swr_capped = []
    quality = []  # 'good' if SWR < 3 else 'poor'
    for v in swr:
        if (not math.isfinite(v)) or v > swr_cap_value:
            swr_capped.append(swr_cap_value)
        else:
            swr_capped.append(v)
        quality.append('good' if (math.isfinite(v) and v < 3.0) else 'poor')

    # Replace inf/nan in the raw swr list too (JSON cannot represent them)
    swr_safe = [v if math.isfinite(v) else swr_cap_value for v in swr]

    # Sanitize impedance values (JSON cannot represent inf/nan)
    rs_safe = [v if math.isfinite(v) else 0.0 for v in rs]
    xs_safe = [v if math.isfinite(v) else 0.0 for v in xs]

    return {
        "swr_sweep": {
            "freq_mhz": freqs,
            "swr": swr_safe,
            "swr_capped": swr_capped,
            "swr_cap": swr_cap_value,
            "quality": quality,
        },
        "impedance_sweep": {"freq_mhz": freqs, "r": rs_safe, "x": xs_safe, "z0": z0},
    }


def parse_pattern_output(txt: str):
    """
    Heuristic parser for NEC2/nec2++ far-field pattern tables.
    Searches for a header containing THETA and PHI and a gain-like column (DB or GAIN),
    then extracts theta, phi, and gain (dB) numbers from subsequent lines until a blank/separator.
    Returns lists; empty lists if not found.
    """
    lines = txt.splitlines()
    theta_vals, phi_vals, gain_vals = [], [], []
    header_idx = -1
    header_cols = []
    gain_col_idx = None
    theta_col_idx = None
    phi_col_idx = None
    gain_keys = (
        'DB', 'GAIN', 'G(DB)', 'G(TOT)', 'GTOT', 'POWER', 'POWERGAIN', 'DIRECT', 'DIRECTIVITY', 'D(DB)'
    )

    # Find header. nec2c prints two header lines; we collapse them and detect indices.
    for i, line in enumerate(lines):
        u = line.upper()
        if 'THETA' in u and 'PHI' in u and any(k in u for k in gain_keys):
            header_idx = i
            # nec2c uses two lines; combine with previous if it's part of header block
            header_block = u
            if i > 0 and ('RADIATION PATTERNS' in lines[i-1].upper() or '---- ANGLES' in lines[i-1].upper()):
                header_block = lines[i-1].upper() + ' ' + u
            # Tokenize header conservatively
            header_cols = re.split(r"\s+", header_block.strip())
            # Try to find indices
            def idx_of(label):
                try:
                    return header_cols.index(label)
                except ValueError:
                    return None
            theta_col_idx = idx_of('THETA')
            phi_col_idx = idx_of('PHI')
            # For gains, prefer 'TOTAL' column in the POWER GAINS block if present
            gain_col_idx = idx_of('TOTAL')
            if gain_col_idx is None:
                for key in ('DB', 'G(DB)', 'GAIN', 'G(TOT)', 'GTOT', 'POWER', 'POWERGAIN', 'DIRECT', 'DIRECTIVITY', 'D(DB)'):
                    if key in header_cols:
                        gain_col_idx = header_cols.index(key)
                        break
            if gain_col_idx is None and header_cols:
                gain_col_idx = len(header_cols) - 1
            break

    if header_idx == -1:
        return {"theta": theta_vals, "phi": phi_vals, "gain": gain_vals}

    # Parse data lines after header
    for j in range(header_idx + 1, len(lines)):
        line = lines[j]
        if not line.strip():
            # stop at blank line
            if theta_vals:
                break
            else:
                continue
        if set(line.strip()) <= set('-='):
            # separator line
            continue
        # Extract tokens; nec2c rows may contain words like LINEAR. We'll split and filter numbers later.
        tokens = re.split(r"\s+", line.strip())
        # Skip lines that don't start with a number
        if not tokens or not re.match(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$", tokens[0]):
            continue
        # Convert to floats where possible, preserving positions; non-numbers become None
        nums = []
        for t in tokens:
            try:
                nums.append(float(t))
            except ValueError:
                nums.append(None)
        # Map columns by index if available; otherwise heuristic: first=theta, second=phi, and TOTAL dB as 5th numeric
        theta_v = None
        phi_v = None
        gain_v = None
        if theta_col_idx is not None and theta_col_idx < len(nums):
            theta_v = nums[theta_col_idx]
        if phi_col_idx is not None and phi_col_idx < len(nums):
            phi_v = nums[phi_col_idx]
        if gain_col_idx is not None and gain_col_idx < len(nums):
            gain_v = nums[gain_col_idx]
        # Heuristics if indices failed: first numeric token ~ theta, second ~ phi
        if theta_v is None:
            theta_v = next((x for x in nums if isinstance(x, float)), None)
        if phi_v is None:
            # find second numeric
            seen = 0
            for x in nums:
                if isinstance(x, float):
                    seen += 1
                    if seen == 2:
                        phi_v = x; break
        if gain_v is None:
            # Prefer the 5th numeric (TOTAL dB) if available; fallback to last numeric
            numerics = [x for x in nums if isinstance(x, float)]
            if len(numerics) >= 5:
                gain_v = numerics[4]
            elif len(numerics) >= 3:
                gain_v = numerics[2]
            elif numerics:
                gain_v = numerics[-1]
        if theta_v is not None and phi_v is not None and gain_v is not None:
            theta_vals.append(theta_v)
            phi_vals.append(phi_v)
            gain_vals.append(gain_v)

    # Sanitize gain values (nec2c can produce -999.99 sentinel for no-gain directions)
    gain_safe = [v if math.isfinite(v) else -999.99 for v in gain_vals]
    return {"theta": theta_vals, "phi": phi_vals, "gain": gain_safe}

@app.get("/healthz")
def health():
    return {"ok": True}

@app.get("/cache/stats")
def cache_stats():
    """Return cache hit/miss stats and total cached entries."""
    try:
        total = sum(1 for f in os.listdir(CACHE_DIR) if f.endswith(".json"))
    except OSError:
        total = 0
    with _cache_lock:
        return {
            "hits": _cache_hits,
            "misses": _cache_misses,
            "cached_entries": total,
            "max_concurrent": _MAX_CONCURRENT,
            "cache_dir": CACHE_DIR,
        }

@app.post("/cache/clear")
def cache_clear():
    """Remove all cached simulation results."""
    removed = 0
    try:
        for f in os.listdir(CACHE_DIR):
            if f.endswith(".json"):
                try:
                    os.remove(os.path.join(CACHE_DIR, f))
                    removed += 1
                except OSError:
                    pass
    except OSError:
        pass
    return {"cleared": removed}

@app.post("/sanitize_debug")
def sanitize_debug(payload: dict = Body(...)):
    """Debug endpoint: return the sanitized NEC deck without running nec2c."""
    nec = payload.get("nec_deck") or payload.get("nec_text") or payload.get("nec") or ""
    return {"sanitized": sanitize_nec(nec)}

@app.post("/run")
def run(payload: dict = Body(...)):
    # Accept legacy embedded form {"nec_deck": "..."} plus optional z0, dump_raw
    if not isinstance(payload, dict):
        return {"ok": False, "error": "invalid_payload"}
    nec_deck = payload.get("nec_deck") or payload.get("nec") or ""
    if not nec_deck:
        return {"ok": False, "error": "missing_nec_deck"}
    z0 = float(payload.get("z0", 50.0))
    dump_raw = bool(payload.get("dump_raw"))

    sanitized = sanitize_nec(nec_deck)

    # --- Cache check (skip when dump_raw requested) ---
    if not dump_raw:
        key = _cache_key(sanitized, "run", z0)
        cached = _cache_get(key)
        if cached is not None:
            return cached

    # --- Semaphore: limit concurrent nec2c processes ---
    if not _sim_semaphore.acquire(timeout=60):
        return {"ok": False, "error": "server_busy",
                "detail": "Too many concurrent simulations; try again shortly"}
    try:
        with tempfile.TemporaryDirectory() as td:
            inp, outp = f"{td}/model.nec", f"{td}/out.txt"
            with open(inp, "w") as f:
                f.write(sanitized)
            cmd = [NEC_BIN, f"-i{inp}", f"-o{outp}"]
            p = subprocess.run(cmd, text=True, capture_output=True, timeout=120)
            if p.returncode != 0:
                return {"ok": False, "error": "nec_failed", "stderr": p.stderr, "stdout": p.stdout}
            try:
                with open(outp) as rf:
                    raw = rf.read()
            except Exception:
                raw = ""
            if dump_raw and raw:
                try:
                    h = hashlib.sha256(nec_deck.encode()).hexdigest()[:10]
                    ts = int(time.time())
                    out_dir = "/raw_out"
                    if os.path.isdir(out_dir):
                        try:
                            os.chmod(out_dir, 0o777)
                        except Exception:
                            pass
                        fname = f"run_{ts}_{h}.txt"
                        with open(os.path.join(out_dir, fname), "w") as rf:
                            rf.write(raw)
                except Exception:
                    pass
    finally:
        _sim_semaphore.release()

    result = {"ok": True, "parsed": parse_nec_output(raw, z0=z0)}
    # Cache successful results
    if not dump_raw:
        _cache_put(key, result)
    return result


@app.post("/currents")
def currents(payload: dict = Body(...)):
    """Extract per-segment structure currents from nec2c output.

    Returns normalized current magnitudes grouped by wire tag, suitable
    for heat-map overlay on a 3D wire model.
    """
    nec_deck = payload.get("nec_deck") or payload.get("nec") or ""
    if not nec_deck:
        return {"ok": False, "error": "missing_nec_deck"}

    sanitized = sanitize_nec(nec_deck)

    key = _cache_key(sanitized, "currents")
    cached = _cache_get(key)
    if cached is not None:
        return cached

    if not _sim_semaphore.acquire(timeout=60):
        return {"ok": False, "error": "server_busy",
                "detail": "Too many concurrent simulations; try again shortly"}
    try:
        with tempfile.TemporaryDirectory() as td:
            inp, outp = f"{td}/model.nec", f"{td}/out.txt"
            with open(inp, "w") as f:
                f.write(sanitized)
            cmd = [NEC_BIN, f"-i{inp}", f"-o{outp}"]
            p = subprocess.run(cmd, text=True, capture_output=True, timeout=120)
            if p.returncode != 0:
                return {"ok": False, "error": "nec_failed", "stderr": p.stderr}
            try:
                with open(outp) as rf:
                    raw = rf.read()
            except Exception:
                raw = ""
    finally:
        _sim_semaphore.release()

    parsed = parse_currents(raw)
    if parsed["n_segments"] == 0:
        return {"ok": False, "error": "no_currents_found"}

    result = {"ok": True, **parsed}
    _cache_put(key, result)
    return result


@app.post("/pattern")
def pattern(payload: dict = Body(...)):
    """
    Execute nec2++ with the provided NEC deck and parse far-field pattern data.
    Expects the deck to include an RP card. Returns theta, phi, and gain (dB where available).
    """
    # Back-compat: allow either raw dict or embedded nec_text
    if isinstance(payload, dict):
        nec_text = payload.get('nec_text') if 'nec_text' in payload else payload.get('nec')
        debug = bool(payload.get('debug') or payload.get('include_raw'))
    else:
        nec_text = str(payload)
        debug = False
    if not nec_text:
        return {"ok": False, "error": "missing_nec_text"}

    sanitized = sanitize_nec(nec_text)

    # --- Cache check (skip when debug requested) ---
    if not debug:
        key = _cache_key(sanitized, "pattern")
        cached = _cache_get(key)
        if cached is not None:
            return cached

    # --- Semaphore: limit concurrent nec2c processes ---
    if not _sim_semaphore.acquire(timeout=60):
        return {"ok": False, "error": "server_busy",
                "detail": "Too many concurrent simulations; try again shortly"}
    try:
        with tempfile.TemporaryDirectory() as td:
            inp, outp = f"{td}/model.nec", f"{td}/out.txt"
            with open(inp, "w") as f:
                f.write(sanitized)
            cmd = [NEC_BIN, f"-i{inp}", f"-o{outp}"]
            p = subprocess.run(cmd, text=True, capture_output=True, timeout=180)
            if p.returncode != 0:
                resp = {"ok": False, "error": "nec_failed", "stderr": p.stderr, "stdout": p.stdout}
                if debug:
                    try:
                        from pathlib import Path
                        resp["cmd"] = cmd
                        resp["cwd"] = str(Path.cwd())
                        resp["inp_exists"] = Path(inp).exists()
                        resp["out_exists"] = Path(outp).exists()
                        if Path(outp).exists():
                            resp["raw"] = open(outp).read()[-4000:]
                    except Exception:
                        pass
                return resp
            raw = open(outp).read()
            if debug:
                try:
                    h = hashlib.sha256(nec_text.encode()).hexdigest()[:10]
                    ts = int(time.time())
                    out_dir = "/raw_out"
                    if os.path.isdir(out_dir):
                        try:
                            os.chmod(out_dir, 0o777)
                        except Exception:
                            pass
                        fname = f"pattern_{ts}_{h}.txt"
                        with open(os.path.join(out_dir, fname), "w") as rf:
                            rf.write(raw)
                except Exception:
                    pass
    finally:
        _sim_semaphore.release()

    # Always attempt to parse both pattern and impedance from a single solver run so the
    # caller can avoid two separate container executions. This unifies the data path for
    # the NEC JSON Analyzer (pattern + impedance/swr) while remaining backwards compatible
    # with earlier clients that only consumed pattern arrays.
    pat = parse_pattern_output(raw)
    sweep = parse_nec_output(raw)  # may contain empty lists if frequency/Z lines absent
    if not pat.get("theta"):
        resp = {"ok": False, "error": "no_pattern_detected"}
        if debug:
            # Include a small window around any detected header-like lines for troubleshooting
            try:
                lines = raw.splitlines()
                hdr_idx = -1
                for i, line in enumerate(lines):
                    u = line.upper()
                    if 'THETA' in u and 'PHI' in u:
                        hdr_idx = i
                        break
                window = []
                if hdr_idx != -1:
                    start = max(0, hdr_idx - 5)
                    end = min(len(lines), hdr_idx + 20)
                    window = lines[start:end]
                resp["raw_header_window"] = "\n".join(window)[-4000:]
            except Exception:
                pass
        return resp
    resp = {"ok": True, **pat}
    # Attach impedance / SWR sweeps when available (even if empty for single-frequency decks)
    if sweep:
        if sweep.get("impedance_sweep"):
            resp["impedance_sweep"] = sweep["impedance_sweep"]
        if sweep.get("swr_sweep"):
            resp["swr_sweep"] = sweep["swr_sweep"]
    if debug:
        # Attach a brief trailer of raw output for validation
        try:
            resp["raw_tail"] = raw[-2000:]
        except Exception:
            pass
    # Cache successful results
    if not debug:
        _cache_put(key, resp)
    return resp
