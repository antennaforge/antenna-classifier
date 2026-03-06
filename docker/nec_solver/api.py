from fastapi import FastAPI, Body
import tempfile, subprocess, shlex, re, json, os, time, hashlib, math

# Allow selecting solver binary via environment (nec2c, nec2++, xnec2c). Default nec2c.
NEC_BIN = os.getenv("NEC_BIN", "nec2c")

app = FastAPI(title="NEC Solver API", version="1.0")

def sanitize_nec(nec_text: str) -> str:
    """
    Prepare a NEC deck for nec2c:
    - Remove CM/CE comment cards (nec2c can choke on formatting).
    - Insert a GE card before the first control card if missing.
    - Append an EN card at the end if missing.
    """
    geometry_cards = {"GW", "GA", "GH", "GM", "GR", "GS", "GX", "SP", "SM", "SC"}
    control_cards = {"EX", "FR", "GN", "RP", "LD", "TL", "NT", "NE", "NH", "PQ", "KH",
                     "XQ", "PT", "NX", "WG", "CP", "PL"}
    cleaned = []
    has_ge = False
    has_en = False
    for line in nec_text.splitlines():
        s = line.lstrip()
        tag = s[:2].upper() if len(s) >= 2 else ""
        if tag in ("CM", "CE"):
            continue
        if tag == "GE":
            has_ge = True
        if tag == "EN":
            has_en = True
        cleaned.append(line)

    # Insert GE before first control card if missing
    if not has_ge:
        insert_idx = len(cleaned)
        for i, line in enumerate(cleaned):
            s = line.lstrip()
            tag = s[:2].upper() if len(s) >= 2 else ""
            if tag in control_cards:
                insert_idx = i
                break
        cleaned.insert(insert_idx, "GE 0")

    # Append EN if missing
    if not has_en:
        cleaned.append("EN")

    return "\n".join(cleaned) + "\n"

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

    return {
        "swr_sweep": {
            "freq_mhz": freqs,
            "swr": swr,
            "swr_capped": swr_capped,
            "swr_cap": swr_cap_value,
            "quality": quality,
        },
        "impedance_sweep": {"freq_mhz": freqs, "r": rs, "x": xs, "z0": z0},
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

    return {"theta": theta_vals, "phi": phi_vals, "gain": gain_vals}

@app.get("/healthz")
def health():
    return {"ok": True}

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
    with tempfile.TemporaryDirectory() as td:
        inp, outp = f"{td}/model.nec", f"{td}/out.txt"
        with open(inp, "w") as f:
            f.write(sanitize_nec(nec_deck))
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
    return {"ok": True, "parsed": parse_nec_output(raw, z0=z0)}


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
    with tempfile.TemporaryDirectory() as td:
        inp, outp = f"{td}/model.nec", f"{td}/out.txt"
        with open(inp, "w") as f:
            f.write(sanitize_nec(nec_text))
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
    return resp
