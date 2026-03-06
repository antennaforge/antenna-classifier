"""
NEC file validator.

Checks structural and domain-compliance rules to determine if a NEC file
is a valid, runnable antenna model.
"""

from dataclasses import dataclass, field
from enum import Enum

from .parser import NECCard, ParseResult


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Issue:
    severity: Severity
    message: str
    line: int | None = None
    card_type: str | None = None


@dataclass
class ValidationResult:
    source: str
    valid: bool = True
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    def add(self, severity: Severity, msg: str, *, line: int | None = None, card_type: str | None = None) -> None:
        self.issues.append(Issue(severity, msg, line, card_type))
        if severity == Severity.ERROR:
            self.valid = False


def validate(parsed: ParseResult) -> ValidationResult:
    """Run all validation checks against a parsed NEC file."""
    result = ValidationResult(source=parsed.source)

    _check_parse_errors(parsed, result)
    _check_geometry(parsed, result)
    _check_excitation(parsed, result)
    _check_frequency(parsed, result)
    _check_ground(parsed, result)
    _check_en_card(parsed, result)
    _check_ge_card(parsed, result)
    _check_card_order(parsed, result)
    _check_wire_params(parsed, result)
    _check_tag_references(parsed, result)

    return result


def _check_parse_errors(parsed: ParseResult, result: ValidationResult) -> None:
    """Propagate card-level parse errors."""
    for card in parsed.cards:
        for err in card.errors:
            result.add(Severity.ERROR, f"{card.card_type}: {err}",
                       line=card.line_number, card_type=card.card_type)
    for w in parsed.warnings:
        result.add(Severity.WARNING, w)


def _check_geometry(parsed: ParseResult, result: ValidationResult) -> None:
    """Must have at least one geometry card."""
    geo = parsed.geometry_cards
    if not geo:
        result.add(Severity.ERROR, "No geometry cards (GW, GA, etc.) found")
    wires = parsed.wire_cards
    if wires:
        result.add(Severity.INFO, f"{len(wires)} GW wire(s) defined")


def _check_excitation(parsed: ParseResult, result: ValidationResult) -> None:
    """Must have at least one EX card."""
    ex_cards = [c for c in parsed.cards if c.card_type == "EX"]
    if not ex_cards:
        result.add(Severity.ERROR, "No excitation (EX) card found")


def _check_frequency(parsed: ParseResult, result: ValidationResult) -> None:
    """Must have at least one FR card."""
    fr_cards = [c for c in parsed.cards if c.card_type == "FR"]
    if not fr_cards:
        result.add(Severity.ERROR, "No frequency (FR) card found")
    for fr in fr_cards:
        freq = fr.labeled_params.get("freq")
        if isinstance(freq, (int, float)) and freq <= 0:
            result.add(Severity.ERROR, f"FR frequency must be > 0, got {freq}",
                       line=fr.line_number, card_type="FR")


def _check_ground(parsed: ParseResult, result: ValidationResult) -> None:
    """Check GN card if present."""
    gn_cards = [c for c in parsed.cards if c.card_type == "GN"]
    if not gn_cards:
        result.add(Severity.INFO, "No GN card — free-space model assumed")


def _check_en_card(parsed: ParseResult, result: ValidationResult) -> None:
    """EN card should be present (many NEC engines require it)."""
    en_cards = [c for c in parsed.cards if c.card_type == "EN"]
    if not en_cards:
        result.add(Severity.WARNING, "No EN (end) card — some NEC engines require it")


def _check_ge_card(parsed: ParseResult, result: ValidationResult) -> None:
    """GE card terminates geometry section."""
    ge_cards = [c for c in parsed.cards if c.card_type == "GE"]
    if not ge_cards:
        result.add(Severity.WARNING, "No GE card — geometry section not explicitly terminated")


def _check_card_order(parsed: ParseResult, result: ValidationResult) -> None:
    """Geometry cards must come before control/excitation cards (after GE)."""
    geo_types = {"GW", "GA", "GH", "GM", "GX", "GR", "GS", "GC", "SP", "SM", "SC"}
    control_types = {"EX", "FR", "LD", "GN", "RP", "TL", "NT", "NE", "NH", "XQ", "EN", "CP", "PT", "KH", "EK"}
    seen_control = False
    for card in parsed.cards:
        if card.card_type in ("CM", "CE", "SY", "GE"):
            if card.card_type == "GE":
                seen_control = True
            continue
        if card.card_type in control_types:
            seen_control = True
        elif card.card_type in geo_types and seen_control:
            result.add(Severity.WARNING,
                       f"Geometry card {card.card_type} appears after control cards",
                       line=card.line_number, card_type=card.card_type)


def _check_wire_params(parsed: ParseResult, result: ValidationResult) -> None:
    """Validate wire segment counts and radii."""
    for card in parsed.wire_cards:
        lp = card.labeled_params
        segments = lp.get("segments")
        if isinstance(segments, int) and segments < 1:
            result.add(Severity.ERROR, f"GW segments must be >= 1, got {segments}",
                       line=card.line_number, card_type="GW")
        radius = lp.get("radius")
        if isinstance(radius, (int, float)) and radius <= 0:
            result.add(Severity.ERROR, f"GW radius must be > 0, got {radius}",
                       line=card.line_number, card_type="GW")
        # Check for zero-length wire
        x1, y1, z1 = lp.get("x1"), lp.get("y1"), lp.get("z1")
        x2, y2, z2 = lp.get("x2"), lp.get("y2"), lp.get("z2")
        if all(isinstance(v, (int, float)) for v in (x1, y1, z1, x2, y2, z2)):
            length = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2) ** 0.5
            if length == 0:
                result.add(Severity.ERROR, "Zero-length wire",
                           line=card.line_number, card_type="GW")


def _check_tag_references(parsed: ParseResult, result: ValidationResult) -> None:
    """Check that EX/LD/TL tags reference defined wire tags."""
    defined_tags: set[int] = set()
    for card in parsed.cards:
        if card.card_type == "GW":
            tag = card.labeled_params.get("tag")
            if isinstance(tag, int):
                defined_tags.add(tag)

    if not defined_tags:
        return

    # Check EX source tags
    for card in parsed.cards:
        if card.card_type == "EX":
            tag = card.labeled_params.get("tag")
            if isinstance(tag, int) and tag not in defined_tags and tag != 0:
                result.add(Severity.WARNING,
                           f"EX references undefined wire tag {tag}",
                           line=card.line_number, card_type="EX")
        elif card.card_type == "LD":
            tag = card.labeled_params.get("tag")
            if isinstance(tag, int) and tag not in defined_tags and tag != 0:
                result.add(Severity.WARNING,
                           f"LD references undefined wire tag {tag}",
                           line=card.line_number, card_type="LD")
