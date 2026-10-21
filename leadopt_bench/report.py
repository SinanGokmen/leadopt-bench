"""Presentation layer: colour-coded verdicts for notebooks.

Kept out of `metrics.py` so the scoring stays free of formatting concerns. The
colour carries one message only -- how much the published record settles -- and
deliberately does not say whether a molecule is *good*:

  green   the chemists made this and it worked. Settled by measurement.
  orange  they made it and it did not help. Also settled.
  blue    nobody made it. NOT a judgement -- there is no ground truth, so this
          is the one a medicinal chemist has to answer.
  red     not a usable answer: unparseable, or it abandoned the scaffold.
  grey    not an answer at all: the lead returned unchanged, or a repeat.
"""

from __future__ import annotations

from .metrics import SuggestionScore, explain

# category -> (colour, short label)
VERDICT_STYLE: dict[str, tuple[str, str]] = {
    "recovered_target": ("#1a7f37", "CHEMISTS MADE THIS — and it was more potent"),
    "recovered_nonimprover": ("#bc4c00", "chemists made this — but it did not help"),
    "novel_plausible": ("#0969da", "nobody made this — a medicinal chemist has to judge it"),
    "scaffold_violation": ("#cf222e", "left the scaffold — this is not lead optimization"),
    "invalid": ("#cf222e", "not a valid molecule"),
    "unchanged": ("#6e7781", "this is the lead itself — nothing changed"),
    "duplicate": ("#6e7781", "already proposed"),
}
NEUTRAL = "#6e7781"


def verdict_colour(category: str) -> str:
    return VERDICT_STYLE.get(category, (NEUTRAL, category))[0]


def verdict_label(category: str) -> str:
    return VERDICT_STYLE.get(category, (NEUTRAL, category))[1]


def verdict_html(score: SuggestionScore, series: dict | None = None) -> str:
    """One self-contained HTML block: the verdict, then the evidence behind it."""
    colour = verdict_colour(score.category)
    label = verdict_label(score.category)

    rows: list[tuple[str, str]] = []
    if series:
        rows.append(("series", f"{series.get('target', '')} "
                               f"({series.get('series_id', '')})"))
        rows.append(("lead potency", f"p{series.get('endpoint', '')} = "
                                     f"{series.get('lead_pchembl', '')}"))
    if score.valid:
        rows.append(("similarity to lead",
                     f"Tanimoto {score.tanimoto_to_lead:.3f} "
                     f"(this series' floor: {score.similarity_floor:.3f})"))
        rows.append(("same scaffold", "yes" if score.same_murcko_scaffold else "no"))
        if score.heavy_atom_delta:
            rows.append(("size change",
                         f"{score.heavy_atom_delta:+d} heavy atoms, "
                         f"{score.mw_delta:+.0f} Da"))
    else:
        rows.append(("why", score.invalid_reason))

    if score.matched_series_molecule:
        measured = (f", measured p-value <b>{score.matched_series_pchembl}</b>"
                    if score.matched_series_pchembl is not None else "")
        rows.append(("matched real compound",
                     f"{score.matched_series_molecule}{measured}"))
    if score.direction_precision is not None:
        detail = []
        if score.direction_hits:
            detail.append("worked in this series: " + ", ".join(score.direction_hits))
        if score.direction_misses:
            detail.append("never tried here: " + ", ".join(score.direction_misses))
        rows.append(("change-type accuracy",
                     f"{score.direction_precision:.2f} "
                     f"<span style='color:#6e7781'>(null floor ≈0.53-0.61)</span>"
                     + ("<br><span style='font-size:90%;color:#6e7781'>"
                        + "; ".join(detail) + "</span>" if detail else "")))

    warn = ""
    if score.ungrounded_claims:
        warn = (f"<div style='margin-top:10px;padding:8px 10px;background:#fff8c5;"
                f"border-left:4px solid #bf8700;font-size:93%'>"
                f"<b>Rationale inconsistent.</b> The text names structures that are in "
                f"neither the lead nor this proposal: "
                f"<b>{', '.join(score.ungrounded_claims)}</b>.</div>")

    body = "".join(
        f"<tr><td style='padding:3px 14px 3px 0;color:#6e7781;"
        f"vertical-align:top;white-space:nowrap'>{k}</td>"
        f"<td style='padding:3px 0'>{v}</td></tr>"
        for k, v in rows)

    return (
        f"<div style='font-family:-apple-system,Segoe UI,Roboto,sans-serif;"
        f"max-width:760px;border:1px solid #d0d7de;border-radius:8px;overflow:hidden'>"
        f"<div style='background:{colour};color:#fff;padding:10px 14px;"
        f"font-weight:600;font-size:105%'>{label}</div>"
        f"<div style='padding:12px 14px'>"
        f"<table style='border-collapse:collapse;font-size:94%'>{body}</table>"
        f"{warn}</div></div>"
    )


def legend_html() -> str:
    """Every verdict the tool can return, with its colour."""
    order = ["recovered_target", "recovered_nonimprover", "novel_plausible",
             "scaffold_violation", "invalid", "unchanged", "duplicate"]
    meaning = {
        "recovered_target": "Your molecule is in the published series and it was more "
                            "potent. Settled by measurement.",
        "recovered_nonimprover": "It is in the series, but it did not raise potency. "
                                 "Also settled — someone already tried this.",
        "novel_plausible": "Valid, on-scaffold, absent from the series. <b>Not scored.</b> "
                           "There is no ground truth, so only a chemist can judge it.",
        "scaffold_violation": "Further from the lead than the real campaign ever went. "
                              "Scaffold hopping, not lead optimization.",
        "invalid": "RDKit cannot parse or sanitize it, or it is outside the mass range.",
        "unchanged": "Identical to the starting compound — a described change that was "
                     "never made.",
        "duplicate": "The same molecule proposed twice in one response.",
    }
    rows = "".join(
        f"<tr>"
        f"<td style='padding:6px 10px 6px 0;vertical-align:top'>"
        f"<span style='display:inline-block;width:12px;height:12px;border-radius:3px;"
        f"background:{verdict_colour(c)}'></span></td>"
        f"<td style='padding:6px 12px 6px 0;vertical-align:top;white-space:nowrap'>"
        f"<code>{c}</code></td>"
        f"<td style='padding:6px 0;vertical-align:top'>{meaning[c]}</td></tr>"
        for c in order)
    return (
        f"<div style='font-family:-apple-system,Segoe UI,Roboto,sans-serif;"
        f"max-width:800px;font-size:94%'>"
        f"<table style='border-collapse:collapse'>{rows}</table>"
        f"<p style='color:#6e7781;margin-top:12px'>Colour reflects <b>how much the "
        f"published record settles</b>, not whether a molecule is good. Blue is the "
        f"interesting case: the tool deliberately refuses to score it.</p></div>"
    )


def text_verdict(score: SuggestionScore) -> str:
    """Plain-text fallback for terminals, same content as verdict_html."""
    return f"{verdict_label(score.category)}\n  {explain(score)}"
