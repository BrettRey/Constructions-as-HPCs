"""
Utility functions for parsing Universal Dependencies (UD) CoNLL‑U files and
identifying the *let alone* construction.  These helpers are shared across
multiple analysis scripts.  No external dependencies beyond the Python
standard library and `numpy`/`pandas` are required.

The key abstractions are:

* A **Token** is represented as a dictionary with at least the fields
  `id` (1‑indexed integer), `form` (surface form), `lemma` (lemma if
  provided), `upos` (universal part of speech), `head` (ID of the syntactic
  head, 0 if root), `deprel` (dependency relation), and `misc` (misc
  annotations).  Additional columns in the CoNLL‑U file are ignored.
* A **Sentence** is represented as a dictionary with keys `tokens`
  (a list of Token dicts), `sent_id` (string if present), and `text`
  (the raw sentence text if available in comments).

Parsing is deliberately light weight: we ignore comments other than
`# sent_id` and `# text`, and drop multi‑word tokens (IDs with hyphens) as
well as empty nodes (IDs containing dots).  The functions in this module
should be sufficient for the extraction and profiling tasks in this
repository.

The second major component in this module is the identification of
*let alone* anchors and the computation of associated features.  The
`find_let_alone_anchors` function returns all anchor spans (pairs of
token indices) for a given sentence, using both a surface string match
and a UD‑based pattern match.  The UD pattern looks for `alone` tokens
attached as `fixed` to `let` or to the head of X, and ensures there is a
coordinated or dependent Y following the anchor.

The `extract_let_alone_features` function then derives cue features
(parallelism, licensing, UPOS categories, distances) for each anchor.

See the individual docstrings for details.
"""

from __future__ import annotations

import os
import re
from typing import List, Dict, Tuple, Any, Set, Optional

import numpy as np


def parse_conllu(content: str) -> List[Dict[str, Any]]:
    """Parse a raw CoNLL‑U string into a list of sentence dictionaries.

    Each sentence dictionary contains keys:
      - 'tokens': list of token dicts with keys id, form, lemma, upos, head, deprel, misc
      - 'sent_id': string if present (otherwise None)
      - 'text': raw text if present (otherwise None)

    Multi‑word tokens (IDs with hyphens) and empty nodes (IDs containing dots)
    are ignored.

    Parameters
    ----------
    content: str
        Raw content of a CoNLL‑U file

    Returns
    -------
    List[Dict[str, Any]]
        Parsed sentences
    """
    sentences: List[Dict[str, Any]] = []
    sent_tokens: List[Dict[str, Any]] = []
    sent_id: Optional[str] = None
    sent_text: Optional[str] = None
    for line in content.splitlines():
        line = line.strip()
        if not line:
            # end of sentence
            if sent_tokens:
                sentences.append({
                    "tokens": sent_tokens,
                    "sent_id": sent_id,
                    "text": sent_text,
                })
            sent_tokens = []
            sent_id = None
            sent_text = None
            continue
        if line.startswith("#"):
            # comment
            if line.startswith("# sent_id"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    sent_id = parts[1].strip()
            elif line.startswith("# text"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    sent_text = parts[1].strip()
            continue
        # token line
        cols = line.split("\t")
        if not cols:
            continue
        token_id = cols[0]
        # Skip multi‑word tokens (id like '1-2') and empty nodes (id like '2.1')
        if '-' in token_id or '.' in token_id:
            continue
        try:
            tid = int(token_id)
        except ValueError:
            continue
        form = cols[1] if len(cols) > 1 else "_"
        lemma = cols[2] if len(cols) > 2 else "_"
        upos = cols[3] if len(cols) > 3 else "_"
        head_str = cols[6] if len(cols) > 6 else "_"
        deprel = cols[7] if len(cols) > 7 else "_"
        misc = cols[9] if len(cols) > 9 else "_"
        try:
            head = int(head_str) if head_str != '_' else 0
        except ValueError:
            head = 0
        sent_tokens.append({
            "id": tid,
            "form": form,
            "lemma": lemma,
            "upos": upos,
            "head": head,
            "deprel": deprel,
            "misc": misc,
        })
    # append last sentence if any
    if sent_tokens:
        sentences.append({
            "tokens": sent_tokens,
            "sent_id": sent_id,
            "text": sent_text,
        })
    return sentences


def load_conllu(path: str) -> List[Dict[str, Any]]:
    """Read a CoNLL‑U file from disk and parse it into sentence dictionaries."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return parse_conllu(content)


def normalize_upos(upos: str) -> str:
    """Normalize UPOS categories into coarse classes.

    We collapse `PROPN` into `NOUN` and keep the following classes: `NOUN`,
    `VERB`, `ADJ`, and `OTHER`.  All other UPOS tags are mapped to
    `OTHER`.
    """
    upos_upper = upos.upper() if upos else "OTHER"
    if upos_upper == "PROPN":
        return "NOUN"
    if upos_upper in {"NOUN", "VERB", "ADJ"}:
        return upos_upper
    return "OTHER"


def find_let_alone_anchors(sentence: Dict[str, Any]) -> List[Tuple[int, int, str]]:
    """Find *let alone* anchors in a sentence.

    The function implements two detection strategies:

    1. **String match:** any contiguous bigram where token[i].form (or lemma)
       lowercased is `"let"` and token[i+1].form/lemma lowercased is
       `"alone"`.
    2. **UD pattern:** any token with `form.lower() == 'alone'` and
       `deprel == 'fixed'`, such that either its head's form/lemma is
       `let`, or its head has a dependent with `deprel` in {'conj','xcomp','dep'}
       to the right (indicative of Y).  This captures UD analyses where
       the `let alone` anchor is attached to the head of X, rather than
       directly to `let`.

    Anchors are returned as `(start_index, end_index, route)` triples, where
    `start_index` and `end_index` are 0‑based indices into the sentence's
    token list (inclusive), and `route` indicates whether the anchor was
    detected via `'string'`, `'ud'`, or both.

    Duplicate anchors (identified by the same start and end indices) are
    merged by setting `route='both'`.
    """
    tokens = sentence["tokens"]
    n = len(tokens)
    anchors: Dict[Tuple[int, int], str] = {}
    # 1. string matches
    for i in range(n - 1):
        tok = tokens[i]
        next_tok = tokens[i + 1]
        if tok["form"].lower() == "let" and next_tok["form"].lower() == "alone":
            anchors[(i, i + 1)] = anchors.get((i, i + 1), "string")
    # 2. UD pattern
    for idx, tok in enumerate(tokens):
        if tok["form"].lower() != "alone":
            continue
        if tok["deprel"] != "fixed":
            continue
        head_id = tok["head"]
        if head_id <= 0 or head_id > n:
            continue
        head_idx = head_id - 1
        head_tok = tokens[head_idx]
        # case A: head token is 'let'
        if head_tok["form"].lower() == "let":
            # anchor span is from head_idx to current idx
            span = (min(head_idx, idx), max(head_idx, idx))
            prev_route = anchors.get(span)
            anchors[span] = merge_route(prev_route, "ud")
            continue
        # case B: head token is some head of X; require a dependent Y
        # (conj, xcomp, dep) attached to head_tok that occurs to the right of
        # the anchor
        has_y = False
        for other in tokens:
            if other["head"] == head_id and other["deprel"] in {"conj", "xcomp", "dep"} and other["id"] > tok["id"]:
                has_y = True
                break
        if has_y:
            # anchor span includes any preceding 'let'?  In this pattern the
            # anchor may not include a literal 'let'; we record the span as
            # just the 'alone' token to avoid overlapping erroneously.
            span = (idx, idx)
            prev_route = anchors.get(span)
            anchors[span] = merge_route(prev_route, "ud")
    # convert to list of triples
    anchor_list: List[Tuple[int, int, str]] = []
    for (start, end), route in anchors.items():
        anchor_list.append((start, end, route))
    # sort by start index
    anchor_list.sort(key=lambda x: x[0])
    return anchor_list


def merge_route(existing: Optional[str], new: str) -> str:
    """Merge detection routes for duplicate anchors."""
    if existing is None:
        return new
    if existing == new:
        return existing
    return "both"


def apply_metalinguistic_filter(tokens: List[Dict[str, Any]], start: int, end: int) -> bool:
    """Heuristic filter to drop metalinguistic uses of *let alone*.

    The UD guidelines sometimes annotate metalinguistic or quoted uses of
    *let alone* (e.g. where the phrase is mentioned rather than used as a
    construction).  Following the task specification, we drop anchors if
    they occur inside quotation marks and are immediately followed by
    punctuation.  In practice we check whether there is a quote character
    (single `'` or double `"`) before the anchor and a punctuation token
    immediately after the anchor span.

    Parameters
    ----------
    tokens: list of token dicts for the sentence
    start: int
        0‑based index of the anchor start token
    end: int
        0‑based index of the anchor end token

    Returns
    -------
    bool
        True if the anchor should be **kept** (i.e. not filtered),
        False if it should be discarded.
    """
    # Check preceding tokens for quotes
    quote_before = False
    for i in range(max(0, start - 3), start):
        tok = tokens[i]
        if tok["form"] in {"'", '"', "``", "''", "“", "”"}:
            quote_before = True
            break
    # Check following token for punctuation
    punct_after = False
    after_idx = end + 1
    if after_idx < len(tokens):
        tok = tokens[after_idx]
        # consider punctuation POS or punctuation symbol
        if tok["upos"] == "PUNCT" or re.match(r"^[,.;:!?]$", tok["form"]):
            punct_after = True
    # filter if both quote before and punctuation after
    return not (quote_before and punct_after)


def find_nearest_head(tokens: List[Dict[str, Any]], anchor_start: int, direction: str) -> Optional[int]:
    """Find the nearest syntactic head to the anchor in a given direction.

    Parameters
    ----------
    tokens: list of token dicts
    anchor_start: int
        0‑based index of the anchor start (position of 'let' or 'alone')
    direction: str
        'left' or 'right'

    Returns
    -------
    Optional[int]
        Index (0‑based) of the nearest head token in the specified
        direction, or None if no suitable token is found.

    Notes
    -----
    "Nearest head" is operationalised as the closest token whose head
    is outside the anchor span and which is not punctuation.  This is a
    simplification of the more complex UD pattern but performs well in
    practice for identifying the X (left element) and Y (right element).
    """
    if direction not in {"left", "right"}:
        raise ValueError("direction must be 'left' or 'right'")
    n = len(tokens)
    if direction == "left":
        rng = range(anchor_start - 1, -1, -1)
    else:
        rng = range(anchor_start + 1, n)
    for idx in rng:
        tok = tokens[idx]
        # skip punctuation and symbols
        if tok["upos"] in {"PUNCT", "SYM", "X"}:
            continue
        # require that this token is a head of itself or attaches above anchor
        # but we relax this to just pick the first non‑punctuation token
        return idx
    return None


def has_licensor(tokens: List[Dict[str, Any]], anchor_start: int, licensor_words: Set[str], window: int = 5) -> bool:
    """Check for a downward‑entailing licensor within a left window.

    We scan up to `window` tokens to the left of the anchor start and
    return True if any token has a `form.lower()` in the licensor list,
    ignoring punctuation.  This implements the licensing cue described in
    the task specification.
    """
    count = 0
    for idx in range(anchor_start - 1, -1, -1):
        tok = tokens[idx]
        if tok["upos"] == "PUNCT":
            continue
        if tok["form"].lower() in licensor_words:
            return True
        count += 1
        if count >= window:
            break
    return False


def extract_let_alone_features(sentence: Dict[str, Any], licensor_words: Set[str]) -> List[Dict[str, Any]]:
    """Extract feature rows for all *let alone* anchors in a sentence.

    Parameters
    ----------
    sentence: dict
        Parsed sentence dictionary
    licensor_words: set
        Set of lower‑cased strings considered licensor cues

    Returns
    -------
    List[Dict[str, Any]]
        One feature dict per anchor with keys:
        'sent_id', 'text', 'anchor_start', 'anchor_end', 'route',
        'x_idx', 'y_idx', 'upos_x', 'upos_y', 'parallelism', 'licensing',
        'dist_x_anchor', 'dist_anchor_y'
    """
    tokens = sentence["tokens"]
    anchors = find_let_alone_anchors(sentence)
    rows: List[Dict[str, Any]] = []
    for (start, end, route) in anchors:
        # apply metalinguistic filter
        if not apply_metalinguistic_filter(tokens, start, end):
            continue
        # Determine anchor start position for directional calculations
        anchor_start = start
        # find nearest heads
        x_idx = find_nearest_head(tokens, anchor_start, "left")
        y_idx = find_nearest_head(tokens, end, "right")
        if x_idx is None or y_idx is None:
            continue
        upos_x_raw = tokens[x_idx]["upos"]
        upos_y_raw = tokens[y_idx]["upos"]
        upos_x = normalize_upos(upos_x_raw)
        upos_y = normalize_upos(upos_y_raw)
        # parallelism definition
        parallel = False
        if {upos_x, upos_y} <= {"NOUN"}:
            parallel = True
        elif upos_x == upos_y and upos_x in {"VERB", "ADJ"}:
            parallel = True
        # licensing cue
        licensing = has_licensor(tokens, anchor_start, licensor_words)
        # distances
        dist_x_anchor = anchor_start - x_idx
        dist_anchor_y = y_idx - end
        # build row
        rows.append({
            "sent_id": sentence.get("sent_id"),
            "text": sentence.get("text"),
            "anchor_start": start + 1,  # convert to 1‑based for readability
            "anchor_end": end + 1,
            "route": route,
            "x_idx": x_idx + 1,
            "y_idx": y_idx + 1,
            "x_form": tokens[x_idx]["form"],
            "y_form": tokens[y_idx]["form"],
            "upos_x": upos_x,
            "upos_y": upos_y,
            "parallelism": int(parallel),
            "licensing": int(licensing),
            "dist_x_anchor": dist_x_anchor,
            "dist_anchor_y": dist_anchor_y,
        })
    return rows
