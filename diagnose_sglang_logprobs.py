"""Diagnostic: Are SGLang constrained-generation logprobs pre- or post-mask?

HYPOTHESIS: When we force a single label via regex constraint, SGLang may
report POST-mask logprobs (i.e., after applying the regex mask to the logits).
This would make the forced token's logprob near 0.0 (probability ~1.0)
regardless of the model's true preference — explaining why classify() produces
near-uniform confidence (~25% across 4 labels).

If logprobs are PRE-mask, the forced token's logprob should reflect the model's
genuine P(token|context) and differ meaningfully across labels.

Run with::

    uv run python diagnose_sglang_logprobs.py
"""

import json
import urllib.request
import urllib.error

MODEL = "Qwen/Qwen2.5-3B-Instruct-AWQ"
BASE = "http://localhost:30000/v1"

SYSTEM = "You are a precise text classifier."
TEXT = "The team won the championship after a stunning last-minute goal!"
LABELS = ["sports", "finance", "science", "politics"]


def post(path: str, body: dict) -> dict:
    url = f"{BASE}{path}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode()[:300]}")
        return {}


def post_raw(path: str, body: dict) -> dict:
    """POST to a non-/v1 endpoint (e.g. /tokenize)."""
    url = f"http://localhost:30000{path}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode()[:300]}")
        return {}


def get_tokens_and_logprobs(resp: dict) -> list[tuple[str, float, list[tuple[str, float]]]]:
    """Extract (token, logprob, [(alt_token, alt_logprob), ...]) per position."""
    choice = resp["choices"][0]
    lp_content = (choice.get("logprobs") or {}).get("content") or []
    result = []
    for ti in lp_content:
        if not ti:
            continue
        token = ti.get("token", "")
        logprob = ti.get("logprob", 0.0)
        alts = [(a["token"], a["logprob"]) for a in (ti.get("top_logprobs") or [])]
        result.append((token, logprob, alts))
    return result


def main() -> None:
    print("=" * 70)
    print("DIAGNOSTIC: SGLang constrained logprobs — pre-mask vs post-mask?")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Text:  {TEXT}")
    print(f"Labels: {LABELS}")

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Classify this text:\n{TEXT}"},
    ]

    # =====================================================================
    # EXPERIMENT 1: Multi-label constrained — top_logprobs at position 0
    # =====================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Multi-label constrained generation")
    print("  regex = (sports|finance|science|politics)")
    print("  This shows the model's TRUE preference among all labels.")
    print("=" * 70)

    resp = post(
        "/chat/completions",
        {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 16,
            "regex": f"({'|'.join(LABELS)})",
            "logprobs": True,
            "top_logprobs": 10,
        },
    )
    if resp:
        tlp = get_tokens_and_logprobs(resp)
        content = resp["choices"][0]["message"]["content"]
        print(f"  Generated: {content!r}")
        for i, (tok, lp, alts) in enumerate(tlp[:3]):
            print(f"\n  Position {i}: token={tok!r}  logprob={lp:.6f}")
            print(f"    Top {min(len(alts), 8)} alternatives:")
            for at, alp in sorted(alts, key=lambda x: -x[1])[:8]:
                print(f"      {at!r:20s}  logprob={alp:.6f}  prob={pow(2.718281828, alp):.6f}")

    # =====================================================================
    # EXPERIMENT 2: Force each single label — compare logprobs
    # =====================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Single-label forced generation (score() approach)")
    print("  For each label, force regex=(label) and read back logprobs.")
    print("  If POST-mask: all labels get ~0.0 logprob (constraint forces ~100%).")
    print("  If PRE-mask:  logprobs reflect genuine model preference.")
    print("=" * 70)

    single_results = {}
    for label in LABELS:
        resp = post(
            "/chat/completions",
            {
                "model": MODEL,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 16,
                "regex": f"({label})",
                "logprobs": True,
                "top_logprobs": 5,
            },
        )
        if resp:
            tlp = get_tokens_and_logprobs(resp)
            # First token (label content), skip EOS
            content_tokens = [
                (t, lp, alts) for t, lp, alts in tlp
                if t not in ("<|im_end|>", "<|endoftext|>", "</s>")
            ]
            if content_tokens:
                first_tok, first_lp, first_alts = content_tokens[0]
                all_lps = [lp for _, lp, _ in content_tokens]
                gm = sum(all_lps) / len(all_lps) if all_lps else float("-inf")
                single_results[label] = {"first_lp": first_lp, "gm": gm, "n_tokens": len(content_tokens)}
                print(f"\n  force({label:12s}): first_token={first_tok!r:15s}  logprob={first_lp:.6f}  "
                      f"n_tokens={len(content_tokens)}  geomean={gm:.6f}")
                print(f"    alt top_logprobs: {[(t, round(a,4)) for t,a in first_alts[:5]]}")

    # =====================================================================
    # EXPERIMENT 3: Direct comparison — same first token across approaches
    # =====================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Cross-comparison of first-token logprobs")
    print("=" * 70)

    # From experiment 1, extract the first-position logprob for each label's
    # first token
    if resp:
        tlp = get_tokens_and_logprobs(resp)
        if tlp:
            pos0_token, pos0_lp, pos0_alts = tlp[0]
            alt_dict = dict(pos0_alts)
            print(f"\n  Multi-label position 0 (winning token={pos0_token!r}):")
            print(f"  The model's TRUE per-first-token logprobs (from top_logprobs):")
            for label in LABELS:
                # Get the first character/token of this label
                label_first_char = label[0]
                # Search alts for any token starting with this label's content
                found = None
                for at, alp in pos0_alts:
                    if label.startswith(at) or at.startswith(label_first_char):
                        found = (at, alp)
                        break
                multi_lp = found[1] if found else None
                single_lp = single_results.get(label, {}).get("first_lp")
                print(f"    {label:12s}: multi_label_top_lp={multi_lp!s:20s}  "
                      f"forced_single_lp={single_lp}")

    # =====================================================================
    # EXPERIMENT 4: Completions endpoint with echo — CLEAN label extraction
    # =====================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: /v1/completions with echo — prefill logprobs (no mask)")
    print("  Prompt ends with the label. /tokenize (correct 'prompt' field)")
    print("  pinpoints the exact label-token boundary. max_tokens=1 generated")
    print("  token is discarded by slicing to total_len only.")
    print("=" * 70)

    def tokenize_count(text: str) -> int:
        """Tokenize via the /tokenize endpoint with correct field name."""
        resp = post_raw("/tokenize", {"model": MODEL, "prompt": text})
        return resp.get("count", 0) if resp else 0

    echo_results = {}
    for label in LABELS:
        prompt_without = (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\nClassify this text:\n{TEXT}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        prompt_with_label = prompt_without + label

        prompt_len = tokenize_count(prompt_without)
        total_len = tokenize_count(prompt_with_label)
        label_n = total_len - prompt_len

        resp = post(
            "/completions",
            {
                "model": MODEL,
                "prompt": prompt_with_label,
                "echo": True,
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": 1,
            },
        )
        if resp:
            choice = resp["choices"][0]
            lp_data = choice.get("logprobs") or {}
            tokens = lp_data.get("tokens", [])
            token_lps = lp_data.get("token_logprobs", [])

            # Extract ONLY the label tokens (positions prompt_len..total_len-1)
            label_tokens_info = list(
                zip(
                    tokens[prompt_len:total_len],
                    token_lps[prompt_len:total_len],
                )
            )
            label_lps = [
                lp for _, lp in label_tokens_info if lp is not None
            ]
            gm = sum(label_lps) / len(label_lps) if label_lps else float("-inf")
            echo_results[label] = {"gm": gm, "label_tokens": [t for t, _ in label_tokens_info], "label_n": label_n}
            print(f"  echo({label:12s}): prompt_len={prompt_len} total_len={total_len} label_n={label_n}")
            print(f"    label_tokens={[t for t,_ in label_tokens_info]}")
            print(f"    logprobs  ={[round(lp,4) if lp else None for _,lp in label_tokens_info]}")
            print(f"    geomean   ={gm:.6f}")
            # Show the spurious generated token for reference
            if len(tokens) > total_len:
                print(f"    [discarded gen token: {tokens[total_len]!r} lp={token_lps[total_len]!s}]")

    # =====================================================================
    # EXPERIMENT 5: Regex vs no-regex on the SAME completion
    # =====================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Regex vs no-regex — same prompt, compare logprob of same token")
    print("=" * 70)

    # Force "sports" with regex, then check: is the logprob of 's' different
    # from what we'd see in a multi-label regex where 'sports' is also valid?
    print("\n  Single-label regex '(sports)' position 0 logprob:")
    if "sports" in single_results:
        sl = single_results["sports"]["first_lp"]
        print(f"    forced '(sports)' -> {sl:.6f}  (prob={pow(2.718281828, sl):.6f})")

    # Now from the multi-label run, find 'sports'-prefix token in top_logprobs
    print("\n  Multi-label regex '(sports|...)' — 's' or 'sports' in top_logprobs:")
    resp_multi = post(
        "/chat/completions",
        {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 2,
            "regex": f"({'|'.join(LABELS)})",
            "logprobs": True,
            "top_logprobs": 20,
        },
    )
    if resp_multi:
        tlp = get_tokens_and_logprobs(resp_multi)
        if tlp:
            pos0_token, pos0_lp, pos0_alts = tlp[0]
            print(f"    winning token: {pos0_token!r} (logprob={pos0_lp:.6f})")
            print(f"    ALL top_logprobs at position 0:")
            for at, alp in sorted(pos0_alts, key=lambda x: -x[1]):
                marker = " <-- SPORTS" if "sport" in at.lower() or at == "s" else ""
                print(f"      {at!r:20s}  logprob={alp:.6f}  prob={pow(2.718281828, alp):.6f}{marker}")

    # =====================================================================
    # EXPERIMENT 6: JSON Schema enum constraint (alternative to regex)
    # =====================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: JSON Schema enum constraint (response_format)")
    print("  Multi-label: all labels in the enum, compare with regex approach.")
    print("  Single-label: force each label, read back logprobs.")
    print("=" * 70)

    def build_json_enum(labels_subset: list[str]) -> dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "label",
                "schema": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": labels_subset}
                    },
                    "required": ["label"],
                },
            },
        }

    # 6a: Multi-label JSON Schema
    print("\n  --- 6a: Multi-label JSON Schema enum ---")
    resp_js_multi = post(
        "/chat/completions",
        {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 16,
            "response_format": build_json_enum(LABELS),
            "logprobs": True,
            "top_logprobs": 10,
        },
    )
    js_multi_tlp = []
    if resp_js_multi:
        js_multi_tlp = get_tokens_and_logprobs(resp_js_multi)
        content = resp_js_multi["choices"][0]["message"]["content"]
        print(f"  Generated: {content!r}")
        for i, (tok, lp, alts) in enumerate(js_multi_tlp[:8]):
            print(f"\n  Position {i}: token={tok!r}  logprob={lp:.6f}")
            for at, alp in sorted(alts, key=lambda x: -x[1])[:5]:
                marker = ""
                for lbl in LABELS:
                    if lbl in at.lower() or at.lower() in lbl:
                        marker = f"  <-- {lbl.upper()}"
                        break
                print(f"      {at!r:20s}  logprob={alp:.6f}  prob={pow(2.718281828, alp):.6f}{marker}")

    # 6b: Single-label JSON Schema (force each label)
    print("\n\n  --- 6b: Single-label JSON Schema enum (force each label) ---")
    js_single_results = {}
    for label in LABELS:
        resp_js = post(
            "/chat/completions",
            {
                "model": MODEL,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 16,
                "response_format": build_json_enum([label]),
                "logprobs": True,
                "top_logprobs": 5,
            },
        )
        if resp_js:
            tlp = get_tokens_and_logprobs(resp_js)
            # Extract label-value tokens (skip JSON structure + EOS)
            full_str = "".join(t for t, _, _ in tlp)
            label_tokens = []
            try:
                idx = full_str.index(label)
                pos = 0
                for tok, lp, alts in tlp:
                    tok_end = pos + len(tok)
                    if tok_end > idx and pos < idx + len(label):
                        label_tokens.append((tok, lp, alts))
                    pos = tok_end
            except ValueError:
                label_tokens = [(t, lp, alts) for t, lp, alts in tlp if t.strip() and t.strip() not in ('{','}','"',':','<|im_end|>','<|endoftext|>')]
            if label_tokens:
                all_lps = [lp for _, lp, _ in label_tokens]
                gm = sum(all_lps) / len(all_lps) if all_lps else float("-inf")
                js_single_results[label] = {"first_lp": label_tokens[0][1], "gm": gm, "n_tokens": len(label_tokens)}
                print(f"\n  force_js({label:12s}): tokens={[t for t,_,_ in label_tokens]}  "
                      f"first_lp={label_tokens[0][1]:.6f}  geomean={gm:.6f}")
                first_alts = label_tokens[0][2][:5]
                print(f"    alt top_logprobs (pos 0): {[(t, round(a,4)) for t,a in first_alts]}")

    # =====================================================================
    # EXPERIMENT 7: Unconstrained generation — natural model output
    # =====================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Unconstrained generation (no regex/JSON)")
    print("  No constraint — model generates freely. top_k=20 per position.")
    print("  This shows the model's genuine next-token distribution at each step.")
    print("=" * 70)

    resp_unc = post(
        "/chat/completions",
        {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 8,
            "logprobs": True,
            "top_logprobs": 20,
        },
    )
    unc_tlp = []
    if resp_unc:
        unc_tlp = get_tokens_and_logprobs(resp_unc)
        content = resp_unc["choices"][0]["message"]["content"]
        print(f"\n  Generated: {content!r}")
        for i, (tok, lp, alts) in enumerate(unc_tlp):
            print(f"\n  Position {i}: token={tok!r}  logprob={lp:.6f}")
            print(f"    Top {min(len(alts), 20)} alternatives:")
            for at, alp in sorted(alts, key=lambda x: -x[1])[:20]:
                marker = ""
                for lbl in LABELS:
                    if at.strip().lower().startswith(lbl[:3]) or lbl.startswith(at.strip().lower()[:3]):
                        if len(at.strip()) >= 2 or len(lbl[:3]) == 3:
                            marker = f"  <-- {lbl.upper()}"
                            break
                print(f"      {at!r:20s}  logprob={alp:.6f}  prob={pow(2.718281828, alp):.6f}{marker}")

    # =====================================================================
    # EXPERIMENT 8: Three-way comparison — echo vs forced-single vs multi-label
    # =====================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: Three-way comparison of scoring approaches")
    print("  A) Echo/prefill: /v1/completions echo=True (no constraint)")
    print("  B) Forced-single: regex=(label) forced generation")
    print("  C) Multi-label: regex=(l1|l2|...) winner's top_logprobs")
    print("=" * 70)

    import math

    def softmax(logprobs: dict[str, float]) -> dict[str, float]:
        valid = {k: v for k, v in logprobs.items() if v > float("-inf")}
        if not valid:
            return {k: 1.0 / len(logprobs) for k in logprobs}
        mx = max(valid.values())
        exps = {k: math.exp(v - mx) if v > float("-inf") else 0.0 for k, v in logprobs.items()}
        total = sum(exps.values())
        return {k: v / total for k, v in exps.items()} if total > 0 else {k: 1.0 / len(logprobs) for k in logprobs}

    # --- Approach A: Echo geomean logprobs ---
    echo_scores = {lbl: echo_results.get(lbl, {}).get("gm", float("-inf")) for lbl in LABELS}
    echo_probs = softmax(echo_scores)

    # --- Approach B: Forced-single geomean logprobs ---
    forced_scores = {lbl: single_results.get(lbl, {}).get("gm", float("-inf")) for lbl in LABELS}
    forced_probs = softmax(forced_scores)

    # --- Approach C: Multi-label first-token logprobs from top_logprobs ---
    # Re-run multi-label with top_logprobs=20 to get all label-prefixed tokens
    resp_c = post(
        "/chat/completions",
        {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 2,
            "regex": f"({'|'.join(LABELS)})",
            "logprobs": True,
            "top_logprobs": 20,
        },
    )
    multi_scores = {lbl: float("-inf") for lbl in LABELS}
    if resp_c:
        tlp_c = get_tokens_and_logprobs(resp_c)
        if tlp_c:
            pos0_alts = tlp_c[0][2]  # (token, logprob) pairs at position 0
            alt_dict = dict(pos0_alts)
            for label in LABELS:
                best = float("-inf")
                for tok, lp in alt_dict.items():
                    if label.startswith(tok) or tok.startswith(label):
                        best = max(best, lp)
                multi_scores[label] = best
    multi_probs = softmax(multi_scores)

    # --- Print comparison table ---
    print(f"\n  {'Label':<14s} | {'Echo (prefill)':>20s} | {'Forced-single':>20s} | {'Multi-label':>20s}")
    print(f"  {'':14s} | {'geomean -> prob':>20s} | {'geomean -> prob':>20s} | {'first_lp -> prob':>20s}")
    print(f"  {'-'*14}-+-{'-'*20}-+-{'-'*20}-+-{'-'*20}")
    for lbl in LABELS:
        e_lp = echo_scores[lbl]
        f_lp = forced_scores[lbl]
        m_lp = multi_scores[lbl]
        print(f"  {lbl:<14s} | {e_lp:>9.6f} -> {echo_probs[lbl]:.4f}  | {f_lp:>9.6f} -> {forced_probs[lbl]:.4f}  | {m_lp:>9.6f} -> {multi_probs[lbl]:.4f}")

    # --- Entropy comparison ---
    def entropy(probs: dict[str, float]) -> float:
        return -sum(p * math.log(p) for p in probs.values() if p > 0)

    max_entropy = math.log(len(LABELS))
    print(f"\n  Entropy (max={max_entropy:.4f} = uniform):")
    print(f"    Echo (prefill):   {entropy(echo_probs):.4f}  (confidence spread: {max(echo_probs.values()) - min(echo_probs.values()):.4f})")
    print(f"    Forced-single:    {entropy(forced_probs):.4f}  (confidence spread: {max(forced_probs.values()) - min(forced_probs.values()):.4f})")
    print(f"    Multi-label:      {entropy(multi_probs):.4f}  (confidence spread: {max(multi_probs.values()) - min(multi_probs.values()):.4f})")

    print(f"\n  Verdict:")
    best_echo = max(echo_probs, key=echo_probs.get)
    best_forced = max(forced_probs, key=forced_probs.get)
    best_multi = max(multi_probs, key=multi_probs.get)
    print(f"    Echo predicts:     {best_echo} ({echo_probs[best_echo]:.2%})")
    print(f"    Forced predicts:   {best_forced} ({forced_probs[best_forced]:.2%})")
    print(f"    Multi predicts:    {best_multi} ({multi_probs[best_multi]:.2%})")

    # =====================================================================
    # CONCLUSION
    # =====================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    if single_results:
        lps = [v["first_lp"] for v in single_results.values()]
        spread = max(lps) - min(lps)
        print(f"\n  Regex forced single-label first-token logprob spread: {spread:.6f}")
        print(f"  Values: {dict((k, round(v['first_lp'], 6)) for k,v in single_results.items())}")
        if spread < 0.01:
            print("\n  >>> POST-MASK CONFIRMED: All forced labels get near-identical")
            print("  >>> near-zero logprobs. The regex constraint masks logits before")
            print("  >>> reporting, so the model's true preference is hidden.")
        else:
            print("\n  >>> PRE-MASK (or partially masked): logprobs show differentiation.")
            print("  >>> The uniform confidence has a different cause.")

    if js_single_results:
        print("\n  --- JSON Schema vs Regex comparison ---")
        for label in LABELS:
            r_lp = single_results.get(label, {}).get("first_lp")
            j_lp = js_single_results.get(label, {}).get("first_lp")
            r_gm = single_results.get(label, {}).get("gm")
            j_gm = js_single_results.get(label, {}).get("gm")
            print(f"    {label:12s}: regex_first_lp={r_lp!s:20s} json_first_lp={j_lp!s:20s}")
            print(f"                  regex_geomean={r_gm!s:20s} json_geomean={j_gm!s:20s}")

    if unc_tlp:
        print("\n  --- Unconstrained: does the model output a label naturally? ---")
        content = resp_unc["choices"][0]["message"]["content"]
        print(f"    Free generation: {content!r}")
        matched = [lbl for lbl in LABELS if lbl.lower() in content.lower()]
        print(f"    Label match: {matched if matched else 'NONE (model did not output a clean label)'}")
        print(f"    Position 0 top-5 (unconstrained):")
        if unc_tlp:
            for at, alp in sorted(unc_tlp[0][2], key=lambda x: -x[1])[:5]:
                print(f"      {at!r:20s}  logprob={alp:.6f}")


if __name__ == "__main__":
    main()
