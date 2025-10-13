"""
snorkel_lfs_eval_q2.py

- Loads HuggingFace conll2003
- Implements two Snorkel labeling functions:
    a) lf_years: detects years 1900-2099 -> returns MISC
    b) lf_org_suffix: pattern matches organization suffixes -> returns ORG
- Computes coverage and accuracy per-LF and logs them to W&B via wandb.log().
"""

import os
import re
from collections import Counter
from typing import List

from datasets import load_dataset
import wandb

# Optional: use snorkel decorator if present
try:
    from snorkel.labeling import labeling_function, ABSTAIN
    SNORKEL_AVAILABLE = True
except Exception:
    # If snorkel is not installed, just use a fallback ABSTAIN constant and callables
    SNORKEL_AVAILABLE = False
    ABSTAIN = -1

# Label mapping (integers for snorkel-style labels)
LABEL_O = 0
LABEL_PER = 1
LABEL_LOC = 2
LABEL_ORG = 3
LABEL_MISC = 4
LABEL_NAMES = {0: "O", 1: "PER", 2: "LOC", 3: "ORG", 4: "MISC"}


# 🔐 Insert your Weights & Biases API Key here:
WANDB_API_KEY = ""
def wandb_login():
    """Login to W&B using provided API key."""
    if WANDB_API_KEY == "your_key_here":
        raise ValueError("Please replace WANDB_API_KEY with your actual W&B key.")
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    wandb.login(key=WANDB_API_KEY)


def coarse_label_from_conll_tag(tag: str) -> int:
    """
    Map a tag like 'B-PER', 'I-ORG', 'O' -> coarse label integer.
    """
    if tag == "O":
        return LABEL_O
    parts = tag.split("-", 1)
    if len(parts) == 2:
        ent = parts[1]
    else:
        ent = tag
    if ent == "PER":
        return LABEL_PER
    if ent == "LOC":
        return LABEL_LOC
    if ent == "ORG":
        return LABEL_ORG
    # anything else -> MISC
    return LABEL_MISC


def build_token_level_dataset(max_samples=None):
    """
    Flatten sentence examples into token-level records.
    Returns list of dicts: {"token": str, "gold": int}
    max_samples: optional limit on number of sentences to process (for speed).
    """
    ds = load_dataset("eriktks/conll2003", revision="convert/parquet")
    records = []
    # use train+validation+test or just train? we'll use all splits for evaluation
    for split_name in ds.keys():
        for i, ex in enumerate(ds[split_name]):
            if max_samples is not None and i >= max_samples:
                break
            tokens = ex["tokens"]
            tags = ex["ner_tags"]  # integers -> label names from dataset features
            label_names = ds[split_name].features["ner_tags"].feature.names
            # convert tag ids to label names then to coarse
            for tag_id, token in zip(tags, tokens):
                tag_name = label_names[tag_id]
                coarse = coarse_label_from_conll_tag(tag_name)
                records.append({"token": token, "gold": coarse, "split": split_name})
    return records


# -------------------------
# Labeling functions
# -------------------------

YEAR_REGEX = re.compile(r"^(19\d{2}|20\d{2})$")  # matches 1900-2099 strictly
# To avoid matching things like '2020,' or '(1999)' we strip punctuation boundaries when checking.

ORG_SUFFIXES = [r"\bInc\.?$", r"\bCorp\.?$", r"\bLtd\.?$", r"\bLLC\.?$", r"\bPLC\.?$"]
ORG_SUFFIX_REGEX = re.compile(r"(?i)(" + r"|".join(s[:-1] if s.endswith("$") else s for s in ORG_SUFFIXES) + r")\.?$")

# Snorkel decorator (if available) expects a function taking a single example arg.
if SNORKEL_AVAILABLE:
    @labeling_function()
    def lf_years(x):
        token = x["token"]
        t = token.strip().strip(".,;:()[]\"'")  # basic punctuation strip
        if YEAR_REGEX.match(t):
            return LABEL_MISC  # DATE/MISC as requested
        return ABSTAIN

    @labeling_function()
    def lf_org_suffix(x):
        token = x["token"]
        # check token endswith org suffix (case-insensitive)
        t = token.strip()
        # match whole token suffixes like 'Inc.' or 'Corp' etc.
        if re.search(r"(Inc\.?|Corp\.?|Ltd\.?|LLC\.?|PLC\.?)$", t, flags=re.IGNORECASE):
            return LABEL_ORG
        return ABSTAIN
else:
    # Fallback plain functions returning ints or ABSTAIN
    def lf_years(x):
        token = x["token"]
        t = token.strip().strip(".,;:()[]\"'")
        if YEAR_REGEX.match(t):
            return LABEL_MISC
        return ABSTAIN

    def lf_org_suffix(x):
        token = x["token"]
        t = token.strip()
        if re.search(r"(Inc\.?|Corp\.?|Ltd\.?|LLC\.?|PLC\.?)$", t, flags=re.IGNORECASE):
            return LABEL_ORG
        return ABSTAIN


# -------------------------
# Evaluation helpers
# -------------------------

def evaluate_labeling_function(records: List[dict], lf_callable, lf_name: str):
    """
    Apply lf_callable to every token record.
    Compute coverage = fraction of tokens where LF != ABSTAIN.
    Compute accuracy = fraction of labeled tokens where LF_label == gold label.
    Returns dict of metrics.
    """
    n = len(records)
    labeled = 0
    correct = 0
    for rec in records:
        lab = lf_callable(rec)
        if lab != ABSTAIN:
            labeled += 1
            if lab == rec["gold"]:
                correct += 1
    coverage = labeled / n if n > 0 else 0.0
    accuracy = (correct / labeled) if labeled > 0 else 0.0
    return {
        "lf_name": lf_name,
        "n_tokens": n,
        "n_labeled": int(labeled),
        "coverage": float(coverage),
        "accuracy": float(accuracy),
    }


def main():
    wandb_login()
    # Build token-level dataset (use all sentences; if large, you can pass max_samples to limit)
    print("Building token-level dataset (this may take a moment)...")
    records = build_token_level_dataset(max_samples=None)  # None -> use all
    print(f"Total tokens: {len(records)}")

    # Evaluate both LFs
    metrics_years = evaluate_labeling_function(records, lf_years, "lf_years")
    metrics_org = evaluate_labeling_function(records, lf_org_suffix, "lf_org_suffix")

    # Print results locally
    print("\nLabeling function metrics:")
    for m in (metrics_years, metrics_org):
        print(f"- {m['lf_name']}: labeled {m['n_labeled']}/{m['n_tokens']} "
              f"({m['coverage']:.4f}), accuracy={m['accuracy']:.4f}")

    # Log to W&B
    run = wandb.init(
        project="Q1-weak-supervision-ner",
        job_type="lf-eval",
        reinit=True,
        config={"note": "LF coverage and accuracy for Q2"}
    )
    # Log each LF metrics
    wandb.log({
        "lf_years/coverage": metrics_years["coverage"],
        "lf_years/accuracy": metrics_years["accuracy"],
        "lf_years/n_labeled": metrics_years["n_labeled"],
        "lf_years/n_tokens": metrics_years["n_tokens"],

        "lf_org_suffix/coverage": metrics_org["coverage"],
        "lf_org_suffix/accuracy": metrics_org["accuracy"],
        "lf_org_suffix/n_labeled": metrics_org["n_labeled"],
        "lf_org_suffix/n_tokens": metrics_org["n_tokens"],
    })

    # Also store the raw metrics in run.summary for quick top-level view
    wandb.run.summary.update({
        "lf_years/coverage": metrics_years["coverage"],
        "lf_years/accuracy": metrics_years["accuracy"],
        "lf_org_suffix/coverage": metrics_org["coverage"],
        "lf_org_suffix/accuracy": metrics_org["accuracy"],
    })

    print("\nLogged LF metrics to W&B. Run URL:", wandb.run.get_url() if wandb.run else "(no run url)")
    run.finish()


if __name__ == "__main__":
    main()
