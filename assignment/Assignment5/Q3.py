"""
majority_label_voter_q3.py

Implements Snorkel-style majority label aggregation (MajorityLabelVoter).
Uses two LFs:
 - lf_years -> LABEL_MISC for 1900-2099 tokens
 - lf_org_suffix -> LABEL_ORG for tokens ending with org suffixes

If snorkel is installed, uses snorkel.labeling.majority_label_model.MajorityLabelVoter (or the
LabelModel alternative). Otherwise uses local majority-vote implementation.

Logs coverage/accuracy to W&B using wandb.log().
"""

import os
import re
import numpy as np
from collections import Counter, defaultdict

from datasets import load_dataset
import wandb

# Try to import snorkel's MajorityLabelVoter if available
try:
    from snorkel.labeling import MajorityLabelVoter, ABSTAIN
    SNORKEL_AVAILABLE = True
except Exception:
    SNORKEL_AVAILABLE = False
    ABSTAIN = -1  # we'll use -1 for abstain

# Label mapping (same as Q2)
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

# ----- Build token-level records (flatten dataset) -----
def coarse_label_from_conll_tag(tag: str) -> int:
    if tag == "O":
        return LABEL_O
    parts = tag.split("-", 1)
    ent = parts[1] if len(parts) == 2 else tag
    if ent == "PER":
        return LABEL_PER
    if ent == "LOC":
        return LABEL_LOC
    if ent == "ORG":
        return LABEL_ORG
    return LABEL_MISC


def build_token_records(max_sentences_per_split=None):
    ds = load_dataset("eriktks/conll2003", revision="convert/parquet")
    records = []
    for split in ds.keys():
        for i, ex in enumerate(ds[split]):
            if max_sentences_per_split is not None and i >= max_sentences_per_split:
                break
            tokens = ex["tokens"]
            tags = ex["ner_tags"]
            label_names = ds[split].features["ner_tags"].feature.names
            for token, tag_id in zip(tokens, tags):
                tag_name = label_names[tag_id]
                gold = coarse_label_from_conll_tag(tag_name)
                records.append({"token": token, "gold": gold, "split": split})
    return records


# ----- Labeling functions (same as Q2) -----
YEAR_REGEX = re.compile(r"^(19\d{2}|20\d{2})$")
def lf_years(rec):
    t = rec["token"].strip().strip(".,;:()[]\"'")
    if YEAR_REGEX.match(t):
        return LABEL_MISC
    return ABSTAIN

def lf_org_suffix(rec):
    t = rec["token"].strip()
    if re.search(r"(Inc\.?|Corp\.?|Ltd\.?|LLC\.?|PLC\.?)$", t, flags=re.IGNORECASE):
        return LABEL_ORG
    return ABSTAIN

LABELING_FUNCTIONS = [lf_years, lf_org_suffix]


# ----- Build label matrix L (n_tokens x n_lfs) -----
def build_label_matrix(records, lfs):
    n = len(records)
    m = len(lfs)
    L = np.full((n, m), ABSTAIN, dtype=int)
    for i, rec in enumerate(records):
        for j, lf in enumerate(lfs):
            try:
                L[i, j] = lf(rec)
            except Exception:
                L[i, j] = ABSTAIN
    return L


# ----- Majority voting aggregator (fallback) -----
def majority_vote_row(row):
    """
    row: 1D array of LF outputs for a single token (values are label ints or ABSTAIN)
    Return: aggregated label int or ABSTAIN
    Tie-breaking strategy: if tie between two or more labels for max count, return ABSTAIN.
    """
    counts = Counter([int(x) for x in row if int(x) != ABSTAIN])
    if not counts:
        return ABSTAIN
    most_common = counts.most_common()
    if len(most_common) == 0:
        return ABSTAIN
    # Check tie: compare top two counts
    top_label, top_count = most_common[0]
    if len(most_common) > 1 and most_common[1][1] == top_count:
        # tie -> abstain
        return ABSTAIN
    return top_label


def aggregate_majority_local(L):
    # Apply majority_vote_row for each row
    n = L.shape[0]
    aggregated = np.full(n, ABSTAIN, dtype=int)
    for i in range(n):
        aggregated[i] = majority_vote_row(L[i, :])
    return aggregated


# ----- Evaluation -----
def evaluate_aggregated_labels(aggr_labels, gold_labels):
    n = len(gold_labels)
    assert len(aggr_labels) == n
    labeled_mask = (aggr_labels != ABSTAIN)
    n_labeled = int(np.sum(labeled_mask))
    coverage = n_labeled / n if n > 0 else 0.0
    correct = int(np.sum((aggr_labels == gold_labels) & labeled_mask))
    accuracy = (correct / n_labeled) if n_labeled > 0 else 0.0
    return {"n_tokens": n, "n_labeled": n_labeled, "coverage": coverage, "accuracy": accuracy}


# ----- Main flow -----
def main():
    wandb_login()
    print("Building token-level records (this may take a moment)...")
    records = build_token_records(max_sentences_per_split=None)  # use all
    n_tokens = len(records)
    print(f"Total tokens: {n_tokens}")

    # Build label matrix L
    L = build_label_matrix(records, LABELING_FUNCTIONS)
    print("Label matrix shape:", L.shape)

    # If snorkel is available, use its MajorityLabelVoter
    if SNORKEL_AVAILABLE:
        print("Using Snorkel's MajorityLabelVoter for aggregation...")
        mv = MajorityLabelVoter()
        # Snorkel expects L as numpy array with shape (n, m)
        aggregated = mv.predict(L)  # returns a 1D array of aggregated labels (or ABSTAIN)
    else:
        print("Snorkel not available — using local majority-vote aggregator.")
        aggregated = aggregate_majority_local(L)

    # Evaluate aggregated labels vs gold
    gold = np.array([rec["gold"] for rec in records], dtype=int)
    metrics = evaluate_aggregated_labels(aggregated, gold)

    print("\nMajority-voter metrics:")
    print(f"  tokens labeled: {metrics['n_labeled']}/{metrics['n_tokens']}")
    print(f"  coverage: {metrics['coverage']:.6f}")
    print(f"  accuracy (on labeled tokens): {metrics['accuracy']:.6f}")

    # Log to W&B
    run = wandb.init(
        project="Q1-weak-supervision-ner",
        job_type="majority-aggregation",
        reinit=True,
        config={"n_lfs": len(LABELING_FUNCTIONS)}
    )

    wandb.log({
        "majority/n_tokens": metrics["n_tokens"],
        "majority/n_labeled": metrics["n_labeled"],
        "majority/coverage": metrics["coverage"],
        "majority/accuracy": metrics["accuracy"],
    })
    # Also store a small confusion-like breakdown: counts for aggregated labels vs gold (only labeled tokens)
    aggr_counts = Counter(int(x) for x in aggregated if int(x) != ABSTAIN)
    wandb.log({"majority/aggregated_label_counts": dict(aggr_counts)})

    # Optionally: log a tiny table of examples where aggregator labeled (token, aggr, gold)
    # We'll log up to 100 examples to avoid huge uploads
    table_rows = []
    max_examples = 100
    added = 0
    for i, rec in enumerate(records):
        if added >= max_examples:
            break
        if aggregated[i] != ABSTAIN:
            table_rows.append([rec["token"], int(aggregated[i]), LABEL_NAMES.get(int(aggregated[i]), str(aggregated[i])), rec["gold"], LABEL_NAMES.get(rec["gold"], str(rec["gold"])), rec["split"]])
            added += 1
    # create a W&B table if there are rows
    if table_rows:
        tb = wandb.Table(columns=["token", "aggr_label_id", "aggr_label_name", "gold_id", "gold_name", "split"], data=table_rows)
        wandb.log({"majority/labeled_examples_table": tb})

    wandb.run.summary.update({
        "majority/coverage": metrics["coverage"],
        "majority/accuracy": metrics["accuracy"],
    })

    run.finish()
    print("\nLogged majority aggregation metrics to W&B:", wandb.run.get_url() if wandb.run else "(no run url)")


if __name__ == "__main__":
    main()
