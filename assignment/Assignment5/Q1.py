"""
Script: log_conll2003_stats_to_wandb.py
Purpose:
  Load the HuggingFace CoNLL-2003 dataset (eriktks/conll2003),
  compute dataset statistics (samples + entity distribution),
  and log them to Weights & Biases project "Q1-weak-supervision-ner".
"""

# =============================
# Install dependencies first
# pip install datasets wandb
# =============================

import os
import json
import tempfile
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import load_dataset
import wandb


# 🔐 Insert your Weights & Biases API Key here:
WANDB_API_KEY = ""
def wandb_login():
    """Login to W&B using provided API key."""
    if WANDB_API_KEY == "your_key_here":
        raise ValueError("Please replace WANDB_API_KEY with your actual W&B key.")
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    wandb.login(key=WANDB_API_KEY)





def compute_entity_counts(split_dataset):
    """
    Given a HF dataset split with 'ner_tags', return:
      - samples: number of examples
      - tokens: total tokens
      - entity_counts_by_tag: Counter of tag-name -> token count (e.g., 'B-PER', 'I-PER', 'O', ...)
    """
    features = split_dataset.features
    if "ner_tags" not in features:
        raise ValueError("Split does not contain 'ner_tags' feature.")
    label_names = features["ner_tags"].feature.names  # maps id -> tag name

    entity_counts = Counter()
    total_tokens = 0
    for example in split_dataset:
        tags = example["ner_tags"]
        total_tokens += len(tags)
        for tag_id in tags:
            entity_counts[label_names[tag_id]] += 1

    return {
        "samples": len(split_dataset),
        "tokens": total_tokens,
        "entity_counts_by_tag": entity_counts,
    }


def aggregate_coarse(entity_counts_by_tag):
    """
    Convert B-/I- style tags into coarse entity categories: PER, LOC, ORG, MISC, and O.
    Returns Counter with keys: PER, LOC, ORG, MISC, O
    """
    coarse = Counter()
    for tag, count in entity_counts_by_tag.items():
        if tag == "O":
            coarse["O"] += count
        else:
            # tag like "B-PER" or "I-LOC"
            parts = tag.split("-", 1)
            ent = parts[-1] if len(parts) > 1 else tag
            if ent in {"PER", "LOC", "ORG", "MISC"}:
                coarse[ent] += count
            else:
                # map unknown to MISC
                coarse["MISC"] += count
    return coarse


def make_bar_chart(coarse_counts, title="Entity counts (coarse)"):
    """Return a matplotlib figure for the coarse entity counts (PER/LOC/ORG/MISC)."""
    ents = ["PER", "LOC", "ORG", "MISC"]
    counts = [coarse_counts.get(e, 0) for e in ents]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ents, counts)
    ax.set_xlabel("Entity type")
    ax.set_ylabel("Token count")
    ax.set_title(title)
    for i, v in enumerate(counts):
        ax.text(i, v + max(counts) * 0.01, str(v), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    return fig


def main():
    # 0) Setup W&B login (if API key present in env)
    wandb_login()

    # 1) Load official conll2003 dataset
    print("Loading dataset 'conll2003' from Hugging Face...")
    dataset = load_dataset("eriktks/conll2003", revision="convert/parquet")
    print("Loaded splits:", list(dataset.keys()))

    # 2) Compute per-split stats
    split_stats = {}
    overall_entity_by_tag = Counter()
    total_samples = 0
    total_tokens = 0

    for split_name, split_data in dataset.items():
        print(f"Computing stats for split: {split_name} ...")
        stats = compute_entity_counts(split_data)
        split_stats[split_name] = stats
        overall_entity_by_tag.update(stats["entity_counts_by_tag"])
        total_samples += stats["samples"]
        total_tokens += stats["tokens"]

    # 3) Aggregate coarse counts (overall) and per-split coarse counts
    coarse_overall = aggregate_coarse(overall_entity_by_tag)
    coarse_per_split = {
        s: aggregate_coarse(split_stats[s]["entity_counts_by_tag"]) for s in split_stats
    }

    # 4) Prepare numeric summary dict
    summary = {
        "dataset": "conll2003",
        "total_samples": total_samples,
        "total_tokens": total_tokens,
    }
    for split in ["train", "validation", "test"]:
        summary[f"{split}_samples"] = split_stats.get(split, {}).get("samples", 0)
        summary[f"{split}_tokens"] = split_stats.get(split, {}).get("tokens", 0)

    # add coarse counts and fractions overall
    for ent in ["PER", "LOC", "ORG", "MISC", "O"]:
        c = int(coarse_overall.get(ent, 0))
        summary[f"entity_count_{ent}"] = c
        summary[f"entity_frac_{ent}"] = c / total_tokens if total_tokens > 0 else 0.0

    # add per-split coarse counts into summary
    for split, coarse in coarse_per_split.items():
        for ent in ["PER", "LOC", "ORG", "MISC", "O"]:
            summary[f"{split}_entity_count_{ent}"] = int(coarse.get(ent, 0))

    # 5) Initialize W&B run (use finish_previous=True to avoid deprecation warning)
    run = wandb.init(
        project="Q1-weak-supervision-ner",
        job_type="dataset-stats",
        reinit=True,
        config={"dataset_name": "conll2003"},
    )

    # 6) Update run summary with the numeric summary
    wandb.run.summary.update(summary)

    # 7) Log entity distribution dict (coarse overall) and per-split distributions
    wandb.log({"entity_distribution_overall": dict(coarse_overall)})
    for split, coarse in coarse_per_split.items():
        wandb.log({f"entity_distribution_{split}": dict(coarse)})

    # 8) Create and log bar chart (matplotlib figure) for overall coarse counts
    fig = make_bar_chart(coarse_overall, title="Coarse entity counts (overall)")
    wandb.log({"entity_counts_bar": wandb.Image(fig)})
    plt.close(fig)

    # 9) Save summary JSON and log as artifact
    with tempfile.TemporaryDirectory() as td:
        summary_path = Path(td) / "conll2003_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        artifact = wandb.Artifact("conll2003-dataset-stats", type="dataset")
        artifact.add_file(str(summary_path), name="conll2003_summary.json")
        run.log_artifact(artifact)
        print("Saved and logged artifact:", artifact.name)

    # 10) print summary to console and finish run
    print("\nSummary (sample):")
    keys_to_show = [
        "total_samples",
        "total_tokens",
        "entity_count_PER",
        "entity_count_LOC",
        "entity_count_ORG",
        "entity_count_MISC",
        "entity_count_O",
    ]
    for k in keys_to_show:
        print(f"  {k}: {summary.get(k)}")

    run.finish()
    print("\nW&B run finished. View the run at:", wandb.run.get_url() if wandb.run else "(no run url)")


if __name__ == "__main__":
    main()
