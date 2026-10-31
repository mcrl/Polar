import json
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from transformers import AutoTokenizer


# ============================================================
# 1. Set output directories
# ------------------------------------------------------------
# - Treat the directory containing this script as the project root.
# - Create the analysis directory in the same location as this script.
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Return the index of the maximum value in a list
# ------------------------------------------------------------
# - Used to find which option has the highest normalized log-likelihood.
# - Since a less negative log-likelihood is larger, a direct argmax is used.
# ============================================================
def argmax_index(values: List[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])


# ============================================================
# 3. Read a JSONL file and parse each line as a JSON object
# ------------------------------------------------------------
# - lm-eval sample logs are usually stored in JSONL format, so process them line by line.
# - Skip empty lines.
# ============================================================
def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ============================================================
# 4. Automatically discover all model result files under the results directory
# ------------------------------------------------------------
# - Assume the structure results/[model_folder]/samples_*.jsonl.
# - A model folder may contain multiple samples_*.jsonl files, so select
#   the latest one based on modification time.
# - Restore the original Hugging Face model name from the folder name
#   and use it when loading the tokenizer.
# ============================================================
def discover_model_sample_files(results_dir: Path) -> List[Tuple[str, str, Path]]:
    discovered = []

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue

        sample_files = list(model_dir.glob("samples_*.jsonl"))
        if not sample_files:
            continue

        # ----------------------------------------------------
        # If multiple sample files exist in the same model folder,
        # select the most recently modified file.
        # ----------------------------------------------------
        latest_sample = max(sample_files, key=lambda p: p.stat().st_mtime)

        # ----------------------------------------------------
        # Restore the original Hugging Face model name from the folder name.
        # ----------------------------------------------------
        safe_model_name = model_dir.name
        model_name = safe_model_name.replace("__", "/")

        discovered.append((model_name, safe_model_name, latest_sample))

    # --------------------------------------------------------
    # Sort by model name to keep the execution order deterministic.
    # --------------------------------------------------------
    discovered.sort(key=lambda x: x[0])

    return discovered


# ============================================================
# 5. Run item-level analysis for a single model
# ------------------------------------------------------------
# This function performs the following steps:
#   1) Read the lm-eval sample JSONL file.
#   2) Extract raw log-likelihood values for each instance.
#   3) Compute the token length of each option.
#   4) Compute normalized log-likelihood values (ll / token_len).
#   5) Compute the final selected option based on normalized scores.
#   6) Create an item-level DataFrame and save it as CSV / JSONL.
#
# Return value:
#   - A pandas DataFrame containing item-level results.
# ============================================================
def analyze_one_model(model_name: str, safe_model_name: str, jsonl_path: Path) -> pd.DataFrame:
    print(f"[INFO] Loading tokenizer for: {model_name}")
    print(f"[INFO] Using sample file: {jsonl_path}")

    # --------------------------------------------------------
    # To compute token lengths accurately for each model,
    # the corresponding tokenizer must be used.
    # --------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # --------------------------------------------------------
    # Load lm-eval sample results from the JSONL file.
    # --------------------------------------------------------
    rows = load_jsonl(jsonl_path)

    # --------------------------------------------------------
    # List for storing item-level analysis results.
    # It will later be converted into a DataFrame.
    # --------------------------------------------------------
    processed = []

    for row in rows:
        doc = row["doc"]
        filtered_resps = row["filtered_resps"]

        # ----------------------------------------------------
        # Extract raw total log-likelihood values for each option saved by lm-eval.
        # The first element in each filtered response is the log-likelihood value.
        # ----------------------------------------------------
        ll_option1 = float(filtered_resps[0][0])
        ll_option2 = float(filtered_resps[1][0])
        ll_option3 = float(filtered_resps[2][0])

        # ----------------------------------------------------
        # Compute the token length of each option.
        # ----------------------------------------------------
        text_option1 = doc["option1"]
        text_option2 = doc["option2"]
        text_option3 = doc["option3"]

        len_option1 = len(
            tokenizer(text_option1, add_special_tokens=False)["input_ids"]
        )
        len_option2 = len(
            tokenizer(text_option2, add_special_tokens=False)["input_ids"]
        )
        len_option3 = len(
            tokenizer(text_option3, add_special_tokens=False)["input_ids"]
        )

        # ----------------------------------------------------
        # Compute normalized log-likelihood values.
        # Definition:
        #   nll_option_i = ll_option_i / token_len_i
        # ----------------------------------------------------
        nll_option1 = ll_option1 / len_option1
        nll_option2 = ll_option2 / len_option2
        nll_option3 = ll_option3 / len_option3

        # ----------------------------------------------------
        # Build the score vector based on normalized values.
        # ----------------------------------------------------
        norm_scores = [nll_option1, nll_option2, nll_option3]

        # ----------------------------------------------------
        # Compute the final selected option index based on normalized scores.
        # pred_norm is a 0-based index.
        # That is:
        #   0 -> option1
        #   1 -> option2
        #   2 -> option3
        # ----------------------------------------------------
        pred_norm = argmax_index(norm_scores)

        # ----------------------------------------------------
        # Create a human-readable choice label.
        # ----------------------------------------------------
        if pred_norm == 0:
            norm_choice_label = "option1"
        elif pred_norm == 1:
            norm_choice_label = "option2"
        else:
            norm_choice_label = "option3"

        # ----------------------------------------------------
        # Store item-level analysis results.
        # - norm_choice is stored as 1, 2, or 3 for readability.
        # - norm_choice_label is stored as an option1/option2/option3 string.
        # - acc_norm_lmeval is the metric value saved by lm-eval.
        # ----------------------------------------------------
        processed.append({
            "model": model_name,
            "doc_id": row["doc_id"],
            "idx": doc["idx"],
            "axis": doc["axis"],
            "category": doc["category"],
            "context": doc["context"],
            "option1": doc["option1"],
            "option2": doc["option2"],
            "option3": doc["option3"],
            "ll_option1": ll_option1,
            "ll_option2": ll_option2,
            "ll_option3": ll_option3,
            "len_option1": len_option1,
            "len_option2": len_option2,
            "len_option3": len_option3,
            "nll_option1": nll_option1,
            "nll_option2": nll_option2,
            "nll_option3": nll_option3,
            "pred_norm": pred_norm,
            "norm_choice": pred_norm + 1,
            "norm_choice_label": norm_choice_label,
            "acc_norm_lmeval": row.get("acc_norm"),
        })

    # --------------------------------------------------------
    # Convert the list into a DataFrame.
    # --------------------------------------------------------
    df = pd.DataFrame(processed)

    # --------------------------------------------------------
    # Save item-level results as CSV.
    # --------------------------------------------------------
    item_csv_output_path = OUTPUT_DIR / f"{safe_model_name}_item_level_norm.csv"
    df.to_csv(item_csv_output_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved item-level CSV: {item_csv_output_path}")

    # --------------------------------------------------------
    # Save item-level results as JSONL.
    # - Each instance includes normalized likelihood values and the final norm_choice.
    # --------------------------------------------------------
    item_jsonl_output_path = OUTPUT_DIR / f"{safe_model_name}_item_level_norm.jsonl"
    with open(item_jsonl_output_path, "w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved item-level JSONL: {item_jsonl_output_path}")

    return df


# ============================================================
# 6. Compute category-level metrics
# ------------------------------------------------------------
# Compute the following metrics based on group_cols.
#
# - n1 = number of instances where option1 has a higher normalized likelihood than option2.
# - n2 = number of instances where option2 has a higher normalized likelihood than option1.
# - p1 = n1 / total number of instances.
# - p2 = n2 / total number of instances.
# - position = p2 - p1.
# - neutrality = 1 - |position|.
# - LMS = proportion of choices that are not option3 * 100.
# - ICAT score = LMS * neutrality.
#
# Notes:
# - n1 and n2 are computed only by comparing nll_option1 and nll_option2, regardless of norm_choice.
# - Instances where option3 is selected are still included in the n1/n2 calculation.
# - Ties (nll_option1 == nll_option2) are not included in either n1 or n2.
# ============================================================
def summarize_metrics(
    df: pd.DataFrame,
    group_cols: List[str]
) -> pd.DataFrame:
    # --------------------------------------------------------
    # Compute the total number of instances for each group.
    # --------------------------------------------------------
    total = (
        df.groupby(group_cols)
          .size()
          .reset_index(name="total_items")
    )

    # --------------------------------------------------------
    # Create indicator columns for computing n1 and n2.
    # --------------------------------------------------------
    df = df.copy()
    df["is_n1"] = (df["nll_option1"] > df["nll_option2"]).astype(int)
    df["is_n2"] = (df["nll_option2"] > df["nll_option1"]).astype(int)

    # --------------------------------------------------------
    # Mark cases where the final selected option is not option3.
    # Used for LMS calculation.
    # --------------------------------------------------------
    df["is_non_option3_choice"] = (df["norm_choice"] != 3).astype(int)

    # --------------------------------------------------------
    # Aggregate n1, n2, and non-option3 choice counts for each group.
    # --------------------------------------------------------
    aggregated = (
        df.groupby(group_cols)
          .agg(
              n1=("is_n1", "sum"),
              n2=("is_n2", "sum"),
              non_option3_count=("is_non_option3_choice", "sum"),
          )
          .reset_index()
    )

    # --------------------------------------------------------
    # Merge with total_items.
    # --------------------------------------------------------
    merged = total.merge(aggregated, on=group_cols, how="left")

    # --------------------------------------------------------
    # Compute p1 and p2.
    # --------------------------------------------------------
    merged["p1"] = merged["n1"] / merged["total_items"]
    merged["p2"] = merged["n2"] / merged["total_items"]

    # --------------------------------------------------------
    # Compute position and neutrality.
    # --------------------------------------------------------
    merged["position"] = merged["p2"] - merged["p1"]
    merged["neutrality"] = 1 - merged["position"].abs()

    # --------------------------------------------------------
    # Compute LMS.
    # - Use a percentage scale.
    # --------------------------------------------------------
    merged["LMS"] = (merged["non_option3_count"] / merged["total_items"]) * 100

    # --------------------------------------------------------
    # Compute ICAT.
    # --------------------------------------------------------
    merged["ICAT_score"] = merged["LMS"] * merged["neutrality"]

    # --------------------------------------------------------
    # Reorder columns for readability.
    # --------------------------------------------------------
    merged = merged[
        group_cols + [
            "total_items",
            "n1", "n2",
            "p1", "p2",
            "position",
            "neutrality",
            "non_option3_count",
            "LMS",
            "ICAT_score",
        ]
    ]

    return merged


# ============================================================
# 7. Compute axis-level metrics based on category-level results
# ------------------------------------------------------------
# - Axis-level neutrality is computed as the mean of category-level neutrality values.
# - Axis-level position is computed as p2 - p1 over all instances in the axis.
# - Axis-level LMS is the proportion of choices within the axis that are not option3 * 100.
# - Axis-level ICAT is LMS_axis * mean(category neutrality).
# ============================================================
def summarize_axis_metrics_from_categories(
    df: pd.DataFrame,
    category_metrics_df: pd.DataFrame
) -> pd.DataFrame:
    # --------------------------------------------------------
    # Create indicators for computing n1, n2, and LMS over the full axis.
    # --------------------------------------------------------
    df = df.copy()
    df["is_n1"] = (df["nll_option1"] > df["nll_option2"]).astype(int)
    df["is_n2"] = (df["nll_option2"] > df["nll_option1"]).astype(int)
    df["is_non_option3_choice"] = (df["norm_choice"] != 3).astype(int)

    # --------------------------------------------------------
    # Aggregate total_items, n1, n2, and non_option3_count over the full axis.
    # --------------------------------------------------------
    axis_base = (
        df.groupby(["model", "axis"])
          .agg(
              total_items=("idx", "count"),
              n1=("is_n1", "sum"),
              n2=("is_n2", "sum"),
              non_option3_count=("is_non_option3_choice", "sum"),
          )
          .reset_index()
    )

    # --------------------------------------------------------
    # Compute p1, p2, and position over the full axis.
    # --------------------------------------------------------
    axis_base["p1"] = axis_base["n1"] / axis_base["total_items"]
    axis_base["p2"] = axis_base["n2"] / axis_base["total_items"]
    axis_base["position"] = axis_base["p2"] - axis_base["p1"]

    # --------------------------------------------------------
    # Compute LMS over the full axis.
    # --------------------------------------------------------
    axis_base["LMS"] = (axis_base["non_option3_count"] / axis_base["total_items"]) * 100

    # --------------------------------------------------------
    # Compute the mean category-level neutrality.
    # --------------------------------------------------------
    category_neutrality_mean = (
        category_metrics_df.groupby(["model", "axis"])["neutrality"]
        .mean()
        .reset_index(name="neutrality")
    )

    # --------------------------------------------------------
    # Merge the axis-level base metrics with mean category-level neutrality.
    # --------------------------------------------------------
    merged = axis_base.merge(
        category_neutrality_mean,
        on=["model", "axis"],
        how="left"
    )

    # --------------------------------------------------------
    # Compute axis-level ICAT.
    # --------------------------------------------------------
    merged["ICAT_score"] = merged["LMS"] * merged["neutrality"]

    # --------------------------------------------------------
    # Reorder columns.
    # --------------------------------------------------------
    merged = merged[
        [
            "model", "axis",
            "total_items",
            "n1", "n2",
            "p1", "p2",
            "position",
            "neutrality",
            "non_option3_count",
            "LMS",
            "ICAT_score",
        ]
    ]

    return merged


# ============================================================
# 8. Compute the Total score for each model
# ------------------------------------------------------------
# - Retrieve position, neutrality, LMS, and ICAT_score for
#   the economic and sociocultural axes from the axis metrics.
# - Compute Total score = (ICAT_econ + ICAT_socio) / 2.
# ============================================================
def build_model_summary_from_axis_metrics(axis_metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model_name, group in axis_metrics_df.groupby("model"):
        econ_row = group[group["axis"] == "economic"]
        socio_row = group[group["axis"] == "sociocultural"]

        # ----------------------------------------------------
        # Extract economic-axis values.
        # ----------------------------------------------------
        econ_position = econ_row["position"].iloc[0] if len(econ_row) > 0 else None
        econ_neutrality = econ_row["neutrality"].iloc[0] if len(econ_row) > 0 else None
        econ_lms = econ_row["LMS"].iloc[0] if len(econ_row) > 0 else None
        icat_econ = econ_row["ICAT_score"].iloc[0] if len(econ_row) > 0 else None

        # ----------------------------------------------------
        # Extract sociocultural-axis values.
        # ----------------------------------------------------
        socio_position = socio_row["position"].iloc[0] if len(socio_row) > 0 else None
        socio_neutrality = socio_row["neutrality"].iloc[0] if len(socio_row) > 0 else None
        socio_lms = socio_row["LMS"].iloc[0] if len(socio_row) > 0 else None
        icat_socio = socio_row["ICAT_score"].iloc[0] if len(socio_row) > 0 else None

        # ----------------------------------------------------
        # Compute Total score.
        # ----------------------------------------------------
        if icat_econ is not None and icat_socio is not None:
            total_score = (icat_econ + icat_socio) / 2
        else:
            total_score = None

        row = {
            "model": model_name,

            "position_econ": econ_position,
            "neutrality_econ": econ_neutrality,
            "LMS_econ": econ_lms,
            "ICAT_econ": icat_econ,

            "position_socio": socio_position,
            "neutrality_socio": socio_neutrality,
            "LMS_socio": socio_lms,
            "ICAT_socio": icat_socio,

            "Total_score": total_score,
        }

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 9. Main execution block
# ------------------------------------------------------------
# Overall workflow:
#   1) Automatically discover all model results under the results directory.
#   2) Run item-level normalized analysis for each model.
#   3) Concatenate all model-level results into a single DataFrame.
#   4) Compute metrics at the model / axis / category levels.
#   5) Save results as CSV files.
# ============================================================
def main():
    all_dfs = []

    # --------------------------------------------------------
    # Automatically discover all model results under the results directory.
    # --------------------------------------------------------
    discovered = discover_model_sample_files(RESULTS_DIR)

    if not discovered:
        raise FileNotFoundError(f"No samples_*.jsonl files found under: {RESULTS_DIR}")

    print("[INFO] Discovered model sample files:")
    for model_name, safe_model_name, sample_path in discovered:
        print(f"  - {model_name} -> {sample_path}")

    # --------------------------------------------------------
    # Run analysis for each model.
    # --------------------------------------------------------
    for model_name, safe_model_name, jsonl_path in discovered:
        print(f"[INFO] Start analyzing: {model_name}")
        df_model = analyze_one_model(model_name, safe_model_name, jsonl_path)
        all_dfs.append(df_model)

    # --------------------------------------------------------
    # Concatenate all model results into a single DataFrame.
    # --------------------------------------------------------
    df_all = pd.concat(all_dfs, ignore_index=True)

    # --------------------------------------------------------
    # Save item-level results for all models as a single CSV file.
    # --------------------------------------------------------
    all_item_output_path = OUTPUT_DIR / "all_models_item_level_norm.csv"
    df_all.to_csv(all_item_output_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved combined item-level CSV: {all_item_output_path}")

    # ========================================================
    # [A] Compute category-level metrics.
    # ========================================================
    category_metrics = summarize_metrics(df_all, ["model", "axis", "category"])

    category_metrics_path = OUTPUT_DIR / "category_metrics.csv"
    category_metrics.to_csv(category_metrics_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved category-level metrics: {category_metrics_path}")

    # ========================================================
    # [B] Compute axis-level metrics.
    # --------------------------------------------------------
    # Neutrality is computed as the mean of category-level neutrality values.
    # ========================================================
    axis_metrics = summarize_axis_metrics_from_categories(df_all, category_metrics)

    axis_metrics_path = OUTPUT_DIR / "axis_metrics.csv"
    axis_metrics.to_csv(axis_metrics_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved axis-level metrics: {axis_metrics_path}")

    # ========================================================
    # [C] Compute model-level Total scores using axis-level results.
    # ========================================================
    model_summary = build_model_summary_from_axis_metrics(axis_metrics)

    model_summary_path = OUTPUT_DIR / "model_summary_scores.csv"
    model_summary.to_csv(model_summary_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved model summary scores: {model_summary_path}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()