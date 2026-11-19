import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT_FILE = Path("examples/my_eval/outputs/zh_style_compare.jsonl")
DEFAULT_SUMMARY_FILE = Path("examples/my_eval/outputs/zh_style_summary.json")
DEFAULT_REVIEW_FILE = Path("examples/my_eval/outputs/zh_style_manual_review.csv")

EXPLANATION_MARKERS = [
    "原句",
    "修改",
    "建议",
    "说明",
    "原因",
    "可以改",
    "可以写",
    "推荐",
    "如下",
    "版本",
    "解释",
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def has_explanation(output: str) -> bool:
    return any(marker in output for marker in EXPLANATION_MARKERS) or "\n" in output


def score_row(row: dict) -> dict:
    output = row["output"].strip()
    reference = row["reference"].strip()
    required = row.get("must_contain", [])
    contains_required = all(item in output for item in required)
    explanation = has_explanation(output)

    return {
        **row,
        "output_chars": len(output),
        "exact_match": output == reference,
        "contains_required": contains_required,
        "has_explanation": explanation,
        "format_ok": bool(output) and not explanation and len(output) <= 80,
        "changed_input": output != row["input"].strip(),
    }


def summarize(scored_rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in scored_rows:
        grouped[row["model"]].append(row)

    summary = {}
    for model, rows in grouped.items():
        count = len(rows)
        summary[model] = {
            "count": count,
            "avg_output_chars": round(sum(row["output_chars"] for row in rows) / count, 2),
            "exact_match_rate": round(sum(row["exact_match"] for row in rows) / count, 4),
            "contains_required_rate": round(sum(row["contains_required"] for row in rows) / count, 4),
            "no_explanation_rate": round(sum(not row["has_explanation"] for row in rows) / count, 4),
            "format_ok_rate": round(sum(row["format_ok"] for row in rows) / count, 4),
            "changed_input_rate": round(sum(row["changed_input"] for row in rows) / count, 4),
        }

    return summary


def write_manual_review(scored_rows: list[dict], output_path: Path) -> None:
    ids = []
    by_id = defaultdict(dict)
    meta = {}
    for row in scored_rows:
        sample_id = row["id"]
        if sample_id not in meta:
            ids.append(sample_id)
            meta[sample_id] = row
        by_id[sample_id][row["model"]] = row["output"]

    model_names = sorted({row["model"] for row in scored_rows})
    columns = ["id", "category", "instruction", "input", "reference", *model_names, "winner", "reason"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for sample_id in ids:
            row = meta[sample_id]
            review_row = {
                "id": sample_id,
                "category": row["category"],
                "instruction": row["instruction"],
                "input": row["input"],
                "reference": row["reference"],
                "winner": "",
                "reason": "",
            }
            for model_name in model_names:
                review_row[model_name] = by_id[sample_id].get(model_name, "")

            writer.writerow(review_row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY_FILE)
    parser.add_argument("--review-file", type=Path, default=DEFAULT_REVIEW_FILE)
    args = parser.parse_args()

    rows = read_jsonl(args.input_file)
    scored_rows = [score_row(row) for row in rows]
    summary = summarize(scored_rows)

    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_manual_review(scored_rows, args.review_file)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved summary to: {args.summary_file}")
    print(f"saved manual review table to: {args.review_file}")


if __name__ == "__main__":
    main()
