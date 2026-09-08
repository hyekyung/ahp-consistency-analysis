"""
AHP Consistency Checker
========================

Checks a raw AHP pairwise-comparison dataset for internal inconsistency.

For each respondent and each category, it:
  1. Rebuilds the pairwise comparison matrix from the raw judgments.
  2. Computes the Consistency Ratio (CR) using Saaty's eigenvalue method.
     CR <= 0.10 is the standard threshold for a reliable judgment set.
  3. Detects explicit intransitive triads (A over B, B over C, C over A).
  4. Reports results per participant, per role, and per category.

Requires: numpy (pip install numpy)

INPUT FILE FORMAT
------------------
A semicolon-delimited CSV with one row per pairwise comparison, columns:
  participant, profile, category, credit_A, credit_B, more_important,
  saaty_intensity, intensity_label

  - "more_important" is either "equal", or matches credit_A or credit_B.
  - "saaty_intensity" is the Saaty scale value (1 to 9).

Update INPUT_FILE below to point at your dataset, then run:
  python3 ahp_consistency_check.py
"""

import csv
import itertools
from collections import defaultdict
from math import comb

import numpy as np

# ---------------------------------------------------------------------------
# CONFIG - edit these as needed
# ---------------------------------------------------------------------------
INPUT_FILE = "Exercise_3_raw_dataset.csv"
DELIMITER = ";"
CR_THRESHOLD = 0.10
OUTPUT_FILE = "ahp_consistency_results.csv"

# Saaty's Random Index (RI), standard published values for matrix size n
RANDOM_INDEX = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
    7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48,
    13: 1.56, 14: 1.57, 15: 1.59,
}


def load_rows(path, delimiter):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return list(reader)


def group_by_participant_category(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["participant"], r["category"])].append(r)
    return groups


def build_matrix(records):
    """
    Build the reciprocal pairwise comparison matrix for one respondent's
    judgments in one category. Returns (items, matrix, preference_edges,
    all_equal_flag).
    """
    items = sorted(set(
        [r["credit_A"] for r in records] + [r["credit_B"] for r in records]
    ))
    idx = {item: i for i, item in enumerate(items)}
    n = len(items)
    matrix = np.ones((n, n))
    preference_edges = []  # list of (winner, loser) for strict preferences
    all_equal = True

    for r in records:
        i, j = idx[r["credit_A"]], idx[r["credit_B"]]
        intensity = float(r["saaty_intensity"])
        winner = r["more_important"]

        if winner == "equal" or intensity == 1:
            matrix[i, j] = 1
            matrix[j, i] = 1
        else:
            all_equal = False
            if winner == r["credit_A"]:
                matrix[i, j] = intensity
                matrix[j, i] = 1 / intensity
                preference_edges.append((r["credit_A"], r["credit_B"]))
            else:
                matrix[j, i] = intensity
                matrix[i, j] = 1 / intensity
                preference_edges.append((r["credit_B"], r["credit_A"]))

    return items, matrix, preference_edges, all_equal


def consistency_ratio(matrix):
    """Saaty's eigenvalue method: CR = CI / RI."""
    n = matrix.shape[0]
    eigenvalues = np.linalg.eigvals(matrix)
    lambda_max = max(eigenvalues.real)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = RANDOM_INDEX.get(n, 1.6)  # fallback estimate for n > 15
    return ci / ri if ri > 0 else 0.0


def find_intransitive_triads(items, preference_edges):
    """
    Returns a list of (a, b, c) triads where a beats b, b beats c,
    and c beats a (a genuine three-way contradiction).
    """
    beats = defaultdict(set)
    for winner, loser in preference_edges:
        beats[winner].add(loser)

    cycles = []
    for combo in itertools.combinations(items, 3):
        found = False
        for a, b, c in itertools.permutations(combo):
            if b in beats[a] and c in beats[b] and a in beats[c]:
                cycles.append(tuple(sorted(combo)))
                found = True
                break
        if found:
            continue
    return cycles


def analyze(rows):
    groups = group_by_participant_category(rows)
    results = []

    for (participant, category), records in groups.items():
        profile = records[0]["profile"]
        items, matrix, edges, all_equal = build_matrix(records)
        n = len(items)
        total_triads = comb(n, 3)

        if all_equal:
            results.append({
                "participant": participant, "profile": profile,
                "category": category, "n_items": n, "status": "all-equal",
                "CR": None, "n_cycles": None, "total_triads": total_triads,
            })
            continue

        cr = consistency_ratio(matrix)
        cycles = find_intransitive_triads(items, edges)
        status = "flagged" if cr > CR_THRESHOLD else "consistent"

        results.append({
            "participant": participant, "profile": profile,
            "category": category, "n_items": n, "status": status,
            "CR": round(cr, 4), "n_cycles": len(cycles),
            "total_triads": total_triads,
        })

    return results


def print_report(results):
    results = sorted(results, key=lambda r: (r["category"], r["participant"]))

    print("\n=== FULL RESULTS, BY CATEGORY ===")
    current_cat = None
    for r in results:
        if r["category"] != current_cat:
            current_cat = r["category"]
            print(f"\n--- {current_cat} ---")
        if r["status"] == "all-equal":
            print(f"  {r['participant']:<6} {r['profile']:<24} "
                  f"n={r['n_items']:<3} ALL-EQUAL (excluded)")
        else:
            flag = "FLAGGED" if r["status"] == "flagged" else "ok"
            print(f"  {r['participant']:<6} {r['profile']:<24} "
                  f"n={r['n_items']:<3} CR={r['CR']:.3f} [{flag}] "
                  f"cycles={r['n_cycles']}/{r['total_triads']}")

    # Summary by role
    print("\n=== SUMMARY BY ROLE ===")
    by_role = defaultdict(lambda: {"diff": 0, "flagged": 0, "crs": []})
    for r in results:
        if r["status"] == "all-equal":
            continue
        s = by_role[r["profile"]]
        s["diff"] += 1
        s["crs"].append(r["CR"])
        if r["status"] == "flagged":
            s["flagged"] += 1
    for role, s in by_role.items():
        rate = 100 * s["flagged"] / s["diff"] if s["diff"] else 0
        avg_cr = sum(s["crs"]) / len(s["crs"]) if s["crs"] else 0
        print(f"  {role:<24} differentiated={s['diff']:<4} "
              f"flagged={s['flagged']:<4} rate={rate:.1f}% avg_CR={avg_cr:.3f}")

    # Summary by category
    print("\n=== SUMMARY BY CATEGORY ===")
    by_cat = defaultdict(lambda: {"diff": 0, "flagged": 0, "crs": [], "cyc": 0})
    for r in results:
        if r["status"] == "all-equal":
            continue
        s = by_cat[r["category"]]
        s["diff"] += 1
        s["crs"].append(r["CR"])
        if r["status"] == "flagged":
            s["flagged"] += 1
        if r["n_cycles"] > 0:
            s["cyc"] += 1
    for cat, s in by_cat.items():
        rate = 100 * s["flagged"] / s["diff"] if s["diff"] else 0
        avg_cr = sum(s["crs"]) / len(s["crs"]) if s["crs"] else 0
        print(f"  {cat:<14} differentiated={s['diff']:<4} "
              f"flagged={s['flagged']:<4} rate={rate:.1f}% "
              f"avg_CR={avg_cr:.3f} with_cycles={s['cyc']}")


def write_csv(results, path):
    fieldnames = ["participant", "profile", "category", "n_items",
                  "status", "CR", "n_cycles", "total_triads"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    rows = load_rows(INPUT_FILE, DELIMITER)
    results = analyze(rows)
    print_report(results)
    write_csv(results, OUTPUT_FILE)
    print(f"\nFull results written to {OUTPUT_FILE}")
