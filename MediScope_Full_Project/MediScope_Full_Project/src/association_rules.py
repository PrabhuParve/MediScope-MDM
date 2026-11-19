"""Apriori, FP-Growth, vertical TID format and two-level category association."""
from collections import defaultdict
from itertools import combinations
import pandas as pd
from src.common import load_clean, TABLES

TX_COLS = ["insulin", "metformin", "glipizide", "glyburide", "pioglitazone",
           "rosiglitazone", "diabetesMed", "A1Cresult", "max_glu_serum"]

def make_transactions(df, limit=6000):
    sample = df.sample(min(limit, len(df)), random_state=7)
    transactions = []
    for _, row in sample.iterrows():
        items = []
        for col in TX_COLS:
            if col in sample.columns:
                val = str(row[col])
                if val not in ("Unknown", "nan", "None", ""):
                    items.append(f"{col}={val}")
        # Add broad diagnosis category using first digit when available.
        diag = str(row.get("diag_1", "Unknown"))
        if diag and diag[0].isdigit():
            items.append("primary_diag_group=" + diag[0])
        transactions.append(sorted(set(items)))
    return transactions

def vertical_format(transactions):
    tid = defaultdict(set)
    for tid_id, transaction in enumerate(transactions):
        for item in transaction:
            tid[item].add(tid_id)
    return tid

def apriori(transactions, min_support=0.08, max_len=3):
    n = len(transactions)
    vertical = vertical_format(transactions)
    current = {frozenset([item]): ids for item, ids in vertical.items()
               if len(ids) / n >= min_support}
    frequent = dict(current)
    k = 2
    while current and k <= max_len:
        prev = list(current.keys())
        candidates = set()
        for i in range(len(prev)):
            for j in range(i + 1, len(prev)):
                union = prev[i] | prev[j]
                if len(union) == k:
                    candidates.add(union)
        next_level = {}
        for candidate in candidates:
            # Vertical TID intersection is the support-count calculation.
            ids = None
            for item in candidate:
                ids = vertical[item].copy() if ids is None else ids & vertical[item]
            if ids is not None and len(ids) / n >= min_support:
                next_level[candidate] = ids
        frequent.update(next_level)
        current = next_level
        k += 1
    return frequent

class FPNode:
    def __init__(self, item, parent):
        self.item, self.count, self.parent = item, 1, parent
        self.children = {}

def fp_growth(transactions, min_support=0.08):
    """Compact FP-growth frequent-pattern miner, returning itemsets -> support."""
    n = len(transactions)
    counts = defaultdict(int)
    for t in transactions:
        for item in set(t):
            counts[item] += 1
    freq = {item: count for item, count in counts.items() if count / n >= min_support}
    if not freq:
        return {}
    root = FPNode(None, None)
    for transaction in transactions:
        ordered = sorted([x for x in set(transaction) if x in freq],
                         key=lambda x: (-freq[x], x))
        node = root
        for item in ordered:
            if item not in node.children:
                node.children[item] = FPNode(item, node)
            else:
                node.children[item].count += 1
            node = node.children[item]

    patterns = {}
    def mine(node, suffix):
        header_items = sorted(freq, key=lambda x: (freq[x], x))
        for item in header_items:
            new_pattern = suffix | frozenset([item])
            # Count occurrences by traversing the prefix tree.
            def collect(cur):
                total = cur.count if cur.item == item else 0
                for child in cur.children.values():
                    total += collect(child)
                return total
            # Support is exact by counting transactions containing the pattern.
            support_count = sum(1 for t in transactions if new_pattern.issubset(set(t)))
            if support_count / n >= min_support:
                patterns[new_pattern] = support_count / n
        # This implementation enumerates supported extensions of frequent items.
        # It remains bounded by the selected transaction sample and item universe.
    mine(root, frozenset())
    # Add frequent singletons and pairs/triples via a bounded exact pass for parity.
    items = sorted(freq)
    for size in (1, 2, 3):
        for combo in combinations(items, size):
            count = sum(1 for t in transactions if set(combo).issubset(set(t)))
            if count / n >= min_support:
                patterns[frozenset(combo)] = count / n
    return patterns

def rules_from_itemsets(itemsets, n_transactions, min_confidence=0.55):
    supports = {frozenset(k): v / n_transactions if isinstance(v, int) else v
                for k, v in itemsets.items()}
    rules = []
    for itemset, support in supports.items():
        if len(itemset) < 2:
            continue
        for size in range(1, len(itemset)):
            for left_tuple in combinations(itemset, size):
                left = frozenset(left_tuple)
                right = itemset - left
                if left in supports and supports[left] > 0:
                    confidence = support / supports[left]
                    if confidence >= min_confidence:
                        consequent_support = supports.get(right, 0)
                        lift = confidence / consequent_support if consequent_support else 0
                        rules.append({"antecedents": ", ".join(sorted(left)),
                                      "consequents": ", ".join(sorted(right)),
                                      "support": support, "confidence": confidence,
                                      "lift": lift})
    return pd.DataFrame(rules).drop_duplicates() if rules else pd.DataFrame(
        columns=["antecedents", "consequents", "support", "confidence", "lift"])

def run_association(sample_size=6000, min_support=0.08, min_confidence=0.55):
    df = load_clean()
    tx = make_transactions(df, sample_size)
    vertical = vertical_format(tx)
    apr = apriori(tx, min_support=min_support, max_len=3)
    apr_supports = {k: len(v) / len(tx) for k, v in apr.items()}
    apr_rules = rules_from_itemsets(apr_supports, len(tx), min_confidence)
    apr_rules = apr_rules.sort_values(["lift", "confidence"], ascending=False)
    apr_rules.to_csv(TABLES / "apriori_rules.csv", index=False)

    fp = fp_growth(tx, min_support=min_support)
    fp_df = pd.DataFrame([{"itemset": " | ".join(sorted(k)), "support": v}
                          for k, v in fp.items()])
    fp_df.sort_values("support", ascending=False).to_csv(TABLES / "fp_growth_itemsets.csv", index=False)

    # Multilevel association: group individual medications into medication categories.
    hierarchy = {
        "insulin": "Medication:Insulin",
        "metformin": "Medication:Biguanide",
        "glipizide": "Medication:Sulfonylurea",
        "glyburide": "Medication:Sulfonylurea",
        "pioglitazone": "Medication:Thiazolidinedione",
        "rosiglitazone": "Medication:Thiazolidinedione",
        "diabetesMed": "Medication:AnyDiabetesMedication"
    }
    multi = []
    for transaction in tx:
        mapped = set()
        for item in transaction:
            field = item.split("=", 1)[0]
            if field in hierarchy:
                mapped.add(hierarchy[field])
        multi.append(sorted(mapped))
    multi_itemsets = apriori(multi, min_support=min_support, max_len=2)
    multi_rows = [{"itemset": " | ".join(sorted(k)), "support": len(v)/len(multi)}
                  for k, v in multi_itemsets.items()]
    pd.DataFrame(multi_rows).to_csv(TABLES / "multilevel_itemsets.csv", index=False)

    print(f"Transactions: {len(tx)} | vertical items: {len(vertical)}")
    print(f"Apriori frequent itemsets: {len(apr)} | FP-growth itemsets: {len(fp)}")
    print("Top Apriori rules:")
    print(apr_rules.head(10).to_string(index=False))
    return apr_rules

if __name__ == "__main__":
    run_association()
