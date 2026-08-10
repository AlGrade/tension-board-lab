import csv
from collections import Counter
from pathlib import Path


def test_official_hold_catalog_is_complete_and_unique() -> None:
    path = Path("configs/tb2_12x12_hold_catalog.csv")
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 996
    keys = {(row["layout_id"], row["set_id"], row["raw_x"], row["raw_y"]) for row in rows}
    assert len(keys) == len(rows)
    assert Counter((row["layout"], row["material"]) for row in rows) == {
        ("mirror", "wood"): 242,
        ("mirror", "plastic"): 256,
        ("spray", "wood"): 242,
        ("spray", "plastic"): 256,
    }
