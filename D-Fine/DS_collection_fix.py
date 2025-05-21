# here the code to extract information about depth and scans limits
# for each NDT file, so we could use it to recollect the data
# with same depths and scans limits
# without necessarity to make it all again manually

import os
import json
import re
from collections import defaultdict
from tqdm import tqdm

INPUT_FOLDER = "WOT-20250521"
OUTPUT_JSON = f"compiled_summary-{INPUT_FOLDER}.json"

result = defaultdict(list)

for file_name in tqdm(os.listdir(INPUT_FOLDER)):
    if not file_name.endswith(".json"):
        continue

    base_part = file_name.split("_Ch-0")[0]
    match = re.search(r"_D(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)", file_name)
    if not match:
        print(f"Skipping {file_name}, depth info not found")
        continue

    depth_min = float(match.group(1))
    depth_max = float(match.group(2))

    with open(os.path.join(INPUT_FOLDER, file_name), "r") as f:
        data = json.load(f)

    if not data:
        continue

    first_key = next(iter(data))
    scan_keys = [int(k.split("_")[0]) for k in data[first_key].keys()]
    if not scan_keys:
        continue

    scan_min = min(scan_keys)
    scan_max = max(scan_keys)

    result[base_part].append({
        "ScanMin": scan_min,
        "ScanMax": scan_max,
        "DepthMin": depth_min,
        "DepthMax": depth_max
    })

with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f, indent=2)

print(f"Summary saved to {OUTPUT_JSON}")

