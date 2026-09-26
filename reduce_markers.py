import csv
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "Genotypes.csv"
output_path = BASE_DIR / "genotype_reduced.csv"
window_size = 33
rng = random.Random(42)

with input_path.open("r", newline="") as infile:
    reader = csv.reader(infile)
    header = next(reader)

    hybrid_col = header[0]
    marker_header = header[1:]

    selected_marker_indices = [
        start + rng.randrange(window_size)
        for start in range(0, len(marker_header) - len(marker_header) % window_size, window_size)
    ]
    selected_header = [hybrid_col] + [marker_header[i] for i in selected_marker_indices]

    with output_path.open("w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(selected_header)

        row_count = 0
        for row in reader:
            hybrid_id = row[0]
            marker_values = row[1:]
            reduced_row = [hybrid_id] + [marker_values[i] for i in selected_marker_indices]
            writer.writerow(reduced_row)
            row_count += 1

print(f"Output file: {output_path.name}")
print(f"Number of rows: {row_count}")
print(f"Number of columns: {len(selected_header)}")
