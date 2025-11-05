#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run find_route.py in batch from chubu_pairs.csv (ALWAYS skip errors).

Example (Windows CMD):
  python run_routes_from_csv.py ^
    --csv "C:\\rikai\\source_code\\pgrouting_api\\data\\chubu_pairs.csv" ^
    --script "C:\\rikai\\source_code\\pgrouting_api\\find_route.py" ^
    --data-folder "C:\\rikai\\source_code\\pgrouting_api\\importer_data" ^
    --mode "truck_train_ship_train" ^
    --out-dir "C:\\rikai\\source_code\\pgrouting_api\\data\\outputs" ^
    --start-index 0 ^
    --end-index 100

Notes:
- Output name pattern: test_{index}
- {index} is the 0-based CSV row index.
- All errors are logged and the batch continues.

Example:
python run_routes_from_csv.py --csv "C:\rikai\source_code\pgrouting_api\data\chubu_pairs.csv" --script "C:\rikai\source_code\pgrouting_api\find_route.py" --data-folder "C:\rikai\source_code\pgrouting_api\importer_data" --mode "truck_train_ship_train" --out-dir "C:\rikai\source_code\pgrouting_api\data\outputs" --start-index 1 --end-index 3
"""

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to chubu_pairs.csv")
    p.add_argument("--script", required=True, help="Path to find_route.py")
    p.add_argument("--data-folder", required=True, help="Path to importer_data folder")
    p.add_argument("--mode", default="truck_train_ship_train")
    p.add_argument(
        "--out-dir",
        required=True,
        help="Directory for outputs (will be created if missing)",
    )
    p.add_argument(
        "--log-dir", default=None, help="Directory for logs (default: CSV folder)"
    )
    p.add_argument("--start-index", type=int, default=0, help="Start index (inclusive)")
    p.add_argument("--end-index", type=int, default=None, help="End index (inclusive)")
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to run find_route.py",
    )
    return p.parse_args()


def read_rows(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def to_float(v):
    try:
        return float(v)
    except Exception:
        return math.nan


def valid_lat(lat):
    return -90.0 <= lat <= 90.0


def valid_lon(lon):
    return -180.0 <= lon <= 180.0


def build_cmd(py_exe, script, x_lat, x_lon, y_lat, y_lon, data_folder, mode, out_path):
    return [
        py_exe,
        str(script),
        f"{x_lat}",
        f"{x_lon}",
        f"{y_lat}",
        f"{y_lon}",
        "--data-folder",
        str(data_folder),
        "--mode",
        str(mode),
        "--output",
        str(out_path),
        "--show-all",
    ]


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    script_path = Path(args.script)
    data_folder = Path(args.data_folder)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir) if args.log_dir else csv_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(csv_path)
    total = len(rows)
    last_index = total - 1
    start_idx = max(args.start_index, 0)
    end_idx = args.end_index if args.end_index is not None else last_index
    end_idx = min(end_idx, last_index)

    log_path = log_dir / "run_log.csv"
    with open(log_path, "w", encoding="utf-8", newline="") as logf:
        log_writer = csv.writer(logf)
        log_writer.writerow(
            [
                "index",
                "Origin_name",
                "Origin_lat",
                "Origin_lon",
                "Departure_name",
                "Departure_lat",
                "Departure_lon",
                "status",
                "returncode",
                "output_path",
                "stdout_log",
                "stderr_log",
                "error",
            ]
        )

        for idx in range(start_idx, end_idx + 1):
            row = rows[idx]
            origin_name = row.get("Origin_name", "")
            depart_name = row.get("Departure_name", "")

            x_lat = to_float(row.get("Origin_lat"))
            x_lon = to_float(row.get("Origin_lon"))
            y_lat = to_float(row.get("Departure_lat"))
            y_lon = to_float(row.get("Departure_lon"))

            out_path = out_dir / f"test_{idx}"
            stdout_path = log_dir / f"test_{idx}.stdout.txt"
            stderr_path = log_dir / f"test_{idx}.stderr.txt"

            # Validate coordinates
            if (
                math.isnan(x_lat)
                or math.isnan(x_lon)
                or math.isnan(y_lat)
                or math.isnan(y_lon)
            ):
                err = "NaN lat/lon after conversion"
                log_writer.writerow(
                    [
                        idx,
                        origin_name,
                        row.get("Origin_lat", ""),
                        row.get("Origin_lon", ""),
                        depart_name,
                        row.get("Departure_lat", ""),
                        row.get("Departure_lon", ""),
                        "invalid_coords",
                        "",
                        str(out_path),
                        str(stdout_path),
                        str(stderr_path),
                        err,
                    ]
                )
                print(f"[{idx}] Skipped: {err}", file=sys.stderr)
                continue
            if not (
                valid_lat(x_lat)
                and valid_lon(x_lon)
                and valid_lat(y_lat)
                and valid_lon(y_lon)
            ):
                err = "Lat/lon out of range"
                log_writer.writerow(
                    [
                        idx,
                        origin_name,
                        x_lat,
                        x_lon,
                        depart_name,
                        y_lat,
                        y_lon,
                        "invalid_coords",
                        "",
                        str(out_path),
                        str(stdout_path),
                        str(stderr_path),
                        err,
                    ]
                )
                print(f"[{idx}] Skipped: {err}", file=sys.stderr)
                continue

            cmd = build_cmd(
                args.python,
                script_path,
                x_lat,
                x_lon,
                y_lat,
                y_lon,
                data_folder,
                args.mode,
                out_path,
            )

            print(f"[{idx}] Running:", " ".join(map(str, cmd)))
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True)
                # Save logs
                stdout_path.write_text(proc.stdout or "", encoding="utf-8")
                stderr_path.write_text(proc.stderr or "", encoding="utf-8")

                status = "ok" if proc.returncode == 0 else "error"
                log_writer.writerow(
                    [
                        idx,
                        origin_name,
                        x_lat,
                        x_lon,
                        depart_name,
                        y_lat,
                        y_lon,
                        status,
                        proc.returncode,
                        str(out_path),
                        str(stdout_path),
                        str(stderr_path),
                        "" if status == "ok" else "see stderr log",
                    ]
                )

                if proc.returncode != 0:
                    print(
                        f"[{idx}] Error: return code {proc.returncode}. See {stderr_path}",
                        file=sys.stderr,
                    )

            except Exception as e:
                # Catch any unexpected exception and continue
                try:
                    stderr_path.write_text(str(e), encoding="utf-8")
                except Exception:
                    pass
                log_writer.writerow(
                    [
                        idx,
                        origin_name,
                        x_lat,
                        x_lon,
                        depart_name,
                        y_lat,
                        y_lon,
                        "exception",
                        "",
                        str(out_path),
                        str(stdout_path),
                        str(stderr_path),
                        str(e),
                    ]
                )
                print(f"[{idx}] Exception: {e}", file=sys.stderr)
                continue

    print(f"Done. Log saved to: {log_path}")


if __name__ == "__main__":
    main()
