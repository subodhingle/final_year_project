#!/usr/bin/env python3
"""Batch transliterate Marathi names from PostgreSQL -> Roman, written back to the DB.

Two steps (run with the project venv):

  1) (optional) create a fresh table and import the sample CSV:
       .\\.venv\\Scripts\\python.exe -u db_batch.py --init --table marathi_names --csv subodh_trial_csv.csv

  2) transliterate every row's Marathi column into the Roman column:
       .\\.venv\\Scripts\\python.exe -u db_batch.py --beam 6
       .\\.venv\\Scripts\\python.exe -u db_batch.py --beam 6 --limit 50    # only first 50

Connection + table/columns come from db_config.json (see db.py). The model is
loaded once on the GPU and reused for all rows (beam search).
"""
import argparse
import sys
import time

import db
import translit_page as xp   # reuses the in-process engine + transliterate_text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", action="store_true",
                    help="create the table and import Marathi/English pairs from --csv")
    ap.add_argument("--table", help="table name (overrides db_config.json batch_source_table)")
    ap.add_argument("--csv", default="subodh_trial_csv.csv",
                    help="CSV file to import when using --init")
    ap.add_argument("--beam", type=int, default=6, help="beam width")
    ap.add_argument("--limit", type=int, default=None, help="max rows to process")
    args = ap.parse_args()

    cfg = db.load_config()
    table = args.table or cfg.get("batch_source_table")
    src_col = cfg["batch_source_column"]
    id_col = cfg["batch_id_column"]
    out_col = cfg["batch_output_column"]
    if not table:
        print("ERROR: no table. Use --table NAME or set batch_source_table in db_config.json.")
        sys.exit(1)

    print("Connecting to PostgreSQL ...")
    ok, msg = db.test_connection()
    print("  ->", "OK" if ok else "FAILED")
    if not ok:
        print("   ", msg)
        print("   Edit db_config.json (database/user/password) or set PG* env vars, then retry.")
        sys.exit(1)

    conn = db.get_conn()

    if args.init:
        db.create_marathi_names_table(conn, table)
        n = db.load_csv_into_table(conn, table, args.csv)
        print(f"[init] table '{table}' ready; imported {n} distinct Marathi/English name pairs.")
        # show a few
        sample = db.fetch_names(conn, table, src_col, id_col)[:5]
        for rid, mr in sample:
            print(f"   id={rid}  {mr}")
        conn.close()
        return

    rows = db.fetch_names(conn, table, src_col, id_col)
    if args.limit:
        rows = rows[: args.limit]
    total = len(rows)
    print(f"[batch] {total} names to transliterate  (table={table}, "
          f"{src_col}->{out_col}, beam={args.beam})")

    # load the model once (warm the engine for the chosen beam)
    xp.get_engine(args.beam)

    t0 = time.time()
    for i, (rid, mr) in enumerate(rows, 1):
        full, _details, _elapsed = xp.transliterate_text(mr, args.beam, 1)
        db.update_name(conn, table, id_col, rid, out_col, full)
        if i % 25 == 0 or i == total or i <= 3:
            print(f"  {i}/{total}  id={rid}  {mr}  ->  {full}")
    conn.close()
    print(f"[batch] done in {time.time()-t0:.1f}s. Results written to {table}.{out_col}.")


if __name__ == "__main__":
    main()
