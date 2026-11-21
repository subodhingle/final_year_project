#!/usr/bin/env python3
"""PostgreSQL connection helpers for the Marathi transliteration project.

Credentials are read from db_config.json (gitignored) next to this file, with
environment variables (PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD) overriding
individual fields. Passwords are never written into source code.

Two usage patterns:
  - save_history(...)        : the web page logs each transliteration to a table.
  - fetch_names / update_name : a batch job reads Marathi names from a table,
                                transliterates them, and writes Roman results back.
"""
import csv
import json
import os
import re
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "db_config.json"

DEFAULTS = {
    "host": "localhost", "port": 5432, "database": "postgres",
    "user": "postgres", "password": "",
    "save_history": False,
    "history_table": "translit_history",
    "batch_source_table": "marathi_names",
    "batch_source_column": "name_mr",
    "batch_id_column": "id",
    "batch_output_column": "name_en",
}

_ENV_MAP = {"host": "PGHOST", "port": "PGPORT", "database": "PGDATABASE",
            "user": "PGUSER", "password": "PGPASSWORD"}

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ident(name):
    """Validate a SQL identifier (table/column) to avoid injection via config."""
    if not isinstance(name, str) or not _IDENT.match(name):
        raise ValueError(f"unsafe SQL identifier: {name!r}")
    return name


def load_config():
    """Merge defaults <- db_config.json <- environment variables."""
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.is_file():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[db] warning: could not read {CONFIG_PATH.name}: {e}")
    for k, env in _ENV_MAP.items():
        v = os.environ.get(env)
        if v is not None and v != "":
            if k == "port":
                try:
                    cfg[k] = int(v)
                except ValueError:
                    pass
            else:
                cfg[k] = v
    return cfg


def get_conn():
    """Open a psycopg2 connection using the merged config."""
    import psycopg2
    cfg = load_config()
    return psycopg2.connect(
        host=cfg["host"], port=int(cfg["port"]), dbname=cfg["database"],
        user=cfg["user"], password=cfg["password"],
    )


def test_connection():
    """Return (ok: bool, message: str)."""
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            ver = cur.fetchone()[0]
        conn.close()
        return True, ver
    except Exception as e:
        return False, str(e)


def ensure_history_table(conn, table):
    table = _ident(table)
    with conn.cursor() as cur:
        cur.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
            id BIGSERIAL PRIMARY KEY,
            input_text TEXT,
            output_text TEXT,
            beam INT,
            nbest INT,
            device TEXT,
            elapsed_sec REAL,
            created_at TIMESTAMPTZ DEFAULT now()
        )""")
    conn.commit()


def save_history(input_text, output_text, beam, nbest, device, elapsed):
    """Log one transliteration. Returns True on success, None if disabled/failed."""
    cfg = load_config()
    if not cfg.get("save_history"):
        return None
    try:
        conn = get_conn()
        ensure_history_table(conn, cfg["history_table"])
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {_ident(cfg['history_table'])} "
                "(input_text, output_text, beam, nbest, device, elapsed_sec) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (input_text, output_text, int(beam), int(nbest), device, float(elapsed)),
            )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[db] save_history failed: {e}")
        return None


def fetch_names(conn, table, col, id_col):
    """Return [(id, marathi_name), ...] for non-empty rows."""
    table, col, id_col = _ident(table), _ident(col), _ident(id_col)
    with conn.cursor() as cur:
        cur.execute(f"SELECT {id_col}, {col} FROM {table} "
                    f"WHERE {col} IS NOT NULL AND {col} <> ''")
        return cur.fetchall()


def update_name(conn, table, id_col, id_val, out_col, value):
    """Write the roman result back into one row."""
    table, id_col, out_col = _ident(table), _ident(id_col), _ident(out_col)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {table} SET {out_col} = %s WHERE {id_col} = %s",
                    (value, id_val))
    conn.commit()


def create_marathi_names_table(conn, table):
    """Create a fresh table (id, name_mr, name_en) if it does not exist."""
    table = _ident(table)
    with conn.cursor() as cur:
        cur.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
            id BIGSERIAL PRIMARY KEY,
            name_mr TEXT,
            name_en TEXT
        )""")
    conn.commit()


def load_csv_into_table(conn, table, csv_path, mr_col=3, en_col=2, limit=None):
    """Import distinct Marathi/English name pairs from the project CSV.

    Default columns: index 2 = villageNameEnglish (Roman), index 3 = villageNameLocal
    (Marathi). Returns the number of imported pairs.
    """
    rows = set()
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) > max(mr_col, en_col):
                mr = row[mr_col].strip()
                en = row[en_col].strip()
                if mr and en:
                    rows.add((mr, en))
    if limit:
        rows = set(list(rows)[:limit])
    table = _ident(table)
    with conn.cursor() as cur:
        for mr, en in rows:
            cur.execute(f"INSERT INTO {table} (name_mr, name_en) VALUES (%s,%s)",
                        (mr, en))
    conn.commit()
    return len(rows)


if __name__ == "__main__":
    ok, msg = test_connection()
    print("PostgreSQL connection:", "OK" if ok else "FAILED")
    print(msg)
    print("(edit db_config.json for host/db/user/password, or set PG* env vars)")
