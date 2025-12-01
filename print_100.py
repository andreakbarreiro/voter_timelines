#!/usr/bin/env python3
# head_csv.py
import sys, csv, argparse, io, os

def open_text(path: str, encoding: str):
    if path == "-":
        # stdin (text), keep newline control for csv
        if sys.stdin is None:
            raise RuntimeError("stdin is not available")
        return sys.stdin
    return open(path, "r", encoding=encoding, newline="")

def detect_dialect(fp, force_delim: str | None):
    if force_delim:
        class Simple(csv.Dialect):
            delimiter = force_delim
            quotechar = '"'
            doublequote = True
            skipinitialspace = False
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL
        return Simple

    # Only sniff regular files (not stdin)
    if fp is sys.stdin:
        return csv.excel

    pos = fp.tell()
    sample = fp.read(4096)
    fp.seek(pos)
    try:
        return csv.Sniffer().sniff(sample) if sample else csv.excel
    except csv.Error:
        return csv.excel

def main():
    p = argparse.ArgumentParser(description="Print the first N rows of a CSV.")
    p.add_argument("csv_path", help="Path to CSV file, or '-' for stdin")
    p.add_argument("-n", "--num", type=int, default=100, help="Number of rows to print (default: 100)")
    p.add_argument("-d", "--delimiter", help="Force a specific delimiter (e.g., ',' or '\\t')")
    p.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    args = p.parse_args()

    try:
        with open_text(args.csv_path, args.encoding) as f:
            dialect = detect_dialect(f, args.delimiter)
            reader = csv.reader(f, dialect)
            # Ensure consistent newlines when writing
            writer = csv.writer(sys.stdout, lineterminator="\n")
            for i, row in enumerate(reader, start=1):
                writer.writerow(row)
                if i >= args.num:
                    break
    except FileNotFoundError:
        print(f"Error: file not found: {args.csv_path}", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"Error: permission denied: {args.csv_path}", file=sys.stderr)
        sys.exit(1)
    except UnicodeDecodeError as e:
        print(f"Error: decoding failed ({e}). Try --encoding.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
