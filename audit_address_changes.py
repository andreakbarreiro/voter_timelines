#!/usr/bin/env python3
import argparse
import os
import sys
import re
import pandas as pd
from typing import List, Set

def status(msg: str):
    print(msg, flush=True)

def error(msg: str):
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Audit address-change spikes by date. "
            "Filters to voters who ever transitioned S->V across *_Status columns, "
            "and (optionally) by precinct membership."
        )
    )
    p.add_argument("input_csv", help="Path to compiled county CSV (e.g., output_county001.csv).")
    p.add_argument("--precincts", help="Comma-separated precinct IDs/names (e.g., 13,101,224).", default=None)
    p.add_argument("--precincts-file", help="Text file with one precinct per line.", default=None)
    p.add_argument("--out", help="Output CSV path. Default: <input_basename>_analysis.csv", default=None)
    p.add_argument("--suspicious-threshold", type=float, default=0.5,
                   help="Flag a date as suspicious if changed_rate >= threshold (default 0.5 = 50%).")
    p.add_argument("--scope", choices=["any","all","latest"], default="any",
                   help="Precinct scope: 'any' (default), 'all', or 'latest'.")
    return p.parse_args()

def _coerce_str_strip_floaty(x: pd.Series) -> pd.Series:
    return (x.astype("string")
              .str.replace(r"\.0$", "", regex=True)
              .str.strip())

def _get_dates(df: pd.DataFrame) -> List[str]:
    dates = [c[:-12] for c in df.columns if c.endswith("_AddrChanged")]
    dates = sorted(dates, key=lambda s: pd.to_datetime(s, format="%Y-%m", errors="coerce"))
    return dates

def _status_cols(df: pd.DataFrame, dates: List[str]) -> List[str]:
    return [f"{d}_Status" for d in dates if f"{d}_Status" in df.columns]

def _precinct_cols(df: pd.DataFrame, dates: List[str]) -> List[str]:
    return [f"{d}_Precinct" for d in dates if f"{d}_Precinct" in df.columns]

def _ever_s_to_v(row_statuses: List[str]) -> bool:
    seen_s = False
    for val in row_statuses:
        if not isinstance(val, str):
            continue
        v = val.strip().upper()
        if v == "S":
            seen_s = True
        elif v == "V" and seen_s:
            return True
    return False

def _filter_s_to_v(df: pd.DataFrame, dates: List[str]) -> pd.DataFrame:
    scols = _status_cols(df, dates)
    if not scols:
        error("No *_Status columns found; cannot apply S->V filter.")
        return df.iloc[0:0]
    tmp = df[scols].astype("string")
    mask = tmp.apply(lambda r: _ever_s_to_v([x for x in r.tolist()]), axis=1)
    return df[mask]

def _load_precinct_set(args) -> Set[str] | None:
    if args.precincts_file:
        if not os.path.exists(args.precincts_file):
            error(f"Precincts file not found: {args.precincts_file}")
            sys.exit(2)
        with open(args.precincts_file, "r") as f:
            vals = [line.strip() for line in f if line.strip() != ""]
    elif args.precincts:
        vals = [v.strip() for v in args.precincts.split(",") if v.strip() != ""]
    else:
        return None  # means: analyze ALL precincts

    vals = [_coerce_str_strip_floaty(pd.Series([v])).iloc[0] for v in vals]
    return set(vals)

def _filter_precincts(df: pd.DataFrame, dates: List[str], precincts: Set[str] | None, scope: str) -> pd.DataFrame:
    pcols = _precinct_cols(df, dates)
    if not pcols:
        error("No *_Precinct columns found; precinct filter cannot be applied.")
        return df.iloc[0:0]

    # normalize all precinct columns
    for c in pcols:
        df[c] = _coerce_str_strip_floaty(df[c])

    if precincts is None:
        # analyze ALL precincts (no filtering)
        return df

    if scope == "any":
        mask_any = pd.Series(False, index=df.index)
        for c in pcols:
            mask_any |= df[c].isin(precincts)
        return df[mask_any]

    if scope == "all":
        mask_all = pd.Series(True, index=df.index)
        for c in pcols:
            non_na = df[c].notna() & (df[c].str.len() > 0)
            in_set_or_na = (~non_na) | (df[c].isin(precincts))
            mask_all &= in_set_or_na
        return df[mask_all]

    # latest
    latest_vals = pd.Series("", index=df.index, dtype="string")
    for c in reversed(pcols):
        fill_mask = (latest_vals.str.len() == 0) & df[c].notna() & (df[c].str.len() > 0)
        latest_vals = latest_vals.mask(fill_mask, df[c])
    return df[latest_vals.isin(precincts)]

def main():
    args = parse_args()

    # Validate input
    if not os.path.exists(args.input_csv):
        error(f"Input file not found: {args.input_csv}")
        sys.exit(2)

    # Derive default output filename if not provided
    if args.out:
        out_path = args.out
    else:
        base = os.path.splitext(os.path.basename(args.input_csv))[0]
        out_path = f"{base}_analysis.csv"

    status(f"Loading input CSV: {args.input_csv}")
    try:
        df = pd.read_csv(args.input_csv, dtype="string")
    except Exception as e:
        error(f"Failed to read input CSV: {e}")
        sys.exit(2)

    dates = _get_dates(df)
    if not dates:
        error("No *_AddrChanged columns found in input. Are you passing a compiled output CSV?")
        sys.exit(2)
    status(f"Detected {len(dates)} dated snapshots: {', '.join(dates[:6])}{' ...' if len(dates)>6 else ''}")

    # S->V filter
    status("Applying S->V transition filter across *_Status columns...")
    df_sv = _filter_s_to_v(df, dates)
    status(f"Rows after S->V filter: {len(df_sv):,} / {len(df):,}")
    if df_sv.empty:
        error("No rows matched S->V transition filter; nothing to analyze.")
        pd.DataFrame(columns=["date","non_na_addr_rows","changed_rows","changed_rate","suspicious"]).to_csv(out_path, index=False)
        status(f"Wrote empty summary: {out_path}")
        sys.exit(0)
    else:
        df_sv.to_csv(f'{out_path[:-4]}_MetSVCrit.csv')

    # Precinct filter
    precincts = _load_precinct_set(args)
    if precincts is None:
        status("No precinct list provided — analyzing ALL precincts.")
    else:
        status(f"Filtering to precincts ({args.scope}): {sorted(list(precincts))[:10]}{' ...' if len(precincts)>10 else ''}")

    df_filt = _filter_precincts(df_sv, dates, precincts, args.scope)
    status(f"Rows after precinct filter: {len(df_filt):,} / {len(df_sv):,}")
    if df_filt.empty:
        error("No rows remained after precinct filter; nothing to analyze.")
        pd.DataFrame(columns=["date","non_na_addr_rows","changed_rows","changed_rate","suspicious"]).to_csv(out_path, index=False)
        status(f"Wrote empty summary: {out_path}")
        sys.exit(0)

    # Per-date counts
    status("Computing per-date address-change counts...")
    rows = []
    for d in dates:
        col_chg = f"{d}_AddrChanged"
        if col_chg not in df_filt.columns:
            continue

        ser = df_filt[col_chg]
        non_na_cnt = int(ser.notna().sum())
        if non_na_cnt == 0:
            changed_cnt = 0
            rate = 0.0
        else:

            # The .csv file was read in as STRING
            changed_cnt = int((ser == '1').sum())
            rate = changed_cnt / non_na_cnt

        suspicious = (non_na_cnt > 0) and (rate >= args.suspicious_threshold)
        rows.append({
            "date": d,
            "non_na_addr_rows": non_na_cnt,
            "changed_rows": changed_cnt,
            "changed_rate": round(rate, 6),
            "suspicious": suspicious
        })

    summary = pd.DataFrame(rows).sort_values("date")
    summary.to_csv(out_path, index=False)
    status(f"Wrote summary: {out_path}")

    # Pretty print quick view
    if not summary.empty:
        print("\nAddress-change audit (filtered to S->V & precinct scope):\n")
        for _, r in summary.iterrows():
            flag = "  <-- suspicious" if r["suspicious"] else ""
            print(f"{r['date']:7}  {int(r['changed_rows']):6d} / {int(r['non_na_addr_rows']):6d}  "
                  f"(rate {r['changed_rate']:.4f}){flag}")
        print("")

if __name__ == "__main__":
    main()
