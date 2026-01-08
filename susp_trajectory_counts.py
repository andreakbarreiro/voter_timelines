#!/usr/bin/env python3
import argparse
import os
import sys
import re
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

def status(msg: str):
    print(msg, flush=True)

def error(msg: str):
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Count voter trajectories from a starting month to an ending month.\n"
            "Population: voters who are S at start. Outcomes at end:\n"
            "  (1) V with AddrChanged=1 on first S->V date\n"
            "  (2) V with AddrChanged=0 on first S->V date (NA treated as 0)\n"
            "  (3) still S\n"
            "  (4) vanished (no record at end)\n\n"
            "If --start/--end are omitted, runs across all valid date pairs in the input."
        )
    )
    p.add_argument("input_csv", help="Compiled county CSV from the compile-with-address script.")
    p.add_argument("--start", help="Starting date (YYYY-MM). Optional.")
    p.add_argument("--end", help="Ending date (YYYY-MM). Optional.")

    # Optional ZIP summary (requires ZIP columns present in the CSV for the dates)
    p.add_argument("--zip", action="store_true", help="Also produce by-ZIP counts (needs per-date ZIP column).")

    # Custom output file names (only honored when BOTH start & end provided)
    p.add_argument("--precinct-out", default=None, help="Output CSV path for precinct summary (single span only).")
    p.add_argument("--zip-out", default=None, help="Output CSV path for ZIP summary (single span only).")

    # Label for missing/blank group values
    p.add_argument("--unknown-label", default="UNKNOWN", help="Label to use when the start group value is missing/blank.")

    # Also include overlapping columns: S->V w/out respect to end result
    p.add_argument("--include_overlaps", action="store_true")
    
    return p.parse_args()

# ---------- column helpers ----------
def _col_status(date: str) -> str:
    return f"{date}_Status"

def _col_precinct(date: str) -> str:
    return f"{date}_Precinct"

def _col_addr_changed(date: str) -> str:
    return f"{date}_AddrChanged"

def _find_zip_col(df: pd.DataFrame, date: str) -> Optional[str]:
    candidates = [f"{date}_Zip", f"{date}_ZIP", f"{date}_ZipCode", f"{date}_ZIPCODE"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ---------- data prep ----------
FLOAT_TAIL = re.compile(r"\.0$")

def _norm_str(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
         .str.replace(FLOAT_TAIL, "", regex=True)
         .str.strip()
    )

def _get_all_dates(df: pd.DataFrame) -> List[str]:
    dates = set()
    for c in df.columns:
        if c.endswith("_Status"):
            dates.add(c[:-7])
        elif c.endswith("_AddrChanged"):
            dates.add(c[:-12])
    return sorted(dates, key=lambda s: pd.to_datetime(s, format="%Y-%m", errors="coerce"))

def _slice_dates_between(all_dates: List[str], start: str, end: str) -> List[str]:
    try:
        i0 = all_dates.index(start)
        i1 = all_dates.index(end)
    except ValueError:
        return []
    if i1 < i0:
        return []
    return all_dates[i0:i1+1]

# ---------- core logic ----------
def compute_trajectories(df: pd.DataFrame, start: str, end: str, include_overlaps: bool, unknown_label: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    all_dates = _get_all_dates(df)
    if start not in all_dates:
        error(f"Start date {start} not found in columns.")
        sys.exit(2)
    if end not in all_dates:
        error(f"End date {end} not found in columns.")
        sys.exit(2)

    span = _slice_dates_between(all_dates, start, end)
    if not span:
        error(f"No valid date span between {start} and {end}.")
        sys.exit(2)

    if _col_status(start) not in df.columns:
        error(f"Missing column: {_col_status(start)}")
        sys.exit(2)
    if _col_status(end) not in df.columns:
        error(f"Missing column: {_col_status(end)}")
        sys.exit(2)

    status_start = _norm_str(df[_col_status(start)])
    status_end   = _norm_str(df[_col_status(end)])
    precinct_start = _norm_str(df[_col_precinct(start)]) if _col_precinct(start) in df.columns else pd.Series("", index=df.index, dtype="string")

    # Optional ZIP at start
    zip_col = _find_zip_col(df, start)
    start_zip = _norm_str(df[zip_col]) if zip_col else None

    # Population: Suspense at start
    mask_S_start = status_start.str.upper().eq("S")
    df_sub = pd.DataFrame(index=df.index)
    df_sub["VUID"]=df["VUID"]
    df_sub["start_precinct"] = precinct_start.where(precinct_start.str.len() > 0, unknown_label)
    if start_zip is not None:
        df_sub["start_zip"] = start_zip.where(start_zip.str.len() > 0, unknown_label)
    df_sub["end_status"] = status_end.str.upper()

    # Category 4: vanished (no record at end)
    mask_vanished = df_sub["end_status"].isna() | (df_sub["end_status"].str.len() == 0)
    # Category 3: still S at end
    mask_still_S = df_sub["end_status"].eq("S")

    # Build status frame across span for first S->V detection
    stat_frame = pd.DataFrame(index=df.index)
    for d in span:
        col = _col_status(d)
        stat_frame[d] = _norm_str(df[col]).str.upper() if col in df.columns else pd.NA

    # Find first date AFTER start where status == 'V' (NA-safe)
    dates_after_start = [d for d in span if d != start]
    if dates_after_start:
        mask_V_after = stat_frame[dates_after_start].eq("V")

        # NA-safe: fill NAs with False, then use any/idxmax guarded by any==True
        mask_filled = mask_V_after.fillna(False)
        has_true = mask_filled.any(axis=1)
        first_true_col = mask_filled.idxmax(axis=1)  # if all False, this returns first col; guard with has_true

        sv_trans_date = pd.Series(pd.NA, index=df.index, dtype="string")
        sv_trans_date.loc[has_true] = first_true_col.loc[has_true].astype("string")
    else:
        sv_trans_date = pd.Series(pd.NA, index=df.index, dtype="string")

    # AddrChanged on the transition date
    sv_addr_changed = pd.Series(pd.NA, index=df.index, dtype="Int64")
    have_trans = sv_trans_date.notna()
    if have_trans.any():
        for i in sv_trans_date[have_trans].index:
            d = sv_trans_date.at[i]
            col = _col_addr_changed(d)
            if col in df.columns:
                val = df.at[i, col]
                if pd.isna(val) or str(val).strip() == "":
                    sv_addr_changed.at[i] = 0
                else:
                    try:
                        sv_addr_changed.at[i] = 1 if int(str(val)) == 1 else 0
                    except Exception:
                        sv_addr_changed.at[i] = 0
            else:
                sv_addr_changed.at[i] = 0

    # Outcome masks (restricted to S at start)
    df_sub["S_at_start"] = mask_S_start
    df_sub["vanished"]   = mask_vanished & mask_S_start
    df_sub["still_s"]    = mask_still_S & mask_S_start
    df_sub["end_is_V"]   = df_sub["end_status"].eq("V") & mask_S_start & sv_trans_date.notna()

    df_sub["sv_trans_date"]   = sv_trans_date
    df_sub["sv_addr_changed"] = sv_addr_changed.fillna(0).astype("Int64")

    df_sub["cat_V_addr1"] = df_sub["end_is_V"] & (df_sub["sv_addr_changed"] == 1)
    df_sub["cat_V_addr0"] = df_sub["end_is_V"] & (df_sub["sv_addr_changed"] == 0)
    df_sub["cat_still_S"] = df_sub["still_s"]
    df_sub["cat_vanished"] = df_sub["vanished"]

    if include_overlaps:
        # Don't condition on where the person ends up
        df_sub["cat_anyStat_addr1"] = mask_S_start & sv_trans_date.notna() & (df_sub["sv_addr_changed"] == 1)
        df_sub["cat_anyStat_addr0"] = mask_S_start & sv_trans_date.notna() & (df_sub["sv_addr_changed"] == 0)
    
    verbose_test     = True
    limit_precinct_flag = False
    if verbose_test:
        ### For now, hard-code this
        # Structure question here: is it better to run "singleCounty_compile_with_address" on a single precinct?
        #    Then run this.
        #  (I see that is not already an option; I misremembered)
        #
        # Recommended here:
        #   Dallas = 3000 (about 900 S at start)
        #   Tarrant = 2331 (>= 30 addr0, 28% of S-to-V are addr0
        #   Erath = entire county
    
        precinct_to_test = 
        if limit_precinct_flag:
            df_sub_lim = df_sub[df_sub['start_precinct']==precinct_to_test]
            #print(df_sub.columns.to_list())
            df_sub_lim.to_csv(f'test_traj_P{ptest}.csv',index=False)
        else:
            df_sub_lim.to_csv(f'test_traj_AllPrec.csv',index=False)    
    
    return df_sub, (df_sub["start_zip"] if "start_zip" in df_sub.columns else None)

def summarize_by_group(df_sub: pd.DataFrame, group_col: str, include_overlaps) -> pd.DataFrame:
    # Restrict to the S-at-start population via union of outcome masks
    s_start_mask = (df_sub["cat_V_addr1"] | df_sub["cat_V_addr0"] | df_sub["cat_still_S"] | df_sub["cat_vanished"])

    # If we also incldue overlaps
    if include_overlaps:
        s_start_mask = (s_start_mask | df_sub["cat_anyStat_addr1"] | df_sub["cat_anyStat_addr0"] )
        
    sstart = df_sub[s_start_mask]

    out = sstart.groupby(group_col, dropna=False).size().rename("n_S_start").reset_index()
    cat1 = sstart.groupby(group_col, dropna=False)["cat_V_addr1"].sum().rename("n_S_to_V_addr1").reset_index()
    cat2 = sstart.groupby(group_col, dropna=False)["cat_V_addr0"].sum().rename("n_S_to_V_addr0").reset_index()
    cat3 = sstart.groupby(group_col, dropna=False)["cat_still_S"].sum().rename("n_S_to_S").reset_index()
    cat4 = sstart.groupby(group_col, dropna=False)["cat_vanished"].sum().rename("n_vanished").reset_index()

    if include_overlaps:
        cat1A = sstart.groupby(group_col, dropna=False)["cat_anyStat_addr1"].sum().rename("n_S_to_V_addr1_AnyStat").reset_index()
        cat2A = sstart.groupby(group_col, dropna=False)["cat_anyStat_addr0"].sum().rename("n_S_to_V_addr0_AnyStat").reset_index()
    
    res = out.merge(cat1, on=group_col, how="left") \
             .merge(cat2, on=group_col, how="left") \
             .merge(cat3, on=group_col, how="left") \
             .merge(cat4, on=group_col, how="left")

    if include_overlaps:
        res = res.merge(cat1A, on=group_col, how="left").merge(cat2A,  on=group_col, how="left")
        
    for c in ["n_S_start", "n_S_to_V_addr1", "n_S_to_V_addr0", "n_S_to_S", "n_vanished"]:
        res[c] = res[c].fillna(0).astype(int)

    if include_overlaps:
        for c in ["n_S_to_V_addr1_AnyStat", "n_S_to_V_addr0_AnyStat"]:
            res[c] = res[c].fillna(0).astype(int)
        
    res = res.sort_values(by=[group_col], key=lambda s: s.astype(str).str.pad(5, fillchar="0"), ignore_index=True)
    res.rename(columns={group_col: "group_value"}, inplace=True)

        
    return res

def run_span(df: pd.DataFrame, start: str, end: str, do_zip: bool, include_overlaps: bool, unknown_label: str,
             precinct_out: Optional[str]=None, zip_out: Optional[str]=None, ):
    status(f"-> Span {start} → {end}: computing trajectories (population = S at start) ...")
    df_sub, start_zip_ser = compute_trajectories(df, start, end, include_overlaps, unknown_label)

    # Output names (auto if not provided)
    if precinct_out is None:
        precinct_out = f"trajectory_by_precinct_{start}_to_{end}.csv"
    if do_zip and zip_out is None:
        zip_out = f"trajectory_by_zip_{start}_to_{end}.csv"

    # Precinct summary
    status("   Summarizing by starting precinct...")
    precinct_summary = summarize_by_group(df_sub, "start_precinct", include_overlaps)
    precinct_summary.to_csv(precinct_out, index=False)
    status(f"   Wrote precinct summary: {precinct_out}")

    # ZIP summary
    if do_zip:
        if start_zip_ser is None:
            status("   No per-date ZIP column found at start; skipping ZIP summary for this span.")
        else:
            if "start_zip" not in df_sub.columns:
                df_sub["start_zip"] = start_zip_ser
            status("   Summarizing by starting ZIP...")
            zip_summary = summarize_by_group(df_sub, "start_zip", include_overlaps)
            zip_summary.to_csv(zip_out, index=False)
            status(f"   Wrote ZIP summary: {zip_out}")

def main():
    args = parse_args()

    if not os.path.exists(args.input_csv):
        error(f"Input file not found: {args.input_csv}")
        sys.exit(2)

    # Load
    status(f"Loading compiled CSV: {args.input_csv}")
    try:
        df = pd.read_csv(args.input_csv, dtype="string", low_memory=False)
    except Exception as e:
        error(f"Failed to read input CSV: {e}")
        sys.exit(2)

    all_dates = _get_all_dates(df)
    if len(all_dates) < 2:
        error("Need at least two dated snapshots in the input to compute trajectories.")
        sys.exit(2)
    status(f"Detected {len(all_dates)} snapshots: {', '.join(all_dates[:8])}{' ...' if len(all_dates)>8 else ''}")

    # Determine spans to run
    spans: List[Tuple[str,str]] = []
    if args.start and args.end:
        spans = [(args.start, args.end)]
    elif args.start and not args.end:
        if args.start not in all_dates:
            error(f"--start {args.start} not found in file.")
            sys.exit(2)
        idx = all_dates.index(args.start)
        spans = [(args.start, d) for d in all_dates[idx+1:]]  # all ends after start
    elif args.end and not args.start:
        if args.end not in all_dates:
            error(f"--end {args.end} not found in file.")
            sys.exit(2)
        idx = all_dates.index(args.end)
        spans = [(s, args.end) for s in all_dates[:idx]]      # all starts before end
    else:
        # Do EVERYTHING: all (start, end) where end is after start
        for i, s in enumerate(all_dates[:-1]):
            for e in all_dates[i+1:]:
                spans.append((s, e))

    # Guard: when multiple spans are requested, custom output filenames are ambiguous
    if len(spans) > 1 and (args.precinct_out or args.zip_out):
        status("Note: Multiple spans requested; ignoring --precinct-out/--zip-out and using auto names per span.")

    # Execute
    status(f"Running {len(spans)} span(s)...")
    for (s, e) in spans:
        run_span(
            df=df,
            start=s,
            end=e,
            do_zip=args.zip,
            include_overlaps=args.include_overlaps,
            unknown_label=args.unknown_label,
            precinct_out=(args.precinct_out if len(spans)==1 else None),
            zip_out=(args.zip_out if len(spans)==1 else None),
        )

    status("Done.")

if __name__ == "__main__":
    main()
