#!/usr/bin/env python3
import os
import sys
import re
import pandas as pd
from datetime import datetime
from typing import Optional

#singleCounty_compile_with_address:
#
# Traverse dated file hierarchy of voter files.
# Purpose: create timeline for each unique VUID which appears.
# At each date, collect: status, precinct, zip code, address
#  Then replace "address" with an "address change flag" 

# ---------------------------------------------
# Helper definitions for VUID / address / synonyms
# ---------------------------------------------
VUID_COLS = [
    'Voter ID', 'SOS_VoterID', 'VUID', 'SOS VOTER ID', 'VUIDNO', 'vuid'
]

# Status and Precinct synonyms
STATUS_COLS = [
    'Status Code', 'STATUS CODE',
    'Voter Status', 'voter_status', 'VOTER STATUS',
    'Status', 'STATUS',
    'STATUS_CD', 'STATUSCODE'
]
PRECINCT_COLS = [
    'Precinct', 'precinct', 'PRECINCT',
    'Precinct Code', 'PRECINCT CODE',
    'PCT', 'PCTCODE', 'PCT_CODE', 'Precinct ID'
]

# Address groups (includes Perm-* residence fields)
ADDRESS_GROUPS = {
    'full': ['Address', 'Residence Address'],
    'street_number': ['Street Number','streetnumber','Str Nbr','STREETNO','House_Number','Perm House Number'],
    'street_building': ['Street Building','streetbuilding','STREETBLD'],
    'street_predir': ['Street Pre-Direction','streetpredirection','Str Dir','Direction Prefix','STREETPREDIR','STRDIR','Perm Directional Prefix'],
    'street_name': ['Street Name','streetname','STREETNAME','Residence_S','STRNAM','Perm Street Name'],
    'street_type': ['Street Type','streettype','STREETTYPE','STRTYP','Perm Street Type','Perm Designator'],
    'street_postdir': ['Street Post-Direction','streetpostdirection','STREETPOSTDIR','Perm Directional Suffix'],
    'unit_type': ['Unit Type','unit_type','UNITYP','Perm Unit Type'],
    'unit': ['Unit','unit','Apt Nbr','Unit Number','UNIT','UNITNO','UNIT_NUM','Perm Unit Number'],
    'city': ['City','city','Residence_','RSCITY','Perm City'],
    'zip': ['Zip Code','zip','Zip','ZIP','Residence_','Perm Zipcode'],
}

# ---------- utilities ----------
def _norm_val(x):
    if pd.isna(x):
        return ''
    s = str(x).strip().upper()
    s = re.sub(r'[^A-Z0-9 ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _matching_columns(df, target_names):
    lower_to_cols = {}
    for c in df.columns:
        lower_to_cols.setdefault(c.lower(), []).append(c)
    found = []
    for name in target_names:
        cols = lower_to_cols.get(name.lower(), [])
        for c in cols:
            if c not in found:
                found.append(c)
    return found

def _extract_series(df, candidates):
    cols = _matching_columns(df, candidates)
    if not cols:
        return None
    if len(cols) == 1:
        return df[cols[0]]
    sub = df[cols]
    return sub.apply(lambda row: next((v for v in row if pd.notna(v) and str(v).strip() != ''), pd.NA), axis=1)

# --- Stable address key helpers ---
ZIP5_RE = re.compile(r'^(\d{5})')
FLOAT_ZERO_RE = re.compile(r'^\s*(\d+)\.0+\s*$')   # 13.0 -> 13
FLOAT_ANY_RE  = re.compile(r'^\s*(\d+)\.(\d+)\s*$') # 13.50 -> 13 (house/unit numbers shouldn't be real floats)

def _zip5(x: str) -> str:
    if not isinstance(x, str):
        x = '' if pd.isna(x) else str(x)
    m = ZIP5_RE.match(x.strip())
    return m.group(1) if m else ''

def _norm_num_token(x) -> str:
    """Normalize numeric-ish tokens: strip .0 tails and non-alnum."""
    if pd.isna(x):
        return ''
    s = str(x).strip().upper()
    m = FLOAT_ZERO_RE.match(s)
    if m:
        return m.group(1)
    m = FLOAT_ANY_RE.match(s)
    if m:
        return m.group(1)
    # Keep only letters/digits, no separators
    s = re.sub(r'[^A-Z0-9]+', '', s)
    return s

def _build_addr_key(df: pd.DataFrame) -> pd.Series:
    """
    Build a canonical, stable address key **from parts** whenever possible.
    Only if all relevant parts are missing do we fall back to a normalized 'full' line.
    """
    parts_map = {}
    keys_in_order = [
        'street_number','street_predir','street_name','street_type',
        'street_postdir','street_building','unit_type','unit','city','zip'
    ]
    have_any_part = False
    for key in keys_in_order:
        ser = _extract_series(df, ADDRESS_GROUPS[key])
        if ser is None:
            parts_map[key] = pd.Series([''] * len(df), index=df.index)
        else:
            have_any_part = True
            if key == 'zip':
                parts_map[key] = ser.map(lambda v: _zip5(str(v)))                 # ZIP → 5-digit
            elif key in ('street_number','unit'):
                parts_map[key] = ser.map(_norm_num_token)                         # strip “.0” & non-alnum
            else:
                parts_map[key] = ser.map(_norm_val)                               # default normalization

    if have_any_part:
        tmp = pd.concat([parts_map[k] for k in keys_in_order], axis=1).fillna('')
        return tmp.apply(lambda r: '|'.join([v for v in r if v]), axis=1)

    # Fallback to full line ONLY if we truly have no parts
    full = _extract_series(df, ADDRESS_GROUPS['full'])
    if full is not None:
        return full.map(_norm_val)
    return pd.Series([''] * len(df), index=df.index)

def _alias_core_columns(df):
    vuid_ser = _extract_series(df, VUID_COLS + ['VUID'])
    df['VUID'] = vuid_ser.astype('string') if vuid_ser is not None else pd.Series(pd.NA, index=df.index, dtype='string')

    status_ser = _extract_series(df, STATUS_COLS)
    df['Status Code'] = status_ser if status_ser is not None else pd.Series(pd.NA, index=df.index)

    pct_ser = _extract_series(df, PRECINCT_COLS + ['Precinct'])
    df['Precinct'] = pct_ser.astype('string') if pct_ser is not None else pd.Series(pd.NA, index=df.index, dtype='string')

    cnty_ser = _extract_series(df, ['County Code','COUNTY CODE','CountyCode','COUNTYCODE','CNTY','County'])
    if cnty_ser is None:
        df['County Code'] = pd.Series(pd.NA, index=df.index)
    else:
        df['County Code'] = pd.to_numeric(cnty_ser, errors='coerce')
    return df

# ---------- master sheet helpers ----------
def _load_master_sheet_or_die():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    meta_path = os.path.join(script_dir, "DirectoryList_TXVoterFiles.xlsx")
    if not os.path.exists(meta_path):
        print("ERROR: DirectoryList_TXVoterFiles.xlsx must be in the same folder as this script.")
        sys.exit(1)
    df = pd.read_excel(meta_path)

    # normalize and build fast lookup
    for c in ['Directory','Sub, if applicable','Date Acquired (GUESS)','Type']:
        if c not in df.columns:
            print(f"ERROR: Master sheet missing required column: {c}")
            sys.exit(1)
    df = df[['Directory','Sub, if applicable','Date Acquired (GUESS)','Type']].copy()

    for c in ['Directory','Sub, if applicable','Type']:
        df[c] = df[c].astype(str).str.strip()

    df['_dir_key'] = df['Directory'].str.lower().str.strip()
    df['_sub_key'] = df['Sub, if applicable'].str.lower().str.strip()

    # build a dict for O(1) lookups on both Directory and Sub
    idx = {}
    for _, row in df.iterrows():
        if row['_dir_key'] and row['_dir_key'] != 'nan':
            idx[row['_dir_key']] = row
        if row['_sub_key'] and row['_sub_key'] != 'nan':
            idx[row['_sub_key']] = row
    return df, idx

def _meta_lookup(meta_index, name) -> Optional[pd.Series]:
    key = str(name).strip().lower()
    return meta_index.get(key)

def _datestr_from_meta_row(row):
    dt = row['Date Acquired (GUESS)']
    if not isinstance(dt, pd.Timestamp):
        dt = pd.to_datetime(dt, errors='coerce')
    if pd.isna(dt):
        return None
    return dt.strftime("%Y-%m")

def _is_history_only(row):
    t = str(row.get('Type', '')).strip().lower()
    return t.startswith('voter history')

# --- "Sub, if applicable" helpers for pre-2013 sets ---
def _sub_pattern_from_meta_row(row) -> Optional[str]:
    """
    Extract a glob-like pattern from 'Sub, if applicable' such as:
    'Set 1: fname contains *20100715*'  -> '*20100715*'
    'Set 1: fname contains *_1071*'     -> '*_1071*'
    'Set 1: fname contains *_1396_*'    -> '*_1396_*'
    Returns the pattern with '*' wildcards or None.
    """
    raw = str(row.get('Sub, if applicable', '') or '').strip()
    if not raw or raw.lower() == 'nan':
        return None
    m = re.search(r'(\*[^"]+\*)', raw)
    if m:
        return m.group(1)
    if 'contains' in raw:
        tail = raw.split('contains', 1)[1].strip().strip(':').strip()
        tail = tail.strip('"').strip("'")
        return tail if '*' in tail else f'*{tail}*'
    return None

def _glob_to_regex(glob_pat: str) -> re.Pattern:
    """
    Convert a simple glob (only *) to a compiled regex, case-insensitive.
    """
    esc = re.escape(glob_pat)
    esc = esc.replace(r'\*', '.*')
    return re.compile(esc, re.IGNORECASE)

# ------------------------------------------------------------
# Core per-directory processing (returns updated combined_data)
# ------------------------------------------------------------
def process_csv_directory(directory, combined_data, which_cnty, meta_index):
    dirname = os.path.basename(directory)

    # Master lookup (strict; no fallback)
    meta_row = _meta_lookup(meta_index, dirname)
    if meta_row is None:
        print(f"SKIP (no master row): '{dirname}'. Add it to DirectoryList_TXVoterFiles.xlsx.")
        return combined_data

    # Skip voter-history-only folders
    if _is_history_only(meta_row):
        print(f"Skipping '{dirname}' due to Type='{meta_row['Type']}' in master sheet.")
        return combined_data

    # Use date from master sheet ONLY
    datestr = _datestr_from_meta_row(meta_row)
    if datestr is None:
        print(f"SKIP (no valid date in master): '{dirname}'.")
        return combined_data

    # Honor Sub pattern, if present
    sub_glob = _sub_pattern_from_meta_row(meta_row)
    sub_regex = _glob_to_regex(sub_glob) if sub_glob else None

    # Support both _57.csv and _057.csv
    suffixes = {f"_{which_cnty}.csv", f"_{str(which_cnty).zfill(3)}.csv"}

    try:
        entries = os.listdir(directory)
    except FileNotFoundError:
        return combined_data

    # First filter: county suffix
    candidates = [file for file in entries if any(file.endswith(sfx) for sfx in suffixes)]

    # Second filter: Sub pattern (if provided)
    if sub_regex:
        candidates = [f for f in candidates if sub_regex.search(f)]

    if not candidates:
        return combined_data

    if len(candidates) > 1:
        print(f'WARNING: more than one CSV file in {dirname} after filters:')
        for f in candidates:
            print(f'   - {f}')
        print('   Proceeding with the first match.\n')

    csv_files = candidates[:1]  # be strict: one set per the master Sub

    print(f'\n\n{dirname}\n\n')
    print(f'\n\n{datestr}\n\n')

    # Prepare per-date columns if absent
    for suffix in ('_Status', '_Precinct', '_AddrKey'):
        col = datestr + suffix
        if col not in combined_data.columns:
            combined_data[col] = pd.NA

    for csv_file in csv_files:
        filepath = os.path.join(directory, csv_file)
        print(filepath)

        # Read CSV: force every column to string to avoid 13 -> 13.0 coercion
        df = pd.read_csv(filepath, dtype=str)

        # Canonicalize / dedupe columns
        df = _alias_core_columns(df)

        # Filter on county
        cnty_int = int(which_cnty)
        df = df[df['County Code'] == cnty_int]
        print(f"Rows post-filter (County {which_cnty}): {df.shape[0]}")
        if df.empty:
            continue

        # Address key (stable recipe + numeric-token fixes)
        df['__ADDR_KEY__'] = _build_addr_key(df)
        nonempty_addr = (df['__ADDR_KEY__'].astype('string').str.len() > 0).sum()
        print(f"Address keys populated: {nonempty_addr}/{len(df)}")

        # --- Vectorized cohort growth ---
        if 'VUID' not in combined_data.columns:
            combined_data['VUID'] = pd.Series(dtype='string')

        existing = set(combined_data['VUID'].astype('string').dropna())
        incoming = pd.Series(df['VUID'].dropna().astype('string').unique())
        to_add = sorted(set(incoming) - existing)
        if to_add:
            add_df = pd.DataFrame({'VUID': pd.Series(to_add, dtype='string')})
            combined_data = pd.concat([combined_data, add_df], ignore_index=True)

        # Merge onto cohort by VUID to align values for assignment
        cols_for_merge = ['VUID', 'Precinct', 'Status Code', '__ADDR_KEY__']
        df_merged = pd.merge(combined_data[['VUID']], df[cols_for_merge], on='VUID', how='left')

        # Assign per-date columns (clean precinct)
        status_col = df_merged['Status Code'] if 'Status Code' in df_merged.columns else pd.NA
        pct_col = df_merged['Precinct'] if 'Precinct' in df_merged.columns else pd.NA
        pct_col = pd.Series(pct_col, copy=False).astype('string').str.replace(r'\.0$', '', regex=True)

        combined_data[datestr + '_Status']   = status_col
        combined_data[datestr + '_Precinct'] = pct_col
        combined_data[datestr + '_AddrKey']  = df_merged['__ADDR_KEY__'] if '__ADDR_KEY__' in df_merged.columns else pd.NA

        print(combined_data.shape)

    return combined_data

# ------------------------------------------------------------
# Traversal (returns updated combined_data)
# ------------------------------------------------------------
def process_directories_NORECUR(directory, which_cnty, meta_index, combined_data, alldirlist, alldirlist2):
    combined_data = process_csv_directory(directory, combined_data, which_cnty, meta_index)
    for root, dirs, files in os.walk(directory):
        # skip literal voter history subtrees
        dirs[:] = [d for d in dirs if d != "VoteHis"]
        alldirlist2.append(dirs)
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            alldirlist.append(subdir_path)
            combined_data = process_csv_directory(subdir_path, combined_data, which_cnty, meta_index)
    return combined_data

# ------------------------------------------------------------
# Audit helpers
# ------------------------------------------------------------
def _expected_dirs_under(root_dir: str, meta_df: pd.DataFrame) -> set[str]:
    # Non-history rows only; take the master 'Directory' under the given root if present
    need = meta_df[~meta_df['Type'].astype(str).str.lower().str.startswith('voter history')]['Directory'].astype(str).str.strip()
    expected = set()
    for d in need:
        abs_path = os.path.join(root_dir, d)
        if os.path.isdir(abs_path):
            expected.add(os.path.abspath(abs_path))
    return expected

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    if len(sys.argv) != 4:
        print("Usage: python singleCounty_compile_with_address.py <base_csv_path> <directory_path> <CNTY>")
        sys.exit(1)

    base_path = sys.argv[1]
    directory_path = sys.argv[2]
    which_cnty = int(sys.argv[3])  # ensure int once

    # Load master sheet from script folder (required) and build lookup index
    meta_df, meta_index = _load_master_sheet_or_die()
    print(meta_df.columns)
    print(meta_df.head())

    # If base CSV missing/blank → create an empty seed file on-the-fly
    if (not os.path.exists(base_path)) or (os.path.isfile(base_path) and os.path.getsize(base_path) == 0):
        if not os.path.exists(base_path):
            pd.DataFrame(columns=['VUID','Precinct']).to_csv(base_path, index=False)

    # Load base cohort (may be empty)
    combined_data = pd.read_csv(base_path, dtype={'VUID': 'str', 'Precinct':'str'})

    # Keep original precinct column (or create blank if absent)
    if 'Precinct' in combined_data.columns:
        combined_data.rename(columns={"Precinct":"Precinct_Orig"}, inplace=True)
    elif 'Precinct_Orig' not in combined_data.columns:
        combined_data['Precinct_Orig'] = pd.Series(dtype='string')

    # --- BEGIN one-time cleanup for legacy columns + precinct dtype normalization ---
    import re as _re
    allowed = {'VUID', 'Precinct_Orig'}
    def _is_ok(col: str) -> bool:
        if col in allowed:
            return True
        # match YYYY-MM_(Status|Precinct|AddrChanged)
        return bool(_re.match(r'^\d{4}-\d{2}_(Status|Precinct|AddrChanged)$', col))

    legacy_cols = [c for c in combined_data.columns if not _is_ok(c)]
    if legacy_cols:
        print(f"Dropping legacy columns: {legacy_cols}")
        combined_data.drop(columns=legacy_cols, inplace=True, errors='ignore')

    # Normalize ALL *_Precinct columns to string and strip trailing '.0'
    precinct_cols = [c for c in combined_data.columns if c.endswith('_Precinct')]
    for c in precinct_cols:
        combined_data[c] = (
            combined_data[c]
              .astype('string')
              .str.replace(r'\.0$', '', regex=True)
              .str.strip()
        )
    # --- END one-time cleanup ---

    # Logging containers
    alldirlist = []
    alldirlist2 = []

    # Process root directory and all subdirs (skip any literally named "VoteHis")
    combined_data = process_directories_NORECUR(directory_path, which_cnty, meta_index, combined_data, alldirlist, alldirlist2)

    # ---------------------------------------------
    # Audit: ensure every non-history master dir was processed
    # ---------------------------------------------
    processed_dirs = set(map(os.path.abspath, alldirlist))
    processed_dirs.add(os.path.abspath(directory_path))
    expected_dirs = _expected_dirs_under(directory_path, meta_df)
    missing = sorted(expected_dirs - processed_dirs)
    if missing:
        pd.DataFrame({'missing_dir': missing}).to_csv('unprocessed_non_history_dirs.csv', index=False)
        print(f"Audit: {len(missing)} non-history directories present in master but not processed. See unprocessed_non_history_dirs.csv")
    else:
        print("Audit: all non-history directories from master were processed.")

    # ---------------------------------------------
    # Compute per-date address-change flags
    # ---------------------------------------------
    addr_key_cols = [c for c in combined_data.columns if c.endswith('_AddrKey')]
    date_order = sorted(
        [c.replace('_AddrKey', '') for c in addr_key_cols],
        key=lambda s: pd.to_datetime(s, format='%Y-%m', errors='coerce')
    )

    prev = pd.Series(pd.NA, index=combined_data.index, dtype='string')
    for d in date_order:
        key_col = f'{d}_AddrKey'
        if key_col not in combined_data.columns:
            continue
        curr = combined_data[key_col].astype('string')

        flag = pd.Series(pd.NA, index=combined_data.index, dtype='Int64')
        has_curr = curr.notna() & (curr.str.len() > 0)
        has_prev = prev.notna() & (prev.str.len() > 0)

        flag = flag.mask(has_curr & ~has_prev, 0)                 # first seen → 0
        flag = flag.mask(has_curr & has_prev & (curr != prev), 1) # changed   → 1
        flag = flag.mask(has_curr & has_prev & (curr == prev), 0) # same      → 0

        combined_data[f'{d}_AddrChanged'] = flag
        prev = prev.mask(has_curr, curr)

    # Drop temporary *_AddrKey columns
    if addr_key_cols:
        combined_data.drop(columns=addr_key_cols, inplace=True)

    # Reorder columns (first 2 preserved if present; rest lexicographically)
    colList = combined_data.columns.tolist()
    colListA = colList[0:2] if len(colList) >= 2 else colList[:]
    colListB = colList[len(colListA):]
    colListB.sort()
    combined_data = combined_data[colListA + colListB]

    # --- Write a fresh, timestamped output every run ---
    run_ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_fname = f'output_county{which_cnty:03d}_{run_ts}.csv'
    combined_data.to_csv(out_fname, index=False)
    print(f"Wrote fresh run output: {out_fname}")

    if False:
        # (Optional) also write a dated county summary if you like
        county_summary = f'combined_data_CNTY{which_cnty:03d}_{run_ts}.csv'
        combined_data.to_csv(county_summary, index=False)
        print(f"Wrote county summary: {county_summary}")

    # Directory audits (raw traversal logs)
    alldirDF = pd.DataFrame(alldirlist)
    alldirDF.to_csv("all_directories_traversed.csv", index=False)
    alldirDF2 = pd.DataFrame(alldirlist2)
    alldirDF2.to_csv("all_directories_traversed_simple.csv", index=False)

if __name__ == "__main__":
    main()
