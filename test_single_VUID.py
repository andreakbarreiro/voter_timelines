#!/usr/bin/env python3
# test.py — single-VUID debugger aligned with the current compile logic

import os
import sys
import re
import pandas as pd
from typing import Optional

# ======== Config: synonyms (kept in sync with main script) ========
VUID_COLS = ['Voter ID','SOS_VoterID','VUID','SOS VOTER ID','VUIDNO','vuid']

STATUS_COLS = [
    'Status Code','STATUS CODE',
    'Voter Status','voter_status','VOTER STATUS',
    'Status','STATUS','STATUS_CD','STATUSCODE'
]

PRECINCT_COLS = [
    'Precinct','precinct','PRECINCT',
    'Precinct Code','PRECINCT CODE','PCT','PCTCODE','PCT_CODE','Precinct ID'
]

ADDRESS_GROUPS = {
    'full': ['Address','Residence Address'],
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

# ======== Small helpers ========
def _norm_val(x):
    if pd.isna(x): return ''
    s = str(x).strip().upper()
    s = re.sub(r'[^A-Z0-9 ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _matching_columns(df, names):
    lowmap = {}
    for c in df.columns:
        lowmap.setdefault(c.lower(), []).append(c)
    out = []
    for n in names:
        for c in lowmap.get(n.lower(), []):
            if c not in out:
                out.append(c)
    return out

def _extract_series_first_nonempty(df, candidates):
    cols = _matching_columns(df, candidates)
    if not cols: return None
    if len(cols) == 1: return df[cols[0]]
    sub = df[cols]
    return sub.apply(lambda r: next((v for v in r if pd.notna(v) and str(v).strip()!=''), pd.NA), axis=1)

# --- Stable key pieces (aligned with compile) ---
ZIP5_RE = re.compile(r'^(\d{5})')
FLOAT_ZERO_RE = re.compile(r'^\s*(\d+)\.0+\s*$')   # 13.0 -> 13
FLOAT_ANY_RE  = re.compile(r'^\s*(\d+)\.(\d+)\s*$') # 13.50 -> 13 (house/unit shouldn't be real floats)

def _zip5(x: str) -> str:
    if not isinstance(x, str):
        x = '' if pd.isna(x) else str(x)
    m = ZIP5_RE.match(x.strip())
    return m.group(1) if m else ''

def _norm_num_token(x) -> str:
    if pd.isna(x):
        return ''
    s = str(x).strip().upper()
    m = FLOAT_ZERO_RE.match(s)
    if m:
        return m.group(1)
    m = FLOAT_ANY_RE.match(s)
    if m:
        return m.group(1)
    s = re.sub(r'[^A-Z0-9]+', '', s)  # drop separators
    return s

def _build_addr_key(df):
    """
    Build a canonical, stable address key **from parts** whenever possible.
    Fallback to normalized 'full' only if parts are entirely unavailable.
    """
    parts_map = {}
    order = ['street_number','street_predir','street_name','street_type',
             'street_postdir','street_building','unit_type','unit','city','zip']
    have_any_part = False
    for k in order:
        ser = _extract_series_first_nonempty(df, ADDRESS_GROUPS[k])
        if ser is None:
            parts_map[k] = pd.Series(['']*len(df), index=df.index)
        else:
            have_any_part = True
            if k == 'zip':
                parts_map[k] = ser.map(lambda v: _zip5(str(v)))
            elif k in ('street_number','unit'):
                parts_map[k] = ser.map(_norm_num_token)
            else:
                parts_map[k] = ser.map(_norm_val)

    if have_any_part:
        tmp = pd.concat([parts_map[k] for k in order], axis=1).fillna('')
        return tmp.apply(lambda r: '|'.join([v for v in r if v]), axis=1)

    full = _extract_series_first_nonempty(df, ADDRESS_GROUPS['full'])
    return full.map(_norm_val) if full is not None else pd.Series(['']*len(df), index=df.index)

def _build_addr_display(df):
    """Human-readable rebuilt address (row-wise; no Series truthiness)."""
    full = _extract_series_first_nonempty(df, ADDRESS_GROUPS['full'])
    if full is not None:
        return full.fillna('').astype(str).str.strip()

    def _ser(names):
        s = _extract_series_first_nonempty(df, names)
        return s.fillna('').astype(str).str.strip() if s is not None else pd.Series(['']*len(df), index=df.index)

    # Keep display readable but remove float tails on number/unit and zip
    num   = _ser(ADDRESS_GROUPS['street_number']).map(_norm_num_token)
    pre   = _ser(ADDRESS_GROUPS['street_predir']).map(_norm_val)
    name  = _ser(ADDRESS_GROUPS['street_name']).map(_norm_val)
    typ   = _ser(ADDRESS_GROUPS['street_type']).map(_norm_val)
    post  = _ser(ADDRESS_GROUPS['street_postdir']).map(_norm_val)
    bldg  = _ser(ADDRESS_GROUPS['street_building']).map(_norm_val)
    ut    = _ser(ADDRESS_GROUPS['unit_type']).map(_norm_val)
    un    = _ser(ADDRESS_GROUPS['unit']).map(_norm_num_token)
    city  = _ser(ADDRESS_GROUPS['city']).map(_norm_val)
    zipc  = _ser(ADDRESS_GROUPS['zip']).map(lambda v: _zip5(str(v)))

    street = (num + ' ' + pre + ' ' + name + ' ' + typ + ' ' + post + ' ' + bldg).str.replace(r'\s+', ' ', regex=True).str.strip()
    unit_pad = un.where(un.str.len() == 0, ' ' + un)
    ut_pad = ut.where(ut.str.len() == 0, ' ' + ut)
    unit_full = (ut_pad.fillna('') + unit_pad.fillna('')).str.replace(r'\s+', ' ', regex=True).str.strip()
    unit_full = unit_full.where(unit_full.str.len() > 0, '')
    line1  = (street + unit_full).str.replace(r'\s+', ' ', regex=True).str.strip()

    has_city = city.str.len() > 0
    disp_with_city = (line1 + ', ' + city + ' ' + zipc).str.replace(r'\s+', ' ', regex=True).str.strip()
    disp = line1.mask(has_city, disp_with_city)
    return disp.fillna('').astype(str).str.strip()

def _alias_core_columns(df):
    vuid_ser = _extract_series_first_nonempty(df, VUID_COLS+['VUID'])
    df['VUID'] = vuid_ser.astype('string') if vuid_ser is not None else pd.Series(pd.NA, index=df.index, dtype='string')

    status_ser = _extract_series_first_nonempty(df, STATUS_COLS)
    df['Status Code'] = status_ser if status_ser is not None else pd.Series(pd.NA, index=df.index)

    pct_ser = _extract_series_first_nonempty(df, PRECINCT_COLS+['Precinct'])
    df['Precinct'] = pct_ser.astype('string') if pct_ser is not None else pd.Series(pd.NA, index=df.index, dtype='string')

    cnty_ser = _extract_series_first_nonempty(df, ['County Code','COUNTY CODE','CountyCode','COUNTYCODE','CNTY','County'])
    df['County Code'] = pd.to_numeric(cnty_ser, errors='coerce') if cnty_ser is not None else pd.Series(pd.NA, index=df.index)
    return df

# ======== Master sheet (must be beside this script) ========
def _load_master():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    xlsx = os.path.join(script_dir, "DirectoryList_TXVoterFiles.xlsx")
    if not os.path.exists(xlsx):
        print("ERROR: DirectoryList_TXVoterFiles.xlsx must be in the same folder as test.py")
        sys.exit(1)
    df = pd.read_excel(xlsx)
    req = ['Directory','Sub, if applicable','Date Acquired (GUESS)','Type']
    for c in req:
        if c not in df.columns:
            print(f"ERROR: master sheet missing column: {c}")
            sys.exit(1)
    meta = df[req].copy()
    for c in ['Directory','Sub, if applicable','Type']:
        meta[c] = meta[c].astype(str).str.strip()
    meta['_dir_key'] = meta['Directory'].str.lower().str.strip()
    meta['_sub_key'] = meta['Sub, if applicable'].str.lower().str.strip()
    idx = {}
    for _, row in meta.iterrows():
        if row['_dir_key'] and row['_dir_key']!='nan':
            idx[row['_dir_key']] = row
        if row['_sub_key'] and row['_sub_key']!='nan':
            idx[row['_sub_key']] = row
    return meta, idx

def _lookup_dir(meta_index, name) -> Optional[pd.Series]:
    return meta_index.get(str(name).strip().lower())

def _is_history_only(row):
    t = str(row.get('Type','')).strip().lower()
    return t.startswith('voter history')

def _datestr(row):
    dt = row['Date Acquired (GUESS)']
    if not isinstance(dt, pd.Timestamp):
        dt = pd.to_datetime(dt, errors='coerce')
    return None if pd.isna(dt) else dt.strftime("%Y-%m")

# --- "Sub, if applicable" helpers for pre-2013 sets ---
def _sub_pattern_from_meta_row(row) -> Optional[str]:
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
    esc = re.escape(glob_pat)
    esc = esc.replace(r'\*', '.*')
    return re.compile(esc, re.IGNORECASE)

# ======== Build timeline for ONE VUID ========
def build_vuid_timeline(directory_root, cnty, vuid, meta_index):
    timeline = {}  # date -> dict(...)
    cnty_int = int(cnty)
    suffixes = {f"_{cnty}.csv", f"_{str(cnty).zfill(3)}.csv"}

    def touch_dir(path):
        dirname = os.path.basename(path)
        mrow = _lookup_dir(meta_index, dirname)
        if mrow is None or _is_history_only(mrow):
            return
        dstr = _datestr(mrow)
        if dstr is None:
            return

        sub_glob = _sub_pattern_from_meta_row(mrow)
        sub_regex = _glob_to_regex(sub_glob) if sub_glob else None

        try:
            entries = os.listdir(path)
        except FileNotFoundError:
            return

        # Filter by county suffix first
        candidates = [f for f in entries if any(f.endswith(sfx) for sfx in suffixes)]
        # Then by Sub pattern (if present)
        if sub_regex:
            candidates = [f for f in candidates if sub_regex.search(f)]

        if not candidates:
            return
        if len(candidates) > 1:
            print(f"WARNING: multiple CSVs in {dirname} after filters; using first")
        csv_file = sorted(candidates)[0]
        fp = os.path.join(path, csv_file)

        # Read CSV as text to avoid float coercion
        try:
            df = pd.read_csv(fp, dtype=str, low_memory=False)
        except Exception as e:
            print(f"ERROR reading {fp}: {e}")
            return

        df = _alias_core_columns(df)
        df = df[df['County Code'] == cnty_int]
        if df.empty:
            return
        df = df[df['VUID'].astype('string') == str(vuid)]
        if df.empty:
            return

        df['__ADDR_KEY__']  = _build_addr_key(df)
        df['__ADDR_DISP__'] = _build_addr_display(df)

        row = df.iloc[0]
        status = row.get('Status Code', pd.NA)
        pct = row.get('Precinct', pd.NA)
        pct = (pd.Series([pct]).astype('string').str.replace(r'\.0$','', regex=True).iloc[0])
        addrkey  = row.get('__ADDR_KEY__', '')
        addrdisp = row.get('__ADDR_DISP__', '')

        timeline[dstr] = {
            'Status': status,
            'Precinct': pct,
            'AddrKey': str(addrkey) if addrkey is not None else '',
            'AddrDisplay': str(addrdisp) if addrdisp is not None else ''
        }

    # process root and subdirs (skip literal "VoteHis")
    touch_dir(directory_root)
    for root, dirs, files in os.walk(directory_root):
        dirs[:] = [d for d in dirs if d != "VoteHis"]
        for sub in dirs:
            touch_dir(os.path.join(root, sub))

    out_rows = []
    prev = ''
    for d in sorted(timeline.keys(), key=lambda s: pd.to_datetime(s, format="%Y-%m", errors="coerce")):
        cur = timeline[d]['AddrKey'] or ''
        if cur == '':
            flag = pd.NA
        elif prev == '':
            flag = 0
            prev = cur
        else:
            flag = 1 if cur != prev else 0
            if cur != '':
                prev = cur
        out_rows.append({
            'Date': d,
            'Status_calc': timeline[d]['Status'],
            'Precinct_calc': timeline[d]['Precinct'],
            'AddrChanged_calc': flag,
            'AddrDisplay': timeline[d]['AddrDisplay'],
            'AddrKey': timeline[d]['AddrKey']
        })
    return pd.DataFrame(out_rows)

# ======== Compare with compiled output row ========
def compare_with_output(output_csv, vuid, df_calc):
    if not os.path.exists(output_csv):
        print("NOTE: compiled output CSV not found; skipping comparison.")
        return df_calc.assign(Status_out=pd.NA, Precinct_out=pd.NA, AddrChanged_out=pd.NA,
                              Status_match=pd.NA, Precinct_match=pd.NA, AddrChanged_match=pd.NA)

    base = pd.read_csv(output_csv, dtype=str, low_memory=False)
    row = base[base['VUID'].astype(str) == str(vuid)]
    if row.empty:
        print("NOTE: VUID not found in compiled output; comparison will show output as NaN.")
        return df_calc.assign(Status_out=pd.NA, Precinct_out=pd.NA, AddrChanged_out=pd.NA,
                              Status_match=pd.NA, Precinct_match=pd.NA, AddrChanged_match=pd.NA)

    row = row.iloc[0]

    status_out, pct_out, addr_out = [], [], []
    for d in df_calc['Date']:
        status_out.append(row.get(f'{d}_Status', pd.NA))
        pv = row.get(f'{d}_Precinct', pd.NA)
        pct_out.append(pd.NA if pd.isna(pv) else str(pv).rstrip('.0'))
        addr_out.append(row.get(f'{d}_AddrChanged', pd.NA))

    res = df_calc.copy()
    res['Status_out'] = status_out
    res['Precinct_out'] = pct_out
    res['AddrChanged_out'] = addr_out

    res['Status_match']   = (res['Status_calc'].astype(str) == res['Status_out'].astype(str))
    res['Precinct_match'] = (res['Precinct_calc'].astype(str) == res['Precinct_out'].astype(str))

    def _eq(a,b): 
        return (pd.isna(a) and pd.isna(b)) or (str(a)==str(b))
    res['AddrChanged_match'] = [_eq(a,b) for a,b in zip(res['AddrChanged_calc'], res['AddrChanged_out'])]
    return res

# ======== Main ========
def main():
    if len(sys.argv) != 5:
        print("Usage: python test.py <compiled_output_csv> <directory_path> <CNTY> <VUID>")
        sys.exit(1)

    output_csv = sys.argv[1]
    directory_root = sys.argv[2]
    cnty = sys.argv[3]
    vuid = sys.argv[4]

    # master sheet (must sit next to this script)
    _, meta_index = _load_master()

    # build single-voter timeline from raw files
    df_timeline = build_vuid_timeline(directory_root, cnty, vuid, meta_index)
    comp = compare_with_output(output_csv, vuid, df_timeline)

    if comp.empty:
        print(f"No observations found for VUID {vuid} in county {cnty}.")
        sys.exit(0)

    # Section 1: status/precinct/addrchanged summary
    print("\n=== Timeline for VUID", vuid, "(County", cnty, ") ===")
    print("{:9} | {:12} | {:10} | {:14} || {:12} | {:10} | {:14}".format(
        "Date","Status_calc","Precinct","AddrChanged_calc","Status_out","Precinct_out","AddrChanged_out"))
    for _, r in comp.iterrows():
        print("{:9} | {:12} | {:10} | {:14} || {:12} | {:10} | {:14}".format(
            r['Date'], str(r['Status_calc'])[:12], str(r['Precinct_calc'])[:10], str(r['AddrChanged_calc'])[:14],
            str(r['Status_out'])[:12], str(r['Precinct_out'])[:10], str(r['AddrChanged_out'])[:14]
        ))

    # Section 2: address-change flags + rebuilt addresses (visual inspection)
    print("\n==================")
    print("{:9} | {:16} | {:16} | {}".format("Date","AddrChanged_calc","AddrChanged_out","Address (rebuilt)"))
    for _, r in comp.iterrows():
        addr_disp = str(r['AddrDisplay']) if pd.notna(r['AddrDisplay']) else ''
        print("{:9} | {:16} | {:16} | {}".format(
            r['Date'], str(r['AddrChanged_calc']), str(r['AddrChanged_out']), addr_disp
        ))
        # Debug key if needed:
        # print("            key:", r['AddrKey'])

    out_path = f"test_{vuid}_CNTY{cnty}_timeline.csv"
    comp.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
