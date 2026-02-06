"""
SMART INCREMENTAL SYNC: Dropbox → GitHub (Tail-First Reading)
==============================================================
Reads from END of Dropbox file in reverse batches until hitting existing data.
Much faster than full file scan!
"""

import pandas as pd
import numpy as np
import os
import requests
from io import StringIO
import time

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Dropbox direct download link
DROPBOX_DIRECT_URL = os.getenv("DROPBOX_DIRECT_URL", "https://www.dropbox.com/scl/fo/1s4zle5elm98zsquvixl5/AMAwqTwpKu9e6hyNCYYoQLI/t20_bbb.csv?rlkey=5k98lxpc1dafyft5poa3bgh5e&st=rxk6baa0&dl=1")

DATASETS_DIR = "Datasets"
os.makedirs(DATASETS_DIR, exist_ok=True)

MIN_YEAR = 2026

# Competition mappings
COMPETITIONS_MAP = {
    "IPL": "IPL_APP_IPL.csv",
    "CPL": "IPL_APP_CPL.csv",
    "ILT20": "IPL_APP_ILT20.csv",
    "LPL": "IPL_APP_LPL.csv",
    "MLC": "IPL_APP_MLC.csv",
    "SA20": "IPL_APP_SA20.csv",
    "Super Smash": "IPL_APP_SUPER_SMASH.csv",
    "T20 Blast": "IPL_APP_T20_BLAST.csv",
    "T20I": "IPL_APP_T20I.csv",
    "BBL": "IPL_APP_BBL.csv",
}

BATCH_SIZE = 1000  # rows per batch
ID_COLS = ['p_match', 'innings', 'ball_id']  # Composite key


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: DOWNLOAD FILE
# ═══════════════════════════════════════════════════════════════════════════

def download_dropbox_file_to_temp():
    """Download Dropbox file to temp location for tail reading."""
    print("📥 Downloading Dropbox file...")
    
    temp_path = "/tmp/t20_bbb_temp.csv"
    
    try:
        response = requests.get(DROPBOX_DIRECT_URL, stream=True)
        response.raise_for_status()
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        print(f"✅ Downloaded {file_size_mb:.2f} MB")
        
        return temp_path
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: BUILD INDEX OF EXISTING DATA
# ═══════════════════════════════════════════════════════════════════════════

def build_github_index_per_competition():
    """Build set of existing (p_match, innings, ball_id) tuples per competition."""
    print("\n📊 Indexing existing GitHub data...")
    
    existing_index = {}
    
    for comp_name, filename in COMPETITIONS_MAP.items():
        filepath = os.path.join(DATASETS_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"  {comp_name}: No file (will create new)")
            existing_index[comp_name] = set()
            continue
        
        try:
            df = pd.read_csv(filepath, usecols=ID_COLS, low_memory=False)
            
            index_set = set()
            for _, row in df.iterrows():
                key = (
                    str(row.get('p_match', '')),
                    str(row.get('innings', '')),
                    str(row.get('ball_id', ''))
                )
                index_set.add(key)
            
            existing_index[comp_name] = index_set
            print(f"  {comp_name}: {len(index_set):,} rows indexed")
        
        except Exception as e:
            print(f"  {comp_name}: Error ({e})")
            existing_index[comp_name] = set()
    
    return existing_index


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: READ FROM TAIL IN REVERSE BATCHES
# ═══════════════════════════════════════════════════════════════════════════

def read_tail_until_existing_data(temp_filepath, existing_index):
    """
    Read from END of file in reverse batches.
    Stop when hitting a batch where ALL rows exist in GitHub.
    """
    print(f"\n🔄 Reading from tail ({BATCH_SIZE} rows/batch)...")
    
    # Get total rows
    total_rows = sum(1 for _ in open(temp_filepath)) - 1
    print(f"  Total rows: {total_rows:,}")
    
    # Get columns
    df_header = pd.read_csv(temp_filepath, nrows=0)
    all_columns = df_header.columns.tolist()
    
    # Validate
    for col in ID_COLS:
        if col not in all_columns:
            print(f"❌ Missing column: {col}")
            return pd.DataFrame()
    
    if 'competition' not in all_columns:
        print("❌ Missing column: competition")
        return pd.DataFrame()
    
    # Detect year column
    year_col = None
    for col in ['year', 'season', 'Year', 'Season']:
        if col in all_columns:
            year_col = col
            break
    
    # Read from tail
    accumulated_new_rows = []
    current_end = total_rows
    batch_num = 0
    
    while current_end > 0:
        batch_num += 1
        
        batch_start = max(0, current_end - BATCH_SIZE)
        batch_size = current_end - batch_start
        
        # Read batch
        batch_df = pd.read_csv(
            temp_filepath,
            skiprows=range(1, batch_start + 1),
            nrows=batch_size,
            low_memory=False
        )
        
        print(f"\n  Batch {batch_num}: rows {batch_start:,}-{current_end:,} ({len(batch_df):,})")
        
        if batch_df.empty:
            break
        
        # Filter by year
        if year_col:
            batch_df['_year_num'] = pd.to_numeric(batch_df[year_col], errors='coerce')
            batch_df = batch_df[batch_df['_year_num'] >= MIN_YEAR]
            batch_df = batch_df.drop(columns=['_year_num'])
        
        if batch_df.empty:
            print(f"    → Filtered out (year < {MIN_YEAR})")
            current_end = batch_start
            continue
        
        # Filter by competition
        batch_df = batch_df[batch_df['competition'].isin(COMPETITIONS_MAP.keys())]
        
        if batch_df.empty:
            print(f"    → No relevant competitions")
            current_end = batch_start
            continue
        
        # Check for new rows
        new_rows_in_batch = []
        all_exist = True
        
        for comp_name in COMPETITIONS_MAP.keys():
            comp_batch = batch_df[batch_df['competition'] == comp_name].copy()
            
            if comp_batch.empty:
                continue
            
            comp_index = existing_index[comp_name]
            
            for _, row in comp_batch.iterrows():
                row_key = (
                    str(row.get('p_match', '')),
                    str(row.get('innings', '')),
                    str(row.get('ball_id', ''))
                )
                
                if row_key not in comp_index:
                    new_rows_in_batch.append(row)
                    all_exist = False
        
        if new_rows_in_batch:
            print(f"    → {len(new_rows_in_batch):,} NEW rows")
            accumulated_new_rows.extend(new_rows_in_batch)
        else:
            print(f"    → All exist in GitHub")
        
        # STOP if all exist
        if all_exist:
            print(f"\n✅ Stopping: All rows before {batch_start:,} are synced")
            break
        
        current_end = batch_start
        
        if batch_num >= 100:
            print(f"\n⚠️  Safety limit: 100 batches")
            break
    
    # Combine
    if accumulated_new_rows:
        new_df = pd.DataFrame(accumulated_new_rows)
        print(f"\n✅ Total NEW rows: {len(new_df):,}")
        
        for comp_name in COMPETITIONS_MAP.keys():
            comp_count = (new_df['competition'] == comp_name).sum()
            if comp_count > 0:
                comp_df = new_df[new_df['competition'] == comp_name]
                print(f"  {comp_name}: {comp_count:,} rows (p_match {comp_df['p_match'].min()}→{comp_df['p_match'].max()})")
        
        return new_df
    else:
        print(f"\n✅ No new data!")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: ALIGN COLUMNS
# ═══════════════════════════════════════════════════════════════════════════

def align_columns_to_github(new_df, github_filepath):
    """Match columns to GitHub schema."""
    if not os.path.exists(github_filepath):
        return new_df
    
    try:
        github_df = pd.read_csv(github_filepath, nrows=0)
        github_cols = github_df.columns.tolist()
        
        aligned_df = new_df.reindex(columns=github_cols)
        
        extra = set(new_df.columns) - set(github_cols)
        if extra:
            print(f"    Dropped: {extra}")
        
        missing = set(github_cols) - set(new_df.columns)
        if missing:
            print(f"    Added (NaN): {missing}")
        
        return aligned_df
    
    except Exception as e:
        print(f"    ⚠️  Align error: {e}")
        return new_df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: APPEND TO FILES
# ═══════════════════════════════════════════════════════════════════════════

def append_to_github_files(new_df):
    """Split by competition and append."""
    if new_df.empty:
        return
    
    print("\n💾 Appending to GitHub files...")
    
    for comp_name, filename in COMPETITIONS_MAP.items():
        comp_df = new_df[new_df['competition'] == comp_name].copy()
        
        if comp_df.empty:
            continue
        
        filepath = os.path.join(DATASETS_DIR, filename)
        
        if os.path.exists(filepath):
            comp_df = align_columns_to_github(comp_df, filepath)
        
        # Sort by ID columns
        if all(col in comp_df.columns for col in ID_COLS):
            comp_df = comp_df.sort_values(by=ID_COLS)
        
        try:
            if os.path.exists(filepath):
                comp_df.to_csv(filepath, mode='a', header=False, index=False)
                print(f"  {comp_name}: ✅ +{len(comp_df):,} rows")
            else:
                comp_df.to_csv(filepath, mode='w', header=True, index=False)
                print(f"  {comp_name}: ✅ Created with {len(comp_df):,} rows")
        
        except Exception as e:
            print(f"  {comp_name}: ❌ {e}")


# ═══════════════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════════════

def cleanup_temp_files(temp_filepath):
    """Remove temp file."""
    try:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            print(f"\n🗑️  Cleaned up temp file")
    except:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("SMART TAIL-FIRST SYNC")
    print("=" * 80)
    
    start = time.time()
    
    temp_filepath = download_dropbox_file_to_temp()
    if not temp_filepath:
        return
    
    existing_index = build_github_index_per_competition()
    new_df = read_tail_until_existing_data(temp_filepath, existing_index)
    append_to_github_files(new_df)
    cleanup_temp_files(temp_filepath)
    
    print(f"\n{'=' * 80}")
    print(f"DONE ({time.time() - start:.1f}s)")
    print("=" * 80)


if __name__ == "__main__":
    main()
