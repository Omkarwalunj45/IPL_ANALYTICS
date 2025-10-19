import streamlit as st
import pandas as pd
import math as mt
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go  
import base64
from io import BytesIO
from PIL import Image
st.set_page_config(layout="wide")

# --- helper: render matplotlib fig as fixed-pixel-height image in Streamlit ---
from PIL import Image

def display_figure_fixed_height(fig, height_px=1200, bg='white'):
    """
    Save `fig` to buffer, open with PIL, resize to desired height (preserve aspect ratio),
    and display with st.image using exact pixel dimensions (no auto-scaling).
    - height_px: desired displayed height in pixels (e.g. 1800)
    """
    buf = BytesIO()
    # Save at a high DPI so the saved image is high-res (avoids blur when resizing)
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).convert('RGBA')

    # preserve aspect ratio, compute new width
    w, h = img.size
    if h == 0:
        st.error("Figure saved with zero height unexpectedly.")
        return
    new_h = int(height_px)
    new_w = int(round((w / h) * new_h))

    # resize using LANCZOS for quality
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # if you prefer a white background (avoid transparency), compose
    if bg is not None:
        bg_img = Image.new('RGB', img_resized.size, bg)
        bg_img.paste(img_resized, mask=img_resized.split()[3] if img_resized.mode == 'RGBA' else None)
        img_resized = bg_img

    # display with explicit width (same as new_w) so Streamlit doesn't auto-scale
    st.image(img_resized, use_column_width=False, width=new_w)
    plt.close(fig)

def display_figure_fixed_height_html(fig, height_px=1200, bg='white', container_id=None):
    """
    Save fig to buffer, encode to base64, and embed using HTML <img> with fixed height in pixels.
    This forces the browser to render the exact height (no Streamlit autoscaling).
    - height_px: desired displayed height in pixels (e.g. 1800)
    - bg: background color for composition to remove transparency
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).convert('RGBA')

    # Compose over background if required (avoid transparency artifacts)
    if bg is not None:
        bg_img = Image.new('RGB', img.size, bg)
        bg_img.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
        img = bg_img
    # encode to base64
    out_buf = BytesIO()
    img.save(out_buf, format='PNG')
    b64 = base64.b64encode(out_buf.getvalue()).decode('ascii')

    # create HTML; width:auto keeps aspect ratio while height is forced
    html = f'<img src="data:image/png;base64,{b64}" style="height:{int(height_px)}px; width:auto; display:block; margin-left:auto; margin-right:auto;" />'
    # Optionally wrap in a container div with max-width:100% to avoid horizontal overflow
    wrapper = f'<div style="max-width:100%;">{html}</div>'
    st.markdown(wrapper, unsafe_allow_html=True)
    plt.close(fig)


st.set_page_config(page_title='IPL Performance Analysis Portal', layout='wide')
st.title('IPL Performance Analysis Portal')
@st.cache_data
def load_data():
    path = "Datasets/ipl_bbb_21_25.xlsx"
    df = pd.read_excel(path)
    return df

df = load_data()    
def rename_rcb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames 'Royal Challengers Bangalore' to 'Royal Challengers Bengaluru' in team_bat, team_bowl, and winner columns.
    Returns the modified DataFrame.
    """
    d = df.copy()
    
    # List of columns to check and rename
    columns_to_rename = ['team_bat', 'team_bowl', 'winner']
    
    # Rename 'Royal Challengers Bangalore' to 'Royal Challengers Bengaluru' in specified columns
    for col in columns_to_rename:
        if col in d.columns:
            d[col] = d[col].replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')
    
    return d
df = rename_rcb(df)
df['is_wicket'] = df['out'].astype(int)
df['venue']=df['ground']
df['batsman']=df['bat']
df['bowler']=df['bowl']

# -----------------------
# Utility helpers
# -----------------------
def safe_get_col(df: pd.DataFrame, choices, default=None):
    for c in choices:
        if c in df.columns:
            return c
    return default

def round_up_floats(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col].dtype):
            df[col] = df[col].round(decimals)
    return df
def bpd(balls, dismissals):
    return balls / dismissals if dismissals > 0 else np.nan

def bpb(balls, boundaries):
    return balls / boundaries if boundaries > 0 else np.nan

def bp6(balls, sixes):
    return balls / sixes if sixes > 0 else np.nan

def bp4(balls, fours):
    return balls / fours if fours > 0 else np.nan

def avg(runs, dismissals, innings):
    return runs / dismissals if dismissals > 0 else np.nan

def categorize_phase(over):
    if over < 6:
        return 'Powerplay'
    elif over < 15:
        return 'Middle'
    else:
        return 'Death'

def bpd(balls, dismissals):
    return balls / dismissals if dismissals > 0 else np.nan

def bpb(balls, boundaries):
    return balls / boundaries if boundaries > 0 else np.nan

def bp6(balls, sixes):
    return balls / sixes if sixes > 0 else np.nan

def bp4(balls, fours):
    return balls / fours if fours > 0 else np.nan

def avg(runs, dismissals, innings):
    return runs / dismissals if dismissals > 0 else np.nan

def categorize_phase(over):
    if over < 6:
        return 'Powerplay'
    elif over < 15:
        return 'Middle'
    else:
        return 'Death'

def cumulator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Batting summary builder implementing:
      - legal_ball: both wide & noball == 0
      - 50s counted only if innings score >=50 and <100
      - Dismissal logic: non-striker is last different batsman in same inns and p_match with out=False
      - No dismissal attribution if non-striker invalid or already dismissed
    Returns bat_rec (one row per batsman).
    """
    if df is None:
        return pd.DataFrame()
    d = df.copy()

    # ---- normalize column names ----
    if 'p_match' in d.columns:
        d = d.rename(columns={'p_match': 'match_id'})
    if 'inns' in d.columns and 'inning' not in d.columns:
        d = d.rename(columns={'inns': 'inning'})
    if 'bat' in d.columns and 'batsman' not in d.columns:
        d = d.rename(columns={'bat': 'batsman'})
    if 'team_bat' in d.columns and 'batting_team' not in d.columns:
        d = d.rename(columns={'team_bat': 'batting_team'})
    if 'batruns' in d.columns and 'runs_off_bat' not in d.columns:
        d = d.rename(columns={'batruns': 'runs_off_bat'})
    elif 'score' in d.columns and 'runs_off_bat' not in d.columns:
        d = d.rename(columns={'score': 'runs_off_bat'})
    elif 'runs_off_bat' not in d.columns:
        d['runs_off_bat'] = 0

    # ensure stable RangeIndex
    d.index = pd.RangeIndex(len(d))

    # ---- safe ball ordering ----
    if 'ball_id' in d.columns:
        tmp = pd.to_numeric(d['ball_id'], errors='coerce')
        seq = pd.Series(np.arange(len(d)), index=d.index)
        tmp = tmp.fillna(seq)
        d['__ball_sort__'] = tmp
    else:
        d['__ball_sort__'] = pd.Series(np.arange(len(d)), index=d.index)

    # ---- dismissal normalization & flags ----
    special_runout_types = set(['run out', 'obstructing the field', 'retired out', 'retired not out (hurt)'])
    d['dismissal_clean'] = d.get('dismissal', '').astype(str).str.lower().str.strip()
    d['dismissal_clean'] = d['dismissal_clean'].replace({'nan': '', 'none': ''})

    # p_bat and p_out are integers, handle missing as NaN
    d['p_bat_num'] = d.get('p_bat', pd.Series(np.nan, index=d.index)).astype(float)
    d['p_out_num'] = d.get('p_out', pd.Series(np.nan, index=d.index)).astype(float)

    # out is boolean (True/False), convert to 0/1
    d['out_flag'] = d.get('out', False).astype(int)

    # ensure match_id exists
    if 'match_id' not in d.columns:
        d['match_id'] = 0

    # sort by match, inning, and ball order
    d.sort_values(['match_id', 'inning', '__ball_sort__'], inplace=True, kind='stable')
    d.reset_index(drop=True, inplace=True)

    # initialize dismissal outputs
    d['dismissed_player'] = None
    d['bowler_wkt'] = 0

    # resolve dismissals per specified logic
    for match in d['match_id'].unique():
        for inning in d[d['match_id'] == match]['inning'].unique():
            idxs = d[(d['match_id'] == match) & (d['inning'] == inning)].index.tolist()
            idxs = sorted(idxs, key=lambda i: d.at[i, '__ball_sort__'])
            for pos, i in enumerate(idxs):
                if d.at[i, 'out_flag'] != 1:  # Check if out=True
                    continue

                disc = (d.at[i, 'dismissal_clean'] or '').strip()
                striker = d.at[i, 'batsman'] if 'batsman' in d.columns else None

                # Rule 1: out=True and dismissal not in [blank, nan, special] -> striker out, bowler credit
                if disc and disc not in special_runout_types:
                    d.at[i, 'dismissed_player'] = striker
                    d.at[i, 'bowler_wkt'] = 1
                    continue

                # Rule 2: out=True and dismissal in special
                if disc in special_runout_types:
                    pbat = d.at[i, 'p_bat_num']
                    pout = d.at[i, 'p_out_num']

                    if (not pd.isna(pbat)) and (not pd.isna(pout)) and (pbat == pout):
                        d.at[i, 'dismissed_player'] = striker
                        d.at[i, 'bowler_wkt'] = 0
                        continue

                    # Nonstriker dismissed: find last different batter in same match and inning
                    nonstriker = None
                    last_idx_of_nonstriker = None
                    for j in reversed(idxs[:pos]):
                        prev_bat = d.at[j, 'batsman'] if 'batsman' in d.columns else None
                        if prev_bat is not None and prev_bat != striker:
                            nonstriker = prev_bat
                            last_idx_of_nonstriker = j
                            break

                    if nonstriker is None:
                        # No valid nonstriker: do not attribute dismissal
                        d.at[i, 'dismissed_player'] = None
                        d.at[i, 'bowler_wkt'] = 0
                        continue

                    prev_out_flag = d.at[last_idx_of_nonstriker, 'out_flag'] if last_idx_of_nonstriker is not None else 0
                    if prev_out_flag == 0:
                        d.at[i, 'dismissed_player'] = nonstriker
                        d.at[i, 'bowler_wkt'] = 0
                    else:
                        # Nonstriker already dismissed: do not attribute dismissal
                        d.at[i, 'dismissed_player'] = None
                        d.at[i, 'bowler_wkt'] = 0

    # ---- compute per-delivery summaries ----
    d['cur_bat_runs'] = pd.to_numeric(d.get('cur_bat_runs', 0), errors='coerce').fillna(0).astype(int)
    d['cur_bat_bf'] = pd.to_numeric(d.get('cur_bat_bf', 0), errors='coerce').fillna(0).astype(int)

    # legal ball: both wide & noball must be 0
    d['noball'] = pd.to_numeric(d.get('noball', 0), errors='coerce').fillna(0).astype(int)
    d['wide'] = pd.to_numeric(d.get('wide', 0), errors='coerce').fillna(0).astype(int)
    d['legal_ball'] = ((d['noball'] == 0) & (d['wide'] == 0)).astype(int)

    # per-delivery run flags
    d['runs_off_bat'] = pd.to_numeric(d.get('runs_off_bat', 0), errors='coerce').fillna(0).astype(int)
    d['is_dot'] = ((d['runs_off_bat'] == 0) & (d['legal_ball'] == 1)).astype(int)
    d['is_one'] = (d['runs_off_bat'] == 1).astype(int)
    d['is_two'] = (d['runs_off_bat'] == 2).astype(int)
    d['is_three'] = (d['runs_off_bat'] == 3).astype(int)
    d['is_four'] = (d['runs_off_bat'] == 4).astype(int)
    d['is_six'] = (d['runs_off_bat'] == 6).astype(int)

    # last snapshot per batsman per match
    last_bat_snapshot = (
        d.groupby(['batsman', 'match_id'], sort=False, as_index=False)
         .agg({'cur_bat_runs': 'last', 'cur_bat_bf': 'last'})
         .rename(columns={'cur_bat_runs': 'match_runs', 'cur_bat_bf': 'match_balls'})
    )

    runs_per_match = last_bat_snapshot[['batsman', 'match_runs', 'match_balls', 'match_id']].copy()
    innings_count = runs_per_match.groupby('batsman')['match_id'].nunique().reset_index().rename(columns={'match_id': 'innings'})
    total_runs = runs_per_match.groupby('batsman')['match_runs'].sum().reset_index().rename(columns={'match_runs': 'runs'})
    total_balls = runs_per_match.groupby('batsman')['match_balls'].sum().reset_index().rename(columns={'match_balls': 'balls'})

    # dismissals: count per resolved dismissed_player, ensuring unique dismissals
    dismissals_df = d[d['dismissed_player'].notna()].groupby(['dismissed_player', 'match_id', 'inning']).size().reset_index(name='dismissals')
    dismissals_df = dismissals_df.groupby('dismissed_player')['dismissals'].sum().reset_index().rename(columns={'dismissed_player': 'batsman'})

    # boundary & running counts
    fours = d.groupby('batsman')['is_four'].sum().reset_index().rename(columns={'is_four': 'fours'})
    sixes = d.groupby('batsman')['is_six'].sum().reset_index().rename(columns={'is_six': 'sixes'})
    dots = d.groupby('batsman')['is_dot'].sum().reset_index().rename(columns={'is_dot': 'dots'})
    ones = d.groupby('batsman')['is_one'].sum().reset_index().rename(columns={'is_one': 'ones'})
    twos = d.groupby('batsman')['is_two'].sum().reset_index().rename(columns={'is_two': 'twos'})
    threes = d.groupby('batsman')['is_three'].sum().reset_index().rename(columns={'is_three': 'threes'})

    # match-level thresholds: 30s (30-49), 50s (50-99), 100s (>=100)
    thirties = runs_per_match[(runs_per_match['match_runs'] >= 30) & (runs_per_match['match_runs'] < 50)].groupby('batsman').size().reset_index(name='30s')
    fifties = runs_per_match[(runs_per_match['match_runs'] >= 50) & (runs_per_match['match_runs'] < 100)].groupby('batsman').size().reset_index(name='50s')
    hundreds = runs_per_match[runs_per_match['match_runs'] >= 100].groupby('batsman').size().reset_index(name='100s')

    highest_score = runs_per_match.groupby('batsman')['match_runs'].max().reset_index().rename(columns={'match_runs': 'HS'})
    median_runs = runs_per_match.groupby('batsman')['match_runs'].median().reset_index().rename(columns={'match_runs': 'median'})

    boundary_runs = (d.groupby('batsman').apply(lambda x: int((x['is_four'] * 4).sum() + (x['is_six'] * 6).sum()))
                     .reset_index(name='boundary_runs'))
    running_runs = (d.groupby('batsman').apply(lambda x: int((x['is_one'] * 1).sum() + (x['is_two'] * 2).sum() + (x['is_three'] * 3).sum()))
                    .reset_index(name='running_runs'))

    # Merge master batting record
    bat_rec = innings_count.merge(total_runs, on='batsman', how='left')
    bat_rec = bat_rec.merge(total_balls, on='batsman', how='left')
    bat_rec = bat_rec.merge(dismissals_df, on='batsman', how='left')
    bat_rec = bat_rec.merge(sixes, on='batsman', how='left')
    bat_rec = bat_rec.merge(fours, on='batsman', how='left')
    bat_rec = bat_rec.merge(dots, on='batsman', how='left')
    bat_rec = bat_rec.merge(ones, on='batsman', how='left')
    bat_rec = bat_rec.merge(twos, on='batsman', how='left')
    bat_rec = bat_rec.merge(threes, on='batsman', how='left')
    bat_rec = bat_rec.merge(boundary_runs, on='batsman', how='left')
    bat_rec = bat_rec.merge(running_runs, on='batsman', how='left')
    bat_rec = bat_rec.merge(thirties, on='batsman', how='left')
    bat_rec = bat_rec.merge(fifties, on='batsman', how='left')
    bat_rec = bat_rec.merge(hundreds, on='batsman', how='left')
    bat_rec = bat_rec.merge(highest_score, on='batsman', how='left')
    bat_rec = bat_rec.merge(median_runs, on='batsman', how='left')

    # fill NaNs & cast types
    fill_zero_cols = ['30s', '50s', '100s', 'runs', 'balls', 'dismissals', 'sixes', 'fours',
                      'dots', 'ones', 'twos', 'threes', 'boundary_runs', 'running_runs', 'HS', 'median']
    for col in fill_zero_cols:
        if col in bat_rec.columns:
            bat_rec[col] = bat_rec[col].fillna(0)

    int_cols = ['30s', '50s', '100s', 'runs', 'balls', 'dismissals', 'sixes', 'fours',
                'dots', 'ones', 'twos', 'threes', 'boundary_runs', 'running_runs']
    for col in int_cols:
        if col in bat_rec.columns:
            bat_rec[col] = bat_rec[col].astype(int)

    # basic ratios & metrics
    bat_rec['RPI'] = bat_rec.apply(lambda x: (x['runs'] / x['innings']) if x['innings'] > 0 else np.nan, axis=1)
    bat_rec['SR'] = bat_rec.apply(lambda x: (x['runs'] / x['balls'] * 100) if x['balls'] > 0 else np.nan, axis=1)
    bat_rec['BPD'] = bat_rec.apply(lambda x: bpd(x['balls'], x['dismissals']), axis=1)
    bat_rec['BPB'] = bat_rec.apply(lambda x: bpb(x['balls'], (x.get('fours', 0) + x.get('sixes', 0))), axis=1)
    bat_rec['BP6'] = bat_rec.apply(lambda x: bp6(x['balls'], x.get('sixes', 0)), axis=1)
    bat_rec['BP4'] = bat_rec.apply(lambda x: bp4(x['balls'], x.get('fours', 0)), axis=1)

    def compute_nbdry_sr(row):
        run_count = (row.get('dots', 0) * 0 + row.get('ones', 0) * 1 + row.get('twos', 0) * 2 + row.get('threes', 0) * 3)
        denom = (row.get('dots', 0) + row.get('ones', 0) + row.get('twos', 0) + row.get('threes', 0))
        return (run_count / denom * 100) if denom > 0 else 0
    bat_rec['nbdry_sr'] = bat_rec.apply(compute_nbdry_sr, axis=1)

    bat_rec['AVG'] = bat_rec.apply(lambda x: avg(x['runs'], x.get('dismissals', 0), x['innings']), axis=1)
    bat_rec['dot_percentage'] = bat_rec.apply(lambda x: (x['dots'] / x['balls'] * 100) if x['balls'] > 0 else 0, axis=1)
    bat_rec['Bdry%'] = bat_rec.apply(lambda x: (x['boundary_runs'] / x['runs'] * 100) if x['runs'] > 0 else 0, axis=1)
    bat_rec['Running%'] = bat_rec.apply(lambda x: (x['running_runs'] / x['runs'] * 100) if x['runs'] > 0 else 0, axis=1)

    # latest team
    if 'batting_team' in d.columns:
        latest_team = (d.sort_values(['match_id', 'inning', '__ball_sort__'])
                       .drop_duplicates(subset=['batsman'], keep='last')
                       [['batsman', 'batting_team']])
        bat_rec = bat_rec.merge(latest_team, on='batsman', how='left')
    else:
        bat_rec['batting_team'] = np.nan

    # phase-wise aggregation
    if 'over' in d.columns:
        d['phase'] = d['over'].apply(categorize_phase)
    else:
        d['phase'] = 'Unknown'

    phase_stats = d.groupby(['batsman', 'phase']).agg({
        'runs_off_bat': 'sum',
        'legal_ball': 'sum',
        'is_dot': 'sum',
        'is_four': 'sum',
        'is_six': 'sum',
        'match_id': 'nunique',
    }).reset_index()

    phase_stats.rename(columns={
        'runs_off_bat': 'Runs',
        'legal_ball': 'Balls',
        'is_dot': 'Dots',
        'is_four': 'Fours',
        'is_six': 'Sixes',
        'match_id': 'Innings'
    }, inplace=True)

    # Add phase dismissals
    phase_dismissals = d[d['dismissed_player'].notna()].groupby(['dismissed_player', 'phase', 'match_id', 'inning']).size().reset_index(name='Dismissals')
    phase_dismissals = phase_dismissals.groupby(['dismissed_player', 'phase'])['Dismissals'].sum().reset_index()
    phase_dismissals.rename(columns={'dismissed_player': 'batsman'}, inplace=True)
    phase_stats = phase_stats.merge(phase_dismissals, on=['batsman', 'phase'], how='left')
    phase_stats['Dismissals'] = phase_stats['Dismissals'].fillna(0).astype(int)

    phase_stats['BPB'] = phase_stats.apply(lambda x: bpb(x['Balls'], (x['Fours'] + x['Sixes'])), axis=1)
    phase_stats['BPD'] = phase_stats.apply(lambda x: bpd(x['Balls'], x['Dismissals']), axis=1)
    phase_stats['SR'] = phase_stats.apply(lambda x: (x['Runs'] / x['Balls'] * 100) if x['Balls'] > 0 else 0, axis=1)
    phase_stats['AVG'] = phase_stats.apply(lambda x: avg(x['Runs'], x['Dismissals'], x['Innings']), axis=1)
    phase_stats['DOT%'] = phase_stats.apply(lambda x: (x['Dots'] / x['Balls'] * 100) if x['Balls'] > 0 else 0, axis=1)

    if not phase_stats.empty:
        phase_pivot = phase_stats.pivot(index='batsman', columns='phase',
                                        values=['SR', 'AVG', 'DOT%', 'BPB', 'BPD', 'Innings', 'Runs', 'Balls'])
        if isinstance(phase_pivot.columns, pd.MultiIndex):
            phase_pivot.columns = [f"{col[1]}_{col[0]}" for col in phase_pivot.columns]
        phase_pivot.reset_index(inplace=True)
    else:
        phase_pivot = pd.DataFrame({'batsman': []})

    bat_rec = bat_rec.merge(phase_pivot, on='batsman', how='left')

    bat_rec.reset_index(drop=True, inplace=True)
    return bat_rec
# bowlerstat - bowling summary with exact dismissal rules
# -------------------------
def bowlerstat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build bowling summary and apply the same dismissal resolution rules as Custom.
      - legal_ball requires both wide and noball be zero
      - bowler wicket credit only when dismissal is non-special (i.e., caught, bowled, lbw, stumped etc.)
    Returns bowl_rec DataFrame.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    d = df.copy()

    # normalize names
    if 'p_match' in d.columns:
        d = d.rename(columns={'p_match': 'match_id'})
    if 'bowl' not in d.columns:
        d['bowl'] = None
    # prefer ball_id or ball order
    if 'ball_id' in d.columns:
        tmp = pd.to_numeric(d['ball_id'], errors='coerce')
        seq = pd.Series(np.arange(len(d)), index=d.index)
        tmp = tmp.fillna(seq)
        d['__ball_sort__'] = tmp
    elif 'ball' in d.columns:
        tmp = pd.to_numeric(d['ball'], errors='coerce')
        seq = pd.Series(np.arange(len(d)), index=d.index)
        tmp = tmp.fillna(seq)
        d['__ball_sort__'] = tmp
    else:
        d['__ball_sort__'] = pd.Series(np.arange(len(d)), index=d.index)

    # legal_ball
    d['noball'] = pd.to_numeric(d.get('noball', 0), errors='coerce').fillna(0).astype(int)
    d['wide'] = pd.to_numeric(d.get('wide', 0), errors='coerce').fillna(0).astype(int)
    d['legal_ball'] = ((d['noball'] == 0) & (d['wide'] == 0)).astype(int)

    # batsman_runs (batruns or score)
    if 'batruns' in d.columns:
        d['batsman_runs'] = pd.to_numeric(d['batruns'], errors='coerce').fillna(0).astype(int)
    elif 'score' in d.columns:
        d['batsman_runs'] = pd.to_numeric(d['score'], errors='coerce').fillna(0).astype(int)
    else:
        d['batsman_runs'] = 0

    # dismissal normalization (same rules)
    d['dismissal_clean'] = d.get('dismissal', pd.Series([None]*len(d))).astype(str).str.lower().str.strip()
    d['dismissal_clean'] = d['dismissal_clean'].replace({'nan':'', 'none':''})
    special_runout_types = set([
        'run out', 'runout',
        'obstructing the field', 'obstructing thefield', 'obstructing',
        'retired out', 'retired not out (hurt)', 'retired not out', 'retired'
    ])

    d['p_bat_num'] = pd.to_numeric(d.get('p_bat', np.nan), errors='coerce')
    d['p_out_num'] = pd.to_numeric(d.get('p_out', np.nan), errors='coerce')
    d['out_flag'] = pd.to_numeric(d.get('out', 0), errors='coerce').fillna(0).astype(int)

    if 'match_id' not in d.columns:
        d['match_id'] = 0
    d.sort_values(['match_id', '__ball_sort__'], inplace=True, kind='stable')
    d.reset_index(drop=True, inplace=True)

    # initialize resolved fields
    d['dismissed_player'] = None
    d['bowler_wkt'] = 0

    # resolve dismissals per match using same exact logic
    for m in d['match_id'].unique():
        idxs = d.index[d['match_id'] == m].tolist()
        idxs = sorted(idxs, key=lambda i: d.at[i, '__ball_sort__'])
        for pos, i in enumerate(idxs):
            out_flag = int(d.at[i, 'out_flag']) if not pd.isna(d.at[i, 'out_flag']) else 0
            disc = (d.at[i, 'dismissal_clean'] or '').strip()
            striker = d.at[i, 'bat'] if 'bat' in d.columns else None
            # If out_flag True:
            if out_flag == 1:
                if disc and (disc not in special_runout_types):
                    d.at[i, 'dismissed_player'] = striker
                    d.at[i, 'bowler_wkt'] = 1
                    continue
                pbat = d.at[i, 'p_bat_num']
                pout = d.at[i, 'p_out_num']
                if (not pd.isna(pbat)) and (not pd.isna(pout)) and (pbat == pout):
                    d.at[i, 'dismissed_player'] = striker
                    d.at[i, 'bowler_wkt'] = 0
                    continue
                # find nonstriker
                nonstriker = None
                last_idx_of_nonstriker = None
                for j in reversed(idxs[:pos]):
                    prev_bat = d.at[j, 'bat'] if 'bat' in d.columns else None
                    if prev_bat is not None and prev_bat != striker:
                        nonstriker = prev_bat
                        last_idx_of_nonstriker = j
                        break
                if nonstriker is None:
                    d.at[i, 'dismissed_player'] = striker
                    d.at[i, 'bowler_wkt'] = 0
                    continue
                prev_out_flag = int(d.at[last_idx_of_nonstriker, 'out_flag']) if last_idx_of_nonstriker is not None else 0
                if prev_out_flag == 0:
                    d.at[i, 'dismissed_player'] = nonstriker
                    d.at[i, 'bowler_wkt'] = 0
                else:
                    d.at[i, 'dismissed_player'] = striker
                    d.at[i, 'bowler_wkt'] = 0
            else:
                # If out==0 => mark striker dismissed per instruction
                d.at[i, 'dismissed_player'] = striker
                d.at[i, 'bowler_wkt'] = 0

    # Now aggregate bowler-level stats using bowler_wkt
    runs = d.groupby('bowl')['batsman_runs'].sum().reset_index().rename(columns={'batsman_runs': 'runs', 'bowl':'bowler'})
    innings = d.groupby('bowl')['match_id'].nunique().reset_index().rename(columns={'match_id': 'innings', 'bowl':'bowler'})
    balls = d.groupby('bowl')['legal_ball'].sum().reset_index().rename(columns={'legal_ball': 'balls', 'bowl':'bowler'})
    wkts = d.groupby('bowl')['bowler_wkt'].sum().reset_index().rename(columns={'bowler_wkt': 'wkts', 'bowl':'bowler'})
    dots = d.groupby('bowl')['is_dot'].sum().reset_index().rename(columns={'is_dot': 'dots', 'bowl':'bowler'})
    fours = d.groupby('bowl')['is_four'].sum().reset_index().rename(columns={'is_four': 'fours', 'bowl':'bowler'})
    sixes = d.groupby('bowl')['is_six'].sum().reset_index().rename(columns={'is_six': 'sixes', 'bowl':'bowler'})

    dismissals_count = d.groupby(['bowl', 'match_id'])['bowler_wkt'].sum().reset_index(name='wkts_in_match')
    three_wicket_hauls = dismissals_count[dismissals_count['wkts_in_match'] >= 3].groupby('bowl').size().reset_index(name='three_wicket_hauls').rename(columns={'bowl':'bowler'})
    bbi = dismissals_count.groupby('bowl')['wkts_in_match'].max().reset_index().rename(columns={'wkts_in_match': 'bbi', 'bowl':'bowler'})

    # over_num extraction
    if 'over' in d.columns:
        try:
            d['over_num'] = pd.to_numeric(d['over'], errors='coerce').fillna(0).astype(int)
        except Exception:
            d['over_num'] = d['over'].astype(str).str.split('.').str[0].astype(int)
    else:
        d['over_num'] = 0

    over_agg = d.groupby(['bowl', 'match_id', 'over_num']).agg(
        balls_in_over=('legal_ball', 'sum'),
        runs_in_over=('bowlruns', 'sum' if 'bowlruns' in d.columns else 'sum')
    ).reset_index()

    # safe runs_in_over (if above syntax caused issues fallback)
    if 'runs_in_over' not in over_agg.columns:
        over_agg = d.groupby(['bowl', 'match_id', 'over_num']).agg(
            balls_in_over=('legal_ball', 'sum'),
            runs_in_over=('batsman_runs', 'sum')
        ).reset_index()

    maiden_overs_count = over_agg[(over_agg['balls_in_over'] == 6) & (over_agg['runs_in_over'] == 0)].groupby('bowl').size().reset_index(name='maiden_overs').rename(columns={'bowl':'bowler'})

    # phase grouping
    if 'over' in d.columns:
        d['phase'] = d['over'].apply(categorize_phase)
    else:
        d['phase'] = 'Unknown'
    phase_group = d.groupby(['bowl', 'phase']).agg(
        phase_balls=('legal_ball','sum'),
        phase_runs=('batsman_runs','sum'),
        phase_wkts=('bowler_wkt','sum'),
        phase_dots=('is_dot','sum'),
        phase_innings=('match_id','nunique')
    ).reset_index().rename(columns={'bowl':'bowler'})

    def pivot_metric(df_pg, metric):
        if df_pg.empty:
            return pd.DataFrame({'bowler':[]})
        pivoted = df_pg.pivot(index='bowler', columns='phase', values=metric).fillna(0)
        for ph in ['Powerplay','Middle1','Middle2','Death']:
            if ph not in pivoted.columns:
                pivoted[ph] = 0
        pivoted = pivoted.rename(columns={ph: f"{metric}_{ph}" for ph in pivoted.columns})
        pivoted = pivoted.reset_index()
        return pivoted

    pb = pivot_metric(phase_group, 'phase_balls')
    pr = pivot_metric(phase_group, 'phase_runs')
    pw = pivot_metric(phase_group, 'phase_wkts')
    pdot = pivot_metric(phase_group, 'phase_dots')
    pi = pivot_metric(phase_group, 'phase_innings')

    # merge everything
    # rename frames to common 'bowler' index
    frames = [innings.rename(columns={'bowl':'bowler'}) if 'bowl' in innings.columns else innings,
              balls.rename(columns={'bowl':'bowler'}) if 'bowl' in balls.columns else balls,
              runs.rename(columns={'bowl':'bowler'}) if 'bowl' in runs.columns else runs,
              wkts.rename(columns={'bowl':'bowler'}) if 'bowl' in wkts.columns else wkts,
              sixes.rename(columns={'bowl':'bowler'}) if 'bowl' in sixes.columns else sixes,
              fours.rename(columns={'bowl':'bowler'}) if 'bowl' in fours.columns else fours,
              dots.rename(columns={'bowl':'bowler'}) if 'bowl' in dots.columns else dots,
              three_wicket_hauls, maiden_overs_count, bbi, pb, pr, pw, pdot, pi]

    bowl_rec = None
    for fr in frames:
        if fr is None or fr.empty:
            continue
        if bowl_rec is None:
            bowl_rec = fr.copy()
        else:
            bowl_rec = bowl_rec.merge(fr, on='bowler', how='outer')

    if bowl_rec is None:
        bowl_rec = pd.DataFrame(columns=['bowler'])

    # fill NaNs and finalize numerical fields
    for col in ['innings','balls','runs','wkts','sixes','fours','dots','three_wicket_hauls','maiden_overs','bbi','Mega_Over_Count']:
        if col in bowl_rec.columns:
            bowl_rec[col] = pd.to_numeric(bowl_rec[col], errors='coerce').fillna(0)

    bowl_rec['dot%'] = bowl_rec.apply(lambda r: (r['dots'] / r['balls'] * 100) if r.get('balls',0) > 0 else np.nan, axis=1)
    bowl_rec['avg'] = bowl_rec.apply(lambda r: (r['runs'] / r['wkts']) if r.get('wkts',0) > 0 else np.nan, axis=1)
    bowl_rec['sr'] = bowl_rec.apply(lambda r: (r['balls'] / r['wkts']) if r.get('wkts',0) > 0 else np.nan, axis=1)
    bowl_rec['econ'] = bowl_rec.apply(lambda r: (r['runs'] * 6.0 / r['balls']) if r.get('balls',0) > 0 else np.nan, axis=1)
    bowl_rec['WPI'] = bowl_rec.apply(lambda r: (r['wkts'] / r['innings']) if r.get('innings',0) > 0 else np.nan, axis=1)
    bowl_rec['RPI'] = bowl_rec.apply(lambda r: (r['runs'] / r['innings']) if r.get('innings',0) > 0 else np.nan, axis=1)

    bowl_rec.reset_index(drop=True, inplace=True)
    return bowl_rec


import pandas as pd
import numpy as np

def categorize_phase(over):
    try:
        o = float(over)
    except Exception:
        return "Unknown"
    if o <= 6:
        return "Powerplay"
    if 6 < o <= 11:
        return "Middle 1"
    if 11 < o <= 16:
        return "Middle 2"
    return "Death"

def bowlerstat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bowler aggregation adapted to the df columns you provided.

    - Runs (for bowler) = sum of 'score' (or fallback) ONLY where byes == 0 and legbyes == 0
    - BBI formatted as 'wkts/runs' (prefers match with max wkts, tie-breaker min runs)
    - Bowler wicket credit only for non-run-out-like dismissals (per rules)
    - legal_ball requires both wide & noball == 0
    - Round float metrics to 2 decimals at the end
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    d = df.copy()

    # ---- normalize minimal column names used in this function ----
    if 'p_match' in d.columns:
        d = d.rename(columns={'p_match': 'match_id'})
    if 'inns' in d.columns and 'inning' not in d.columns:
        d = d.rename(columns={'inns': 'inning'})
    if 'bowl' in d.columns and 'bowler' not in d.columns:
        d = d.rename(columns={'bowl': 'bowler'})
    if 'ball_id' in d.columns and 'ball' not in d.columns:
        d = d.rename(columns={'ball_id': 'ball'})
    # prefer explicit per-ball batsman runs field
    if 'batruns' in d.columns and 'batsman_runs' not in d.columns:
        d = d.rename(columns={'batruns': 'batsman_runs'})
    elif 'score' in d.columns and 'batsman_runs' not in d.columns:
        d = d.rename(columns={'score': 'batsman_runs'})
    if 'batsman_runs' not in d.columns:
        d['batsman_runs'] = 0

    # ---- ensure numeric extras / score fields ----
    # Score used for bowler runs calculation: prefer 'score' column if present else fallback to batsman_runs
    d['score_num'] = pd.to_numeric(d.get('score', d.get('batsman_runs', 0)), errors='coerce').fillna(0).astype(int)

    d['byes'] = pd.to_numeric(d.get('byes', 0), errors='coerce').fillna(0).astype(int)
    d['legbyes'] = pd.to_numeric(d.get('legbyes', 0), errors='coerce').fillna(0).astype(int)
    d['noball'] = pd.to_numeric(d.get('noball', 0), errors='coerce').fillna(0).astype(int)
    d['wide']   = pd.to_numeric(d.get('wide', 0), errors='coerce').fillna(0).astype(int)

    # Runs counted for bowler: score_num only when byes==0 and legbyes==0
    d['runs_for_bowler'] = (d['score_num'] * ((d['byes'] == 0) & (d['legbyes'] == 0)).astype(int))

    # total_runs fallback (useful for over-run aggregates)
    if 'bowlruns' in d.columns and 'total_runs' not in d.columns:
        d = d.rename(columns={'bowlruns': 'total_runs'})
    if 'total_runs' not in d.columns:
        d['total_runs'] = pd.to_numeric(d.get('batsman_runs', 0), errors='coerce').fillna(0).astype(int) + d['byes'] + d['legbyes'] + d['noball'] + d['wide']

    # ---- legal ball: both wide & noball must be 0 ----
    d['legal_ball'] = ((d['noball'] == 0) & (d['wide'] == 0)).astype(int)

    # ---- per-delivery batsman_run flags ----
    d['batsman_runs'] = pd.to_numeric(d.get('batsman_runs', 0), errors='coerce').fillna(0).astype(int)
    d['is_dot']  = ((d['batsman_runs'] == 0) & (d['legal_ball'] == 1)).astype(int)
    d['is_one']  = (d['batsman_runs'] == 1).astype(int)
    d['is_two']  = (d['batsman_runs'] == 2).astype(int)
    d['is_three']= (d['batsman_runs'] == 3).astype(int)
    d['is_four'] = (d['batsman_runs'] == 4).astype(int)
    d['is_six']  = (d['batsman_runs'] == 6).astype(int)

    # ---- dismissal normalization and numeric helpers ----
    special_runout_types = set([
        'run out', 'runout',
        'obstructing the field', 'obstructing thefield', 'obstructing',
        'retired out', 'retired not out (hurt)', 'retired not out', 'retired'
    ])
    d['dismissal_clean'] = d.get('dismissal', pd.Series([None]*len(d), index=d.index)).astype(str).str.lower().str.strip()
    d['dismissal_clean'] = d['dismissal_clean'].replace({'nan': '', 'none': ''})
    d['p_bat_num'] = pd.to_numeric(d.get('p_bat', np.nan), errors='coerce')
    d['p_out_num'] = pd.to_numeric(d.get('p_out', np.nan), errors='coerce')
    d['out_flag'] = pd.to_numeric(d.get('out', 0), errors='coerce').fillna(0).astype(int)

    # ---- ball ordering (use 'ball' or index) ----
    if 'ball' in d.columns:
        tmp = pd.to_numeric(d['ball'], errors='coerce')
        seq = pd.Series(np.arange(len(d)), index=d.index)
        tmp = tmp.fillna(seq)
        d['__ball_sort__'] = tmp
    else:
        d['__ball_sort__'] = pd.Series(np.arange(len(d)), index=d.index)

    if 'match_id' not in d.columns:
        d['match_id'] = 0
    d.sort_values(['match_id', '__ball_sort__'], inplace=True, kind='stable')
    d.reset_index(drop=True, inplace=True)

    # ---- initialize dismissal attribution fields ----
    d['dismissed_player'] = None
    d['bowler_wkt'] = 0  # 1 if bowler credited for wicket

    # ---- dismissal resolution per match using rules you specified ----
    for match in d['match_id'].unique():
        idxs = d.index[d['match_id'] == match].tolist()
        idxs = sorted(idxs, key=lambda i: d.at[i, '__ball_sort__'])
        for pos, i in enumerate(idxs):
            # if out flag not set, skip (no dismissal to resolve)
            if int(d.at[i, 'out_flag']) != 1:
                continue

            disc = (d.at[i, 'dismissal_clean'] or '').strip()
            striker = d.at[i, 'bat'] if 'bat' in d.columns else None

            # Case 1: dismissal text exists and is NOT in special set -> striker is out, bowler credited
            if disc and (disc not in special_runout_types):
                d.at[i, 'dismissed_player'] = striker
                d.at[i, 'bowler_wkt'] = 1
                continue

            # Case 2: dismissal in special set (run out / retired / obstructing) OR blank -> check p_bat/p_out
            pbat = d.at[i, 'p_bat_num']
            pout = d.at[i, 'p_out_num']
            if (not pd.isna(pbat)) and (not pd.isna(pout)) and (pbat == pout):
                # same -> striker out (no bowler credit)
                d.at[i, 'dismissed_player'] = striker
                d.at[i, 'bowler_wkt'] = 0
                continue

            # Otherwise, attribute to nonstriker: find last different 'bat' earlier in same match
            nonstriker = None
            last_idx_of_nonstriker = None
            for j in reversed(idxs[:pos]):
                prev_bat = d.at[j, 'bat'] if 'bat' in d.columns else None
                if prev_bat is not None and prev_bat != striker:
                    nonstriker = prev_bat
                    last_idx_of_nonstriker = j
                    break

            if nonstriker is None:
                # fallback to striker
                d.at[i, 'dismissed_player'] = striker
                d.at[i, 'bowler_wkt'] = 0
                continue

            # Inspect last ball the nonstriker played: if that previous ball's out_flag == 0 => nonstriker is dismissed now
            prev_out_flag = int(d.at[last_idx_of_nonstriker, 'out_flag']) if last_idx_of_nonstriker is not None else 0
            if prev_out_flag == 0:
                d.at[i, 'dismissed_player'] = nonstriker
                d.at[i, 'bowler_wkt'] = 0
            else:
                # fallback to striker
                d.at[i, 'dismissed_player'] = striker
                d.at[i, 'bowler_wkt'] = 0

    # ---- aggregate bowler-level stats using resolved bowler_wkt and flags ----
    if 'bowler' not in d.columns:
        d['bowler'] = d.get('bowl', None)

    runs = d.groupby('bowler')['runs_for_bowler'].sum().reset_index().rename(columns={'runs_for_bowler': 'runs'})
    innings = d.groupby('bowler')['match_id'].nunique().reset_index().rename(columns={'match_id': 'innings'})
    balls = d.groupby('bowler')['legal_ball'].sum().reset_index().rename(columns={'legal_ball': 'balls'})
    wkts = d.groupby('bowler')['bowler_wkt'].sum().reset_index().rename(columns={'bowler_wkt': 'wkts'})
    dots = d.groupby('bowler')['is_dot'].sum().reset_index().rename(columns={'is_dot': 'dots'})
    fours = d.groupby('bowler')['is_four'].sum().reset_index().rename(columns={'is_four': 'fours'})
    sixes = d.groupby('bowler')['is_six'].sum().reset_index().rename(columns={'is_six': 'sixes'})

    # three-wicket hauls & bbi (we will compute formatted BBI below)
    dismissals_count = d.groupby(['bowler', 'match_id'])['bowler_wkt'].sum().reset_index(name='wkts_in_match')
    three_wicket_hauls = dismissals_count[dismissals_count['wkts_in_match'] >= 3].groupby('bowler').size().reset_index(name='three_wicket_hauls')

    # runs conceded per bowler per match (using runs_for_bowler rule)
    runs_per_match = d.groupby(['bowler', 'match_id'])['runs_for_bowler'].sum().reset_index(name='runs_in_match')

    # compute BBI: pick the match with highest wkts_in_match, tie-breaker lower runs_in_match
    bbi_merge = dismissals_count.merge(runs_per_match, on=['bowler', 'match_id'], how='left')
    # select best row per bowler
    def choose_bbi(sub):
        if sub.empty:
            return pd.Series({'bbi_wkts': np.nan, 'bbi_runs': np.nan})
        sub_sorted = sub.sort_values(by=['wkts_in_match', 'runs_in_match'], ascending=[False, True]).reset_index(drop=True)
        return pd.Series({'bbi_wkts': int(sub_sorted.loc[0, 'wkts_in_match']), 'bbi_runs': int(sub_sorted.loc[0, 'runs_in_match'])})
    bbi_choice = bbi_merge.groupby('bowler').apply(choose_bbi).reset_index()
    # create formatted BBI string "w/r"
    if not bbi_choice.empty:
        bbi_choice['BBI'] = bbi_choice.apply(lambda r: f"{int(r['bbi_wkts'])}/{int(r['bbi_runs'])}" if (not pd.isna(r['bbi_wkts']) and not pd.isna(r['bbi_runs'])) else np.nan, axis=1)
    else:
        bbi_choice = pd.DataFrame(columns=['bowler', 'bbi_wkts', 'bbi_runs', 'BBI'])

    # ---- over/maiden logic ----
    if 'over' in d.columns:
        try:
            d['over_num'] = pd.to_numeric(d['over'], errors='coerce').fillna(0).astype(int)
        except Exception:
            d['over_num'] = d['over'].astype(str).str.split('.').str[0].fillna('0').astype(int)
    else:
        d['over_num'] = 0

    over_agg = d.groupby(['bowler', 'match_id', 'over_num']).agg(
        balls_in_over=('legal_ball', 'sum'),
        runs_in_over=('total_runs', 'sum')
    ).reset_index()
    maiden_overs_count = over_agg[(over_agg['balls_in_over'] == 6) & (over_agg['runs_in_over'] == 0)].groupby('bowler').size().reset_index(name='maiden_overs')

    # ---- phase metrics (use "Middle 1" / "Middle 2" naming) ----
    if 'over' in d.columns:
        d['phase'] = d['over'].apply(categorize_phase)
    else:
        d['phase'] = 'Unknown'

    phase_group = d.groupby(['bowler', 'phase']).agg(
        phase_balls=('legal_ball', 'sum'),
        phase_runs=('batsman_runs', 'sum'),
        phase_wkts=('bowler_wkt', 'sum'),
        phase_dots=('is_dot', 'sum'),
        phase_innings=('match_id', 'nunique')
    ).reset_index()

    def pivot_metric(df_pg, metric):
        if df_pg.empty:
            return pd.DataFrame({'bowler': []})
        pivoted = df_pg.pivot(index='bowler', columns='phase', values=metric).fillna(0)
        expected_phases = ['Powerplay', 'Middle 1', 'Middle 2', 'Death', 'Unknown']
        for ph in expected_phases:
            if ph not in pivoted.columns:
                pivoted[ph] = 0
        pivoted = pivoted.rename(columns={ph: f"{metric}_{ph.replace(' ', '')}" for ph in pivoted.columns})
        pivoted = pivoted.reset_index()
        return pivoted

    pb = pivot_metric(phase_group, 'phase_balls')
    pr = pivot_metric(phase_group, 'phase_runs')
    pw = pivot_metric(phase_group, 'phase_wkts')
    pdot = pivot_metric(phase_group, 'phase_dots')
    pi = pivot_metric(phase_group, 'phase_innings')

    phase_df = pb.merge(pr, on='bowler', how='outer').merge(pw, on='bowler', how='outer') \
                 .merge(pdot, on='bowler', how='outer').merge(pi, on='bowler', how='outer').fillna(0)

    # ---- Mega Over detection ----
    df_sorted = d.sort_values(['match_id', '__ball_sort__']).reset_index(drop=True).copy()
    df_sorted['ball_str'] = df_sorted.get('ball', df_sorted['__ball_sort__']).astype(str)
    df_sorted['frac'] = df_sorted['ball_str'].str.split('.').str[1].fillna('0')
    df_sorted['frac_int'] = pd.to_numeric(df_sorted['frac'], errors='coerce').fillna(0).astype(int)
    df_sorted['prev_bowler'] = df_sorted['bowler'].shift(1)
    df_sorted['prev_match'] = df_sorted['match_id'].shift(1)
    df_sorted['prev_bowler_same'] = (df_sorted['prev_bowler'] == df_sorted['bowler']) & (df_sorted['prev_match'] == df_sorted['match_id'])
    df_sorted['Mega_Over'] = (df_sorted['frac_int'] == 1) & (df_sorted['prev_bowler_same'])
    mega_over_count = df_sorted[df_sorted['Mega_Over']].groupby('bowler').size().reset_index(name='Mega_Over_Count')

    # ---- combine components into bowl_rec ----
    bowl_rec = innings.merge(balls, on='bowler', how='outer') \
                      .merge(runs, on='bowler', how='outer') \
                      .merge(wkts, on='bowler', how='outer') \
                      .merge(sixes, on='bowler', how='outer') \
                      .merge(fours, on='bowler', how='outer') \
                      .merge(dots, on='bowler', how='outer') \
                      .merge(three_wicket_hauls, on='bowler', how='left') \
                      .merge(maiden_overs_count, on='bowler', how='left') \
                      .merge(bbi_choice[['bowler', 'bbi_wkts', 'bbi_runs', 'BBI']] if not bbi_choice.empty else bbi_choice, on='bowler', how='left') \
                      .merge(phase_df, on='bowler', how='left') \
                      .merge(mega_over_count, on='bowler', how='left')

    # fill defaults for integer columns
    for c in ['three_wicket_hauls', 'maiden_overs', 'Mega_Over_Count', 'bbi_wkts', 'bbi_runs']:
        if c in bowl_rec.columns:
            bowl_rec[c] = bowl_rec[c].fillna(0).astype(int)

    # If BBI string missing, fill with NaN
    if 'BBI' in bowl_rec.columns:
        bowl_rec['BBI'] = bowl_rec['BBI'].fillna(np.nan)

    # debut / final season
    if 'season' in d.columns:
        debut_final = d.groupby('bowler')['season'].agg(debut_year='min', final_year='max').reset_index()
        bowl_rec = bowl_rec.merge(debut_final, on='bowler', how='left')
    else:
        bowl_rec['debut_year'] = np.nan
        bowl_rec['final_year'] = np.nan

    # ensure numeric defaults
    numeric_defaults = ['balls', 'runs', 'wkts', 'sixes', 'fours', 'dots']
    for col in numeric_defaults:
        if col in bowl_rec.columns:
            bowl_rec[col] = pd.to_numeric(bowl_rec[col], errors='coerce').fillna(0)

    # derived metrics (guard divide-by-zero)
    bowl_rec['dot%'] = bowl_rec.apply(lambda r: (r['dots'] / r['balls'] * 100) if r['balls'] > 0 else np.nan, axis=1)
    bowl_rec['avg'] = bowl_rec.apply(lambda r: (r['runs'] / r['wkts']) if r['wkts'] > 0 else np.nan, axis=1)
    bowl_rec['sr'] = bowl_rec.apply(lambda r: (r['balls'] / r['wkts']) if r['wkts'] > 0 else np.nan, axis=1)
    bowl_rec['econ'] = bowl_rec.apply(lambda r: (r['runs'] * 6.0 / r['balls']) if r['balls'] > 0 else np.nan, axis=1)
    bowl_rec['WPI'] = bowl_rec.apply(lambda r: (r['wkts'] / r['innings']) if r['innings'] > 0 else np.nan, axis=1)
    bowl_rec['DPI'] = bowl_rec.apply(lambda r: (r['dots'] / r['innings']) if r['innings'] > 0 else np.nan, axis=1)
    bowl_rec['RPI'] = bowl_rec.apply(lambda r: (r['runs'] / r['innings']) if r['innings'] > 0 else np.nan, axis=1)

    bowl_rec['bdry%'] = bowl_rec.apply(lambda r: ((r.get('fours',0) + r.get('sixes',0)) / r['balls'] * 100) if r['balls'] > 0 else np.nan, axis=1)
    bowl_rec['BPB'] = bowl_rec.apply(lambda r: (r['balls'] / (r.get('fours',0) + r.get('sixes',0))) if (r.get('fours',0) + r.get('sixes',0)) > 0 else np.nan, axis=1)
    bowl_rec['BPD'] = bowl_rec.apply(lambda r: (r['balls'] / r['wkts']) if r['wkts'] > 0 else np.nan, axis=1)
    bowl_rec['BP6'] = bowl_rec.apply(lambda r: (r['balls'] / r['sixes']) if r['sixes'] > 0 else np.nan, axis=1)

    # over-wise run counts (already computed earlier)
    overwise_runs = d.groupby(['bowler', 'match_id'])['total_runs'].sum().reset_index()
    ten_run_overs = d.groupby(['bowler', 'match_id']).apply(lambda g: (g.groupby('over_num')['total_runs'].sum() >= 10).sum()).reset_index(name='10_run_overs')
    seven_plus_overs = d.groupby(['bowler', 'match_id']).apply(lambda g: (g.groupby('over_num')['total_runs'].sum() >= 7).sum()).reset_index(name='7_plus_run_overs')
    six_minus_overs = d.groupby(['bowler', 'match_id']).apply(lambda g: (g.groupby('over_num')['total_runs'].sum() <= 6).sum()).reset_index(name='6_minus_run_overs')

    # collapse to bowler-level counts
    def collapse_counts(df_counts):
        if df_counts.empty:
            return pd.DataFrame(columns=['bowler', df_counts.columns[-1]])
        agg = df_counts.groupby('bowler').sum().reset_index()
        return agg

    ten_run_overs = collapse_counts(ten_run_overs)
    seven_plus_overs = collapse_counts(seven_plus_overs)
    six_minus_overs = collapse_counts(six_minus_overs)

    bowl_rec = bowl_rec.merge(ten_run_overs, on='bowler', how='left')
    bowl_rec = bowl_rec.merge(seven_plus_overs, on='bowler', how='left')
    bowl_rec = bowl_rec.merge(six_minus_overs, on='bowler', how='left')

    for col in ['10_run_overs', '7_plus_run_overs', '6_minus_run_overs']:
        if col in bowl_rec.columns:
            bowl_rec[col] = bowl_rec[col].fillna(0).astype(int)

    # overs string representation
    if 'balls' in bowl_rec.columns:
        bowl_rec['overs'] = bowl_rec['balls'].apply(lambda x: f"{int(x // 6)}.{int(x % 6)}" if pd.notna(x) else "0.0")
    else:
        bowl_rec['overs'] = "0.0"

    # final cleanup: keep rows with bowler name
    bowl_rec = bowl_rec[bowl_rec['bowler'].notna()].reset_index(drop=True)

    # Round float columns to 2 decimals for presentation
    float_cols = ['econ', 'avg', 'sr', 'dot%', 'WPI', 'DPI', 'RPI', 'bdry%', 'BPB', 'BPD', 'BP6']
    for col in float_cols:
        if col in bowl_rec.columns:
            bowl_rec[col] = bowl_rec[col].round(2)

    return bowl_rec



# -----------------------
# Streamlit integration
# -----------------------
@st.cache_data
def build_idf(df_local):
    return cumulator(df_local)


sidebar_option = st.sidebar.radio(
    "Select an option:",
    ("Player Profile", "Matchup Analysis", "Strength vs Weakness", "Match by Match Analysis")
)

if df is not None:
    idf = build_idf(df)
else:
    idf = pd.DataFrame()

if sidebar_option == "Player Profile":
    st.header("Player Profile")

    if idf is None or idf.empty:
        st.error("⚠️ Please run idf = Custom(df) before showing Player Profile (ensure raw 'df' is loaded).")
        st.stop()
    if df is None:
        st.error("⚠️ This view requires the original raw 'df' (ball-by-ball / match-level dataframe). Please ensure 'df' is loaded.")
        st.stop()

    def as_dataframe(x):
        if isinstance(x, pd.Series):
            return x.to_frame().T.reset_index(drop=True)
        elif isinstance(x, pd.DataFrame):
            return x.copy()
        else:
            try:
                return pd.DataFrame(x)
            except Exception:
                return pd.DataFrame()

    idf = as_dataframe(idf)
    df  = as_dataframe(df)

    if 'batsman' not in idf.columns:
        if 'bat' in idf.columns:
            idf = idf.rename(columns={'bat': 'batsman'})
        else:
            st.error("Dataset must contain a 'batsman' or 'bat' column in idf.")
            st.stop()

    players = sorted(idf['batsman'].dropna().unique().tolist())
    if not players:
        st.error("No players found in idf dataset.")
        st.stop()

    player_name = st.selectbox("Search for a player", players, index=0)

    tabs = st.tabs(["Career Statistics"])
    with tabs[0]:
        st.header("Career Statistics")
        option = st.selectbox("Select Career Stat Type", ("Batting", "Bowling"))

    if option == "Batting":
        player_stats = as_dataframe(idf[idf['batsman'] == player_name])
        if player_stats is None or player_stats.empty:
            st.warning(f"No data available for {player_name}.")
            st.stop()
    
        # cleanup & formatting
        player_stats = player_stats.drop(columns=['final_year'], errors='ignore')
        player_stats.columns = [str(col).upper().replace('_', ' ') for col in player_stats.columns]
        player_stats = round_up_floats(player_stats)
    
        int_cols = ['RUNS', 'HUNDREDS', 'FIFTIES', 'THIRTIES', 'HIGHEST SCORE']
        for c in int_cols:
            if c in player_stats.columns:
                player_stats[c] = pd.to_numeric(player_stats[c], errors='coerce').fillna(0).astype(int)
    
        # Header / metric cards (no emojis)
        st.markdown("### Batting Statistics")
    
        # helper to find column by candidates
        def find_col(df, candidates):
            for cand in candidates:
                if cand in df.columns:
                    return cand
            return None
    
        # preferred top metrics (ordered)
        top_metric_mapping = {
            "Runs": ["RUNS", "RUNS "],
            "Innings": ["INNINGS", "MATCHES"],
            "Average": ["AVG", "AVERAGE"],
            "Strike Rate": ["SR", "STRIKE RATE"],
            "Highest Score": ["HIGHEST SCORE", "HS"],
            "50s": ["FIFTIES", "50S", "FIFTY"],
            "100s": ["HUNDREDS", "100S"],
        }
    
        # collect values for display
        found_top_cols = {}
        for label, candidates in top_metric_mapping.items():
            col = find_col(player_stats, candidates)
            val = None
            if col is not None:
                try:
                    val = player_stats.iloc[0][col]
                except Exception:
                    val = player_stats[col].values[0] if len(player_stats[col].values) > 0 else None
            found_top_cols[label] = val
    
        # Display top metrics as columns (show only those that exist)
        visible_metrics = [(k, v) for k, v in found_top_cols.items() if v is not None and (not (isinstance(v, float) and np.isnan(v)))]
        if visible_metrics:
            cols = st.columns(len(visible_metrics))
            for (label, val), col in zip(visible_metrics, cols):
                if isinstance(val, (int, np.integer)):
                    disp = f"{int(val)}"
                elif isinstance(val, (float, np.floating)) and not np.isnan(val):
                    disp = f"{val:.2f}"
                else:
                    disp = str(val)
                col.metric(label, disp)
        else:
            st.write("Top metrics not available for this player.")
    
        # --------------------
        # Show the rest of the single-row summary as vertical key:value table
        # --------------------
        # Remove displayed top columns from the transposed view EXCEPT keep 'RUNS' (show Runs in Detailed)
        top_cols_used = [find_col(player_stats, cand) for cand in top_metric_mapping.values()]
        top_cols_used = [c for c in top_cols_used if c is not None]
        # ensure RUNS is not removed from the detailed view
        top_cols_used_excluding_runs = [c for c in top_cols_used if c is not None and c.upper() != 'RUNS']
    
        try:
            rest_series = player_stats.iloc[0].drop(labels=top_cols_used_excluding_runs, errors='ignore')
        except Exception:
            rest_series = pd.Series(dtype=object)
    
        if not rest_series.empty:
            rest_df = rest_series.reset_index()
            rest_df.columns = ["Metric", "Value"]
    
            def fmt_val(x):
                if pd.isna(x):
                    return ""
                if isinstance(x, (int, np.integer)):
                    return int(x)
                if isinstance(x, (float, np.floating)):
                    return round(x, 2)
                return x
    
            rest_df["Value"] = rest_df["Value"].apply(fmt_val)
    
            # Light "skin" header color for Detailed stats (peach / light skin tone)
            detailed_header_color = "#fff0e6"  # light skin/peach
            detailed_table_styles = [
                {"selector": "thead th", "props": [("background-color", detailed_header_color), ("color", "#000"), ("font-weight", "600")]},
                {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
                {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#fff9f4")]},
            ]
    
            st.markdown("#### Detailed stats")
            st.dataframe(rest_df.style.set_table_styles(detailed_table_styles), use_container_width=True)
        else:
            st.write("No additional per-player summary metrics available.")
    
        # --------------------
        # Opponent / Year / Inning breakdowns: use scrollable, lightly-colored tables
        # --------------------
        bat_col = 'batsman' if 'batsman' in df.columns else ('bat' if 'bat' in df.columns else None)
        if bat_col:
            opp_col = safe_get_col(df, ['team_bowl', 'team_bow', 'team_bowling'], default=None)
            if opp_col:
                opponents = sorted(df[df[bat_col] == player_name][opp_col].dropna().unique().tolist())
                all_opp = []
                for opp in opponents:
                    temp = df[(df[bat_col] == player_name) & (df[opp_col] == opp)].copy()
                    if temp.empty:
                        continue
                    temp_summary = cumulator(temp)
                    if not temp_summary.empty:
                        temp_summary['OPPONENT'] = opp
                        all_opp.append(temp_summary)
                if all_opp:
                    result_df = pd.concat(all_opp, ignore_index=True).drop(columns=['batsman', 'debut_year', 'final_year'], errors='ignore')
    
                    # Upper-case column names, replace underscores with spaces, and normalize Middle1/Middle2 to "Middle 1"/"Middle 2"
                    new_cols = []
                    for col in result_df.columns:
                        cname = str(col).upper().replace('_', ' ')
                        cname = cname.replace('MIDDLE1', 'MIDDLE 1').replace('MIDDLE2', 'MIDDLE 2')
                        new_cols.append(cname)
                    result_df.columns = new_cols
    
                    # Ensure Opponent is first column
                    if 'OPPONENT' in result_df.columns:
                        cols = ['OPPONENT'] + [c for c in result_df.columns if c != 'OPPONENT']
                        result_df = result_df[cols]
    
                    # cast a few numeric cols safely
                    for c in ['HUNDREDS', 'FIFTIES', '30S', 'RUNS', 'HIGHEST SCORE']:
                        if c in result_df.columns:
                            result_df[c] = pd.to_numeric(result_df[c], errors='coerce').fillna(0).astype(int)
                    result_df = round_up_floats(result_df)
    
                    # Light blue header color for Opponentwise table
                    opp_header_color = "#e6f2ff"
                    opp_table_styles = [
                        {"selector": "thead th", "props": [("background-color", opp_header_color), ("color", "#000"), ("font-weight", "600")]},
                        {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
                        {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#f7fbff")]},
                    ]
                    st.markdown("### Opponentwise Performance")
                    st.dataframe(result_df.style.set_table_styles(opp_table_styles), use_container_width=True)
    
            # Yearwise
            if 'year' in df.columns:
                seasons = sorted(df[df[bat_col] == player_name]['year'].dropna().unique().tolist())
                all_seasons = []
                for season in seasons:
                    temp = df[(df[bat_col] == player_name) & (df['year'] == season)].copy()
                    if temp.empty:
                        continue
                    temp_summary = cumulator(temp)
                    if not temp_summary.empty:
                        temp_summary['YEAR'] = season
                        all_seasons.append(temp_summary)
                if all_seasons:
                    result_df = pd.concat(all_seasons, ignore_index=True).drop(columns=['batsman', 'debut_year', 'final_year'], errors='ignore')
    
                    # Upper-case column names, replace underscores with spaces, normalize Middle1/Middle2
                    new_cols = []
                    for col in result_df.columns:
                        cname = str(col).upper().replace('_', ' ')
                        cname = cname.replace('MIDDLE1', 'MIDDLE 1').replace('MIDDLE2', 'MIDDLE 2')
                        new_cols.append(cname)
                    result_df.columns = new_cols
    
                    # Ensure YEAR is first column
                    if 'YEAR' in result_df.columns:
                        cols = ['YEAR'] + [c for c in result_df.columns if c != 'YEAR']
                        result_df = result_df[cols]
    
                    for c in ['RUNS', 'HUNDREDS', 'FIFTIES', '30S', 'HIGHEST SCORE']:
                        if c in result_df.columns:
                            result_df[c] = pd.to_numeric(result_df[c], errors='coerce').fillna(0).astype(int)
                    result_df = round_up_floats(result_df)
    
                    # Light purple header for Yearwise table
                    year_header_color = "#efe6ff"  # light purple
                    year_table_styles = [
                        {"selector": "thead th", "props": [("background-color", year_header_color), ("color", "#000"), ("font-weight", "600")]},
                        {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
                        {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#fbf7ff")]},
                    ]
                    st.markdown("### Yearwise Performance")
                    st.dataframe(result_df.style.set_table_styles(year_table_styles), use_container_width=True)
    
            # -------------------------
            # Venuewise Performance (Batting) — drop in after Yearwise
            # ------------
            
            bpdf = as_dataframe(df)  # raw ball-by-ball
            bat_col = 'batsman' if 'batsman' in bpdf.columns else ('bat' if 'bat' in bpdf.columns else None)
            
            venue_candidates = ['ground', 'venue', 'stadium', 'ground_name']
            venue_col = safe_get_col(bpdf, venue_candidates, default=None)
            
            if bat_col is None:
                st.info("Venuewise batting breakdown not available (missing 'bat'/'batsman' column).")
            elif venue_col is None:
                st.info("Venuewise batting breakdown not available (missing ground/venue/stadium column).")
            else:
                tdf = bpdf[bpdf[bat_col] == player_name].copy()
                unique_venues = sorted(tdf[venue_col].dropna().unique().tolist())
                all_venues = []
                for venue in unique_venues:
                    temp = tdf[tdf[venue_col] == venue].copy()
                    if temp.empty:
                        continue
                    temp_summary = cumulator(temp)
                    temp_summary = as_dataframe(temp_summary)
                    if temp_summary.empty:
                        continue
                    temp_summary['VENUE'] = venue
                    # make VENUE first
                    cols = temp_summary.columns.tolist()
                    if 'VENUE' in temp_summary.columns:
                        temp_summary = temp_summary[['VENUE'] + [c for c in cols if c != 'VENUE']]
                    all_venues.append(temp_summary)
            
                if all_venues:
                    result_df = pd.concat(all_venues, ignore_index=True).drop(columns=['batsman', 'debut_year', 'final_year'], errors='ignore')
                    # uppercase and space format, normalize middle phase names
                    new_cols = []
                    for col in result_df.columns:
                        cname = str(col).upper().replace('_', ' ')
                        cname = cname.replace('MIDDLE1', 'MIDDLE 1').replace('MIDDLE2', 'MIDDLE 2')
                        new_cols.append(cname)
                    result_df.columns = new_cols
            
                    # safe numeric casts
                    for c in ['RUNS', 'HUNDREDS', 'FIFTIES', '30S', 'HIGHEST SCORE']:
                        if c in result_df.columns:
                            result_df[c] = pd.to_numeric(result_df[c], errors='coerce').fillna(0).astype(int)
            
                    result_df = round_up_floats(result_df)
            
                    # Styling: reuse light-blue style (or change color hex if you prefer)
                    venue_header_color = "#e6f7ff"
                    venue_table_styles = [
                        {"selector": "thead th", "props": [("background-color", venue_header_color), ("color", "#000"), ("font-weight", "600")]},
                        {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
                        {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#f7fdff")]},
                    ]
            
                    st.markdown("### Venuewise Performance")
                    st.dataframe(result_df.style.set_table_styles(venue_table_styles), use_container_width=True)
                else:
                    st.info("No venuewise batting summary available for this player.")
    
    
            # Inningwise
            inning_col = 'inns' if 'inns' in df.columns else ('inning' if 'inning' in df.columns else None)
            if inning_col:
                innings_list = []
                for inn in sorted(df[inning_col].dropna().unique()):
                    temp = df[(df[bat_col] == player_name) & (df[inning_col] == inn)].copy()
                    if temp.empty:
                        continue
                    temp_summary = cumulator(temp)
                    if not temp_summary.empty:
                        temp_summary['INNING'] = inn
                        innings_list.append(temp_summary)
                if innings_list:
                    result_df = pd.concat(innings_list, ignore_index=True).drop(columns=['batsman', 'debut_year', 'final_year'], errors='ignore')
    
                    # Upper-case column names, replace underscores with spaces, normalize Middle1/Middle2
                    new_cols = []
                    for col in result_df.columns:
                        cname = str(col).upper().replace('_', ' ')
                        cname = cname.replace('MIDDLE1', 'MIDDLE 1').replace('MIDDLE2', 'MIDDLE 2')
                        new_cols.append(cname)
                    result_df.columns = new_cols
    
                    # Ensure INNING is first col (and show 1/2 if that is the value)
                    if 'INNING' in result_df.columns:
                        cols = ['INNING'] + [c for c in result_df.columns if c != 'INNING']
                        result_df = result_df[cols]
    
                    for c in ['RUNS', 'HUNDREDS', 'FIFTIES', '30S', 'HIGHEST SCORE']:
                        if c in result_df.columns:
                            result_df[c] = pd.to_numeric(result_df[c], errors='coerce').fillna(0).astype(int)
                    result_df = round_up_floats(result_df)
                    result_df = result_df.drop(columns=['MATCHES'], errors='ignore')
    
                    # Light green header for Inningwise table
                    inning_header_color = "#e9f9ea"
                    inning_table_styles = [
                        {"selector": "thead th", "props": [("background-color", inning_header_color), ("color", "#000"), ("font-weight", "600")]},
                        {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
                        {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#f3fff3")]},
                    ]
                    st.markdown("### Inningwise Performance")
                    st.dataframe(result_df.reset_index(drop=True).style.set_table_styles(inning_table_styles), use_container_width=True)
    
    
    
    
    elif option == "Bowling":
        # Defensive: ensure raw ball-by-ball df exist
    
        # bpdf is the ball-by-ball (same as df)
        bpdf = as_dataframe(df)
    
        # build bidf (bowling summary) from bowlerstat; do it defensively
        try:
            bidf = as_dataframe(bowlerstat(bpdf))
        except Exception as e:
            st.error(f"Failed to build bowling summary (bidf) from bowlerstat(): {e}")
            st.stop()
    
        # Ensure 'bowler' exists in bidf
        if 'bowler' not in bidf.columns:
            st.error("bidf must contain a 'bowler' column returned from bowlerstat().")
            st.stop()
    
        # Select player's bowling summary rows
        player_stats = bidf[bidf['bowler'] == player_name].copy()
        player_stats = as_dataframe(player_stats)
    
        if player_stats.empty:
            st.markdown("No bowling stats available for the selected player.")
        else:
            # display copy: uppercase column names & round floats
            disp_stats = player_stats.copy()
            disp_stats.columns = [str(col).upper().replace('_', ' ') for col in disp_stats.columns]
            disp_stats = round_up_floats(disp_stats)
    
            # helper to find column by candidates
            def find_col(df_local, candidates):
                for cand in candidates:
                    if cand in df_local.columns:
                        return cand
                return None
    
            # top metrics mapping for bowlers (ordered)
            top_metric_mapping = {
                "Innings": ["INNINGS", "MATCHES"],
                "Runs": ["RUNS", "RUNS CONCEDED"],
                "Wickets": ["WKTS", "WICKETS", "WICKETS "],
                "Econ": ["ECON", "ECONOMY"],
                "Average": ["AVG", "AVERAGE"],
                "Strike Rate": ["SR", "STRIKE RATE"],
                "3w Hauls": ["THREE_WICKET_HAULS", "3W", "THREE WICKET HAULS"],
                "BBI": ["BBI", "BEST BBI", "BEST"]
            }
    
            # collect top metric values
            found_top_cols = {}
            for label, candidates in top_metric_mapping.items():
                col = find_col(disp_stats, candidates)
                val = None
                if col is not None:
                    try:
                        val = disp_stats.iloc[0][col]
                    except Exception:
                        val = disp_stats[col].values[0] if len(disp_stats[col].values) > 0 else None
                found_top_cols[label] = val
    
            st.markdown("### Bowling Statistics")
    
            # show top metrics as metric cards
            visible_metrics = [
                (k, v) for k, v in found_top_cols.items()
                if v is not None and not (isinstance(v, float) and np.isnan(v))
            ]
            if visible_metrics:
                cols = st.columns(len(visible_metrics))
                for (label, val), col in zip(visible_metrics, cols):
                    if isinstance(val, (int, np.integer)):
                        disp = f"{int(val)}"
                    elif isinstance(val, (float, np.floating)) and not np.isnan(val):
                        disp = f"{val:.2f}"
                    else:
                        disp = str(val)
                    col.metric(label, disp)
            else:
                st.write("Top bowling metrics not available.")
    
            # -------------------------
            # Detailed stats (vertical key:value). keep RUNS displayed
            # -------------------------
            top_cols_used = [find_col(disp_stats, cand) for cand in top_metric_mapping.values()]
            top_cols_used = [c for c in top_cols_used if c is not None]
            top_cols_used_excluding_runs = [c for c in top_cols_used if c is not None and str(c).upper() != 'RUNS']
    
            try:
                rest_series = disp_stats.iloc[0].drop(labels=top_cols_used_excluding_runs, errors='ignore')
            except Exception:
                rest_series = pd.Series(dtype=object)
    
            if not rest_series.empty:
                rest_df = rest_series.reset_index()
                rest_df.columns = ["Metric", "Value"]
    
                def fmt_val(x):
                    if pd.isna(x):
                        return ""
                    if isinstance(x, (int, np.integer)):
                        return int(x)
                    if isinstance(x, (float, np.floating)):
                        return round(x, 2)
                    return x
    
                rest_df["Value"] = rest_df["Value"].apply(fmt_val)
    
                # light skin / peach header color
                detailed_header_color = "#fff0e6"
                detailed_table_styles = [
                    {"selector": "thead th", "props": [("background-color", detailed_header_color), ("color", "#000"), ("font-weight", "600")]},
                    {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
                    {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#fff9f4")]},
                ]
    
                st.markdown("#### Detailed stats")
                st.dataframe(rest_df.style.set_table_styles(detailed_table_styles), use_container_width=True)
            else:
                st.write("No detailed bowling metrics available.")
    
        # Use the raw ball-by-ball df
        bpdf = as_dataframe(df)
        
        # Determine actual bowler column in ball-by-ball (prioritize 'bowl' then 'bowler')
        bowler_col = safe_get_col(bpdf, ['bowl', 'bowler'], default=None)
        
        # -------------------------
        # Opponentwise Performance
        # -------------------------
        opp_candidates = ['team_bat', 'team_batting', 'batting_team', 'team_bowl']
        opp_col = safe_get_col(bpdf, opp_candidates, default=None)
        
        if bowler_col is None:
            st.info("Ball-by-ball data missing 'bowl'/'bowler' column; opponentwise breakdown not available.")
        elif opp_col is None:
            st.info("Could not detect opponent/team batting column (expected one of team_bat/team_batting/batting_team/team_bowl).")
        else:
            # Filter rows for this bowler name using actual column
            opponents = sorted(bpdf[bpdf[bowler_col] == player_name][opp_col].dropna().unique().tolist())
            all_opp = []
            for opp in opponents:
                temp = bpdf[(bpdf[bowler_col] == player_name) & (bpdf[opp_col] == opp)].copy()
                if temp.empty:
                    continue
                # bowlerstat will normalize 'bowl' -> 'bowler' internally
                temp_summary = bowlerstat(temp)
                temp_summary = as_dataframe(temp_summary)
                if temp_summary.empty:
                    continue
                temp_summary['OPPONENT'] = str(opp).upper()
                # ensure OPPONENT first
                cols = temp_summary.columns.tolist()
                if 'OPPONENT' in temp_summary.columns:
                    temp_summary = temp_summary[['OPPONENT'] + [c for c in cols if c != 'OPPONENT']]
                all_opp.append(temp_summary)
        
            if all_opp:
                result_df = pd.concat(all_opp, ignore_index=True).drop(columns=['bowler'], errors='ignore')
                result_df.columns = [str(col).upper().replace('_', ' ') for col in result_df.columns]
        
                # cast safe numeric columns
                for c in ['RUNS', 'WKTS', 'BALLS', 'OVERS', 'ECON', 'AVG']:
                    if c in result_df.columns:
                        result_df[c] = pd.to_numeric(result_df[c], errors='coerce').fillna(0)
        
                result_df = round_up_floats(result_df)
        
                # styling: light blue
                opp_header_color = "#e6f2ff"
                opp_table_styles = [
                    {"selector": "thead th", "props": [("background-color", opp_header_color), ("color", "#000"), ("font-weight", "600")]},
                    {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
                    {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#f7fbff")]},
                ]
        
                st.markdown("### Opponentwise Performance")
                st.dataframe(result_df.style.set_table_styles(opp_table_styles), use_container_width=True)
            else:
                st.info("No opponentwise bowling summary available for this player.")
        
        # -------------------------
        # Yearwise Performance (season/year)
        # -------------------------
        season_col = safe_get_col(bpdf, ['season', 'year'], default=None)
        if bowler_col is None:
            st.info("Ball-by-ball data missing 'bowl'/'bowler' column; yearwise breakdown not available.")
        elif season_col is None:
            st.info("Yearwise breakdown not available (missing 'season' / 'year' column).")
        else:
            tdf = bpdf[bpdf[bowler_col] == player_name].copy()
            unique_seasons = sorted(tdf[season_col].dropna().unique().tolist())
            all_seasons = []
            for season in unique_seasons:
                temp = tdf[tdf[season_col] == season].copy()
                if temp.empty:
                    continue
                temp_summary = bowlerstat(temp)
                temp_summary = as_dataframe(temp_summary)
                if temp_summary.empty:
                    continue
                temp_summary['YEAR'] = season
                # make YEAR first
                cols = temp_summary.columns.tolist()
                if 'YEAR' in temp_summary.columns:
                    temp_summary = temp_summary[['YEAR'] + [c for c in cols if c != 'YEAR']]
                all_seasons.append(temp_summary)
        
            if all_seasons:
                result_df = pd.concat(all_seasons, ignore_index=True).drop(columns=['bowler'], errors='ignore')
                result_df.columns = [str(col).upper().replace('_', ' ') for col in result_df.columns]
                result_df = round_up_floats(result_df)
        
                # light purple header
                year_header_color = "#efe6ff"
                year_table_styles = [
                    {"selector": "thead th", "props": [("background-color", year_header_color), ("color", "#000"), ("font-weight", "600")]},
                    {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
                    {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#fbf7ff")]},
                ]
        
                st.markdown("### Yearwise Bowling Performance")
                st.dataframe(result_df.style.set_table_styles(year_table_styles), use_container_width=True)
            else:
                st.info("No yearwise bowling summary available for this player.")
        
        # -------------------------
        # Venuewise Performance (Bowling) — drop in after Yearwise
        
        bpdf = as_dataframe(df)  # raw ball-by-ball
        # actual bowler column in your df is 'bowl' (bowlerstat normalizes it internally)
        bowler_col = safe_get_col(bpdf, ['bowl', 'bowler'], default=None)
        
        venue_candidates = ['ground', 'venue', 'stadium', 'ground_name']
        venue_col = safe_get_col(bpdf, venue_candidates, default=None)
        
        if bowler_col is None:
            st.info("Venuewise bowling breakdown not available (missing 'bowl'/'bowler' column).")
        elif venue_col is None:
            st.info("Venuewise bowling breakdown not available (missing ground/venue/stadium column).")
        else:
            tdf = bpdf[bpdf[bowler_col] == player_name].copy()
            unique_venues = sorted(tdf[venue_col].dropna().unique().tolist())
            all_venues = []
            for venue in unique_venues:
                temp = tdf[tdf[venue_col] == venue].copy()
                if temp.empty:
                    continue
                temp_summary = bowlerstat(temp)
                temp_summary = as_dataframe(temp_summary)
                if temp_summary.empty:
                    continue
                temp_summary['VENUE'] = venue.upper()
                # ensure VENUE first
                cols = temp_summary.columns.tolist()
                if 'VENUE' in temp_summary.columns:
                    temp_summary = temp_summary[['VENUE'] + [c for c in cols if c != 'VENUE']]
                all_venues.append(temp_summary)
        
            if all_venues:
                result_df = pd.concat(all_venues, ignore_index=True).drop(columns=['bowler'], errors='ignore')
                result_df.columns = [str(col).upper().replace('_', ' ') for col in result_df.columns]
                # safe numeric casts
                for c in ['RUNS', 'WKTS', 'BALLS', 'OVERS', 'ECON', 'AVG']:
                    if c in result_df.columns:
                        result_df[c] = pd.to_numeric(result_df[c], errors='coerce').fillna(0)
                result_df = round_up_floats(result_df)
        
                # Styling: light blue/teal for venue
                venue_header_color = "#e6f7ff"
                venue_table_styles = [
                    {"selector": "thead th", "props": [("background-color", venue_header_color), ("color", "#000"), ("font-weight", "600")]},
                    {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
                    {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#f7fbff")]},
                ]
        
                st.markdown("### Venuewise Performance")
                st.dataframe(result_df.style.set_table_styles(venue_table_styles), use_container_width=True)
            else:
                st.info("No venuewise bowling summary available for this player.")
        
        
        # -------------------------
        # Inningwise Performance
        # -------------------------
        inning_col = safe_get_col(bpdf, ['inns', 'inning'], default=None)
        if bowler_col is None:
            st.info("Ball-by-ball data missing 'bowl'/'bowler' column; inningwise breakdown not available.")
        elif inning_col is None:
            st.info("Inningwise breakdown not available (missing 'inns'/'inning' column).")
        else:
            tdf = bpdf[bpdf[bowler_col] == player_name].copy()
            innings_list = []
            for inn in sorted(tdf[inning_col].dropna().unique()):
                temp = tdf[tdf[inning_col] == inn].copy()
                if temp.empty:
                    continue
                temp_summary = bowlerstat(temp)
                temp_summary = as_dataframe(temp_summary)
                if temp_summary.empty:
                    continue
                temp_summary['INNING'] = inn
                cols = temp_summary.columns.tolist()
                if 'INNING' in temp_summary.columns:
                    temp_summary = temp_summary[['INNING'] + [c for c in cols if c != 'INNING']]
                innings_list.append(temp_summary)
        
            if innings_list:
                result_df = pd.concat(innings_list, ignore_index=True).drop(columns=['bowler'], errors='ignore')
                result_df.columns = [str(col).upper().replace('_', ' ') for col in result_df.columns]
                result_df = round_up_floats(result_df)
        
                # light green header
                inning_header_color = "#e9f9ea"
                inning_table_styles = [
                    {"selector": "thead th", "props": [("background-color", inning_header_color), ("color", "#000"), ("font-weight", "600")]},
                    {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
                    {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#f3fff3")]},
                ]
        
                st.markdown("### Inningwise Bowling Performance")
                st.dataframe(result_df.style.set_table_styles(inning_table_styles), use_container_width=True)
            else:
                st.info("No inningwise bowling summary available for this player.")

elif sidebar_option == "Matchup Analysis":

    st.header("Matchup Analysis")

    # Defensive helper fallbacks (use your existing ones if present)
    try:
        as_dataframe
    except NameError:
        def as_dataframe(x):
            if isinstance(x, pd.Series):
                return x.to_frame().T.reset_index(drop=True)
            elif isinstance(x, pd.DataFrame):
                return x.copy()
            else:
                return pd.DataFrame(x)


    bdf = as_dataframe(df)

    # Detect column names in your data
    batter_col = safe_get_col(bdf, ['bat', 'batsman'], default=None)
    bowler_col = safe_get_col(bdf, ['bowl', 'bowler'], default=None)
    match_col  = safe_get_col(bdf, ['p_match', 'match_id'], default=None)
    year_col   = safe_get_col(bdf, ['season', 'year'], default=None)
    inning_col = safe_get_col(bdf, ['inns', 'inning'], default=None)
    venue_col  = safe_get_col(bdf, ['ground', 'venue', 'stadium', 'ground_name'], default=None)

    if batter_col is None or bowler_col is None:
        st.error("Dataframe must contain batter and bowler columns (e.g. 'bat' and 'bowl').")
        st.stop()

    # Build unique player lists (filter out nulls and '0')
    unique_batters = sorted([x for x in pd.unique(bdf[batter_col].dropna()) if str(x).strip() not in ("", "0")])
    unique_bowlers  = sorted([x for x in pd.unique(bdf[bowler_col].dropna())  if str(x).strip() not in ("", "0")])

    if not unique_batters or not unique_bowlers:
        st.warning("No batters or bowlers found in the dataset.")
        st.stop()

    # Player selectors
    batter_name = st.selectbox("Select a Batter", unique_batters, index=0)
    bowler_name = st.selectbox("Select a Bowler", unique_bowlers, index=0)

    # Grouping option
    grouping_option = st.selectbox("Group By", ["Year", "Match", "Venue", "Inning"])

    # Raw matchup rows for download/sanity
    matchup_df = bdf[(bdf[batter_col] == batter_name) & (bdf[bowler_col] == bowler_name)].copy()

    if matchup_df.empty:
        st.warning("No data available for the selected matchup.")
    else:
        # Normalize numeric fields defensively
        for col in ['batsman_runs', 'batruns', 'score', 'bowlruns', 'total_runs']:
            if col in matchup_df.columns:
                try:
                    matchup_df[col] = pd.to_numeric(matchup_df[col], errors='coerce')
                except Exception:
                    pass

        # Download raw matchup CSV


        # Helper: Apply formatting to individual summary dataframe
        def format_summary_df(temp_summary):
            """Format a single summary dataframe with proper types"""
            df = temp_summary.copy()
            
            # Convert ALL numeric columns first
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
            
            # Now identify and convert integer columns
            for col in df.columns:
                col_lower = str(col).lower()
                # Check if column name contains 'innings', 'runs', or 'balls'
                if any(keyword in col_lower for keyword in ['innings', 'inning', 'runs', 'balls', 'wickets', 'wkts', 'dismissals', 'matches', 'fours', 'sixes', 'dots']):
                    df[col] = df[col].fillna(0).astype(int)
            
            # Round all other numeric columns to 2 decimals
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            for nc in numeric_cols:
                # Skip if already integer type
                if df[nc].dtype == int:
                    continue
                df[nc] = df[nc].round(2)
            df=round_up_floats(df)
            
            return df

        # Helper: convert column names to display form (UPPER + spaces)
        def normalize_display_columns(df_in):
            df = df_in.copy()
            df.columns = [str(col).upper().replace('_', ' ') for col in df.columns]
            return df

        # Finalize and display helper
        def finalize_and_show(df_list, primary_col_name, title, header_color="#efe6ff"):
            if not df_list:
                st.info(f"No {title.lower()} data available for this matchup.")
                return
            
            # Concatenate all formatted dataframes
            out = pd.concat(df_list, ignore_index=True)
            # st.write(out)
            # Remove batter and bowler columns if they exist
            cols_to_drop = []
            for col in out.columns:
                col_lower = str(col).lower()
                if any(x in col_lower for x in ['bat', 'bowl', 'batsman', 'bowler']):
                    cols_to_drop.append(col)
            
            out = out.drop(columns=cols_to_drop, errors='ignore')
            out=round_up_floats(out)
            # st.write(out)
            
            # # Convert to numeric where possible
            # for col in out.columns:
            #     out[col] = pd.to_numeric(out[col], errors='ignore')
            
            # # Convert specific columns to int, round others to 2 decimals
            # for col in out.columns:
            #     if any(x in col.lower() for x in ['innings', 'runs', 'balls']):
            #         out[col] = out[col].fillna(0).astype(int)
            #     elif out[col].dtype in ['float64', 'float32', 'float']:
            #         out[col] = out[col].round(2)
            
            # Replace None/NaN with 'sadh'
            out = out.fillna('-')
            
            # Capitalize first letter of each column name
            out.columns = [str(col).strip().capitalize() for col in out.columns]
            
            # Ensure primary column name is also capitalized
            primary_col_name_norm = str(primary_col_name).strip().capitalize()
            
            # Put primary column first if present
            # cols = out.columns.tolist()
            # if primary_col_name_norm in cols:
            #     new_order = [primary_col_name_norm] + [c for c in cols if c != primary_col_name_norm]
            #     out = out[new_order]
            
            # Table styling
            # table_styles = [
            #     {"selector": "thead th", "props": [("background-color", header_color), ("color", "#000"), ("font-weight", "600")]},
            #     {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
            #     {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#f7f7fb")]},
            # ]
            def beautify_columns(df):
                df.columns = [
                    ' '.join(word.capitalize() for word in str(col).replace('_', ' ').split())
                    for col in df.columns
                ]
                return df
            
            # Example usage:
            out = beautify_columns(out)
            st.markdown(f"### {title}")
            st.write(out)
            return out
            # st.dataframe(out.style.set_table_styles(table_styles), use_container_width=True)
            


        # -------------------
        # Year grouping
        # -------------------
        if grouping_option == "Year":
            if year_col is None:
                st.info("Year/season column not detected in dataset.")
            else:
                tdf = matchup_df.copy()
                seasons = sorted(tdf[year_col].dropna().unique().tolist())
                all_seasons = []
                for s in seasons:
                    temp = tdf[tdf[year_col] == s].copy()
                    if temp.empty:
                        continue
                    temp_summary = cumulator(temp)
                    temp_summary=round_up_floats(temp_summary)
                    # st.write(temp_summary)
                    # temp_summary = as_dataframe(temp_summary)
                    if temp_summary.empty:
                        continue
                    # FORMAT BEFORE ADDING TO LIST
                    temp_summary = format_summary_df(temp_summary)
                    # st.write(temp_summary)
                    temp_summary=round_up_floats(temp_summary)
                    # st.write(temp_summary)
                    temp_summary.insert(0, 'year', s)
                    all_seasons.append(temp_summary)

                out=finalize_and_show(all_seasons, 'year', "Yearwise Performance", header_color="#efe6ff")
                                # Add Batter and Bowler columns (in uppercase)
                out['Batsman'] = str(batter_name)
                out['Bowler'] = str(bowler_name)
                
                # Move them to the front if you want them as leading columns
                cols = ['Batsman', 'Bowler'] + [c for c in out.columns if c not in ['Batsman', 'Bowler']]
                out = out[cols]
                csv = out.to_csv(index=False)
                st.download_button(
                    label="Download raw matchup rows (CSV)",
                    data=csv,
                    file_name=f"{str(batter_name)}_vs_{str(bowler_name)}_matchup.csv",
                    mime="text/csv"
                )

        # -------------------
        # Match grouping
        # -------------------
        elif grouping_option == "Match":
            if match_col is None:
                st.info("Match ID column not detected in dataset (expected 'p_match' or 'match_id').")
            else:
                tdf = matchup_df.copy()
                match_ids = sorted(tdf[match_col].dropna().unique().tolist())
                all_matches = []
                for m in match_ids:
                    temp = tdf[tdf[match_col] == m].copy()
                    if temp.empty:
                        continue
                    temp_summary = cumulator(temp)
                    temp_summary = as_dataframe(temp_summary)
                    if temp_summary.empty:
                        continue
                    # FORMAT BEFORE ADDING TO LIST
                    temp_summary = format_summary_df(temp_summary)
                    temp_summary.insert(0, 'match_id', m)
                    all_matches.append(temp_summary)

                out=finalize_and_show(all_seasons, 'year', "Yearwise Performance", header_color="#efe6ff")
                                # Add Batter and Bowler columns (in uppercase)
                out['Batsman'] = str(batter_name)
                out['Bowler'] = str(bowler_name)
                
                # Move them to the front if you want them as leading columns
                cols = ['Batsman', 'Bowler'] + [c for c in out.columns if c not in ['Batsman', 'Bowler']]
                out = out[cols]
                csv = out.to_csv(index=False)
                st.download_button(
                    label="Download raw matchup rows (CSV)",
                    data=csv,
                    file_name=f"{str(batter_name)}_vs_{str(bowler_name)}_matchup.csv",
                    mime="text/csv"
                )

        # -------------------
        # Venue grouping
        # -------------------
        elif grouping_option == "Venue":
            if venue_col is None:
                st.info("Venue/ground column not detected in dataset.")
            else:
                tdf = matchup_df.copy()
                venues = sorted(tdf[venue_col].dropna().unique().tolist())
                all_venues = []
                for v in venues:
                    temp = tdf[tdf[venue_col] == v].copy()
                    if temp.empty:
                        continue
                    temp_summary = cumulator(temp)
                    temp_summary = as_dataframe(temp_summary)
                    if temp_summary.empty:
                        continue
                    # FORMAT BEFORE ADDING TO LIST
                    temp_summary = format_summary_df(temp_summary)
                    temp_summary.insert(0, 'venue', v)
                    all_venues.append(temp_summary)

                out=finalize_and_show(all_seasons, 'year', "Yearwise Performance", header_color="#efe6ff")
                                # Add Batter and Bowler columns (in uppercase)
                out['Batsman'] = str(batter_name)
                out['Bowler'] = str(bowler_name)
                
                # Move them to the front if you want them as leading columns
                cols = ['Batsman', 'Bowler'] + [c for c in out.columns if c not in ['Batsman', 'Bowler']]
                cols = ['Batsman', 'Bowler'] + [c for c in out.columns if c not in ['Batsman', 'Bowler']]
                out = out[cols]
                csv = out.to_csv(index=False)
                st.download_button(
                    label="Download raw matchup rows (CSV)",
                    data=csv,
                    file_name=f"{str(batter_name)}_vs_{str(bowler_name)}_matchup.csv",
                    mime="text/csv"
                )

        # -------------------
        # Inning grouping
        # -------------------
        elif grouping_option == "Inning":
            if inning_col is None:
                st.info("Inning column not detected in dataset (expected 'inns' or 'inning').")
            else:
                tdf = matchup_df.copy()
                innings = sorted(tdf[inning_col].dropna().unique().tolist())
                all_inns = []
                for inn in innings:
                    temp = tdf[tdf[inning_col] == inn].copy()
                    if temp.empty:
                        continue
                    temp_summary = cumulator(temp)
                    temp_summary = as_dataframe(temp_summary)
                    if temp_summary.empty:
                        continue
                    # FORMAT BEFORE ADDING TO LIST
                    temp_summary = format_summary_df(temp_summary)
                    temp_summary.insert(0, 'inning', inn)
                    all_inns.append(temp_summary)

                finalize_and_show(all_inns, 'inning', "Inningwise Performance", header_color="#e6f7ff")
                # Add Batter and Bowler columns (in uppercase)
                out['Batsman'] = str(batter_name)
                out['Bowler'] = str(bowler_name)
                cols = ['Batsman', 'Bowler'] + [c for c in out.columns if c not in ['Batsman', 'Bowler']]
                out = out[cols]
                csv = out.to_csv(index=False)
                st.download_button(
                    label="Download raw matchup rows (CSV)",
                    data=csv,
                    file_name=f"{str(batter_name)}_vs_{str(bowler_name)}_matchup.csv",
                    mime="text/csv"
                )

        else:
            st.info("Unknown grouping option selected.")


elif sidebar_option == "Match by Match Analysis":# Match by Match Analysis - full code block
    # Match by Match Analysis - full updated code with sector-label wagon wheel

    import streamlit as st
    import pandas as pd
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    from io import BytesIO
    import warnings
    
    # -------------------------
    # Configuration: suppress non-critical warnings
    # -------------------------
    warnings.filterwarnings("ignore")
    
    # -------------------------
    # safe plotting helper to avoid PIL DecompressionBombError
    # -------------------------
    def safe_st_pyplot(fig,
                       max_pixels: int = 40_000_000,
                       fallback_set_max: bool = False,
                       use_container_width: bool = True):
        """Save matplotlib figure to an in-memory buffer then display via st.image.
        Uses use_container_width param (no deprecated use_column_width).
        """
        try:
            dpi = fig.get_dpi()
            width_in = fig.get_figwidth()
            height_in = fig.get_figheight()
            width_px = int(round(width_in * dpi))
            height_px = int(round(height_in * dpi))
            pixel_count = int(width_px) * int(height_px)
    
            buf = BytesIO()
    
            if pixel_count <= max_pixels:
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                st.image(buf, use_container_width=use_container_width)
                plt.close(fig)
                return
            else:
                scale = math.sqrt(max_pixels / float(pixel_count))
                if scale <= 0 or scale >= 1:
                    scale = min(1.0, 0.7)
                new_dpi = max(50, int(math.floor(dpi * scale)))
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=new_dpi)
                buf.seek(0)
                st.image(buf, use_container_width=use_container_width)
                plt.close(fig)
                return
    
        except Exception as e:
            err_str = str(e)
            if ("DecompressionBombError" in err_str or "DecompressionBomb" in err_str) and fallback_set_max:
                try:
                    from PIL import Image
                    Image.MAX_IMAGE_PIXELS = None
                    buf = BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight")
                    buf.seek(0)
                    st.image(buf, use_container_width=use_container_width)
                    plt.close(fig)
                    return
                except Exception as e2:
                    st.error(f"still failed after disabling MAX_IMAGE_PIXELS: {e2}")
                    raise
            else:
                try:
                    buf = BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=70)
                    buf.seek(0)
                    st.info("Figure was too large; displayed at reduced resolution.")
                    st.image(buf, use_container_width=use_container_width)
                    plt.close(fig)
                    return
                except Exception:
                    st.error("Unable to render figure due to image size / PIL safety check.")
                    raise
    
    # -------------------------
    # Helpers
    # -------------------------
    
    def as_dataframe(x):
        if isinstance(x, pd.Series):
            return x.to_frame().T.reset_index(drop=True)
        elif isinstance(x, pd.DataFrame):
            return x.copy()
        else:
            return pd.DataFrame(x)
    
    
    def safe_get_col(df_local, candidates, default=None):
        for c in candidates:
            if c in df_local.columns:
                return c
        return default
    
    # -------------------------
    # Defensive check: df must be present
    # # -------------------------
    # try:
    #     df
    # except NameError:
    #     st.error("Raw ball-by-ball dataframe 'df' not found. Load your data first.")
    #     st.stop()
    
    bdf = as_dataframe(df)
    
    # detect columns (use your preferred names as preference)
    match_col = safe_get_col(bdf, ['p_match', 'match_id'], default=None)
    batter_col = safe_get_col(bdf, ['bat', 'batsman'], default=None)
    bowler_col = safe_get_col(bdf, ['bowl', 'bowler'], default=None)
    bat_hand_col = safe_get_col(bdf, ['bat_hand', 'batting_style'], default='bat_hand')
    wagon_zone_col = safe_get_col(bdf, ['wagonZone','wagon_zone','wagon_zone_id'], default='wagonZone')
    wagonx_col = safe_get_col(bdf, ['wagonX','wagon_x','wagon_xx'], default='wagonX')
    wagony_col = safe_get_col(bdf, ['wagonY','wagon_y','wagon_yy'], default='wagonY')
    run_col = safe_get_col(bdf, ['batruns','batsman_runs','score','runs'], default='batruns')
    line_col = safe_get_col(bdf, ['line'], default='line')
    length_col = safe_get_col(bdf, ['length'], default='length')
    
    # Basic sanity
    if match_col is None:
        st.error("No match id column found (expecting 'p_match' or 'match_id').")
        st.stop()
    if batter_col is None or bowler_col is None:
        st.error("Dataset must contain batter and bowler columns ('bat'/'batsman' and 'bowl'/'bowler').")
        st.stop()
    
    # match selector
    match_ids = sorted(bdf[match_col].dropna().unique().tolist())
    if not match_ids:
        st.error("No matches found in dataset.")
        st.stop()
    
    match_id = st.selectbox("Select Match ID", options=match_ids, index=0)
    match_rows = bdf[bdf[match_col] == match_id]
    if match_rows.empty:
        st.error("No rows for selected match.")
        st.stop()
    
    # metadata for header
    first_row = match_rows.iloc[0]
    batting_team = first_row.get(safe_get_col(bdf, ['team_bat','batting_team','team_batting']), "Unknown")
    bowling_team = first_row.get(safe_get_col(bdf, ['team_bowl','bowling_team','team_bow']), "Unknown")
    venue = first_row.get(safe_get_col(bdf, ['ground','venue']), "Unknown")
    st.markdown(f"**{batting_team}** vs **{bowling_team}**")
    st.markdown(f"**Venue:** {venue}")
    
    temp_df = match_rows.copy()
    
    # Derived/normalized numeric columns
    if run_col in temp_df.columns:
        temp_df[run_col] = pd.to_numeric(temp_df[run_col], errors='coerce').fillna(0).astype(int)
    else:
        temp_df['batruns'] = 0
        run_col = 'batruns'
    
    # out flag and dismissal text normalization (to know wickets)
    if 'out' in temp_df.columns:
        temp_df['out_flag'] = pd.to_numeric(temp_df['out'], errors='coerce').fillna(0).astype(int)
    else:
        temp_df['out_flag'] = 0
    
    if 'dismissal' in temp_df.columns:
        temp_df['dismissal_clean'] = temp_df['dismissal'].astype(str).str.lower().str.strip().replace({'nan':'','none':''})
    else:
        temp_df['dismissal_clean'] = ''
    
    special_runout_types = set([
        'run out', 'runout',
        'obstructing the field', 'obstructing thefield', 'obstructing',
        'retired out', 'retired not out (hurt)', 'retired not out', 'retired'
    ])
    
    temp_df['is_wkt'] = temp_df.apply(
        lambda r: 1 if (int(r.get('out_flag',0)) == 1 and str(r.get('dismissal_clean','')).strip() not in special_runout_types and str(r.get('dismissal_clean','')).strip() != '') else 0,
        axis=1
    )
    
    # UI: choose Batsman or Bowler
    option = st.selectbox("Select Analysis Dimension", ("Batsman Analysis", "Bowler Analysis"))
    
    # line & length maps (for pitch map grid)
    line_map = {
        'WIDE_OUTSIDE_OFFSTUMP': 0,
        'OUTSIDE_OFFSTUMP': 1,
        'ON_THE_STUMPS': 2,
        'DOWN_LEG': 3,
        'WIDE_DOWN_LEG': 4
    }
    length_map = {
        'SHORT': 0,
        'SHORT_OF_A_GOOD_LENGTH': 1,
        'GOOD_LENGTH': 2,
        'FULL': 3,
        'YORKER': 4
    }
    
    # ---------------------------
    # Batsman Analysis
    # ---------------------------
    if option == "Batsman Analysis":
        bat_choices = sorted([x for x in temp_df[batter_col].dropna().unique() if str(x).strip() not in ("", "0")])
        if not bat_choices:
            st.info("No batsmen found in this match.")
        else:
            batsman_selected = st.selectbox("Select Batsman", options=bat_choices, index=0)
            filtered_df = temp_df[temp_df[batter_col] == batsman_selected].copy()
    
            # Bowler selection (All + actual bowlers)
            bowl_opts = ["All"] + sorted([x for x in filtered_df[bowler_col].dropna().unique() if str(x).strip() not in ("", "0")])
            bowler_selected = st.selectbox("Select Bowler", options=bowl_opts, index=0)
    
            if bowler_selected != "All":
                final_df = filtered_df[filtered_df[bowler_col] == bowler_selected].copy()
            else:
                final_df = filtered_df.copy()
    
            # counts for scoring shots
            total_runs = int(final_df[run_col].sum())
            total_balls = int(final_df.shape[0])  # each row is a ball in match context
            strike_rate = (total_runs / total_balls * 100) if total_balls > 0 else 0.0
    
            # Scoring shot counts (0,1,2,3,4,6)
            counts = {k: int((final_df[run_col] == k).sum()) for k in [0,1,2,3,4,6]}
            counts_pct_balls = {k: (counts[k] / total_balls * 100) if total_balls>0 else 0.0 for k in counts}
    
            # Display header + compact stats
            st.markdown(f"### Analysis for Batsman: {batsman_selected}")
            if bowler_selected == "All":
                st.markdown("Against: All Bowlers")
            else:
                st.markdown(f"Against: {bowler_selected}")
    
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Summary**")
                st.write(f"Runs: {total_runs}")
                st.write(f"Balls: {total_balls}")
            with col2:
                # st.write("**Rates**")
                st.write(f"Strike Rate: {strike_rate:.2f}")
    
            # -------------------------
            # Scoring Shots - colorful tabular view
            # -------------------------
            scoring_table = pd.DataFrame([
                {"Shot": "0s", "Count": counts[0], "Pct (balls)": f"{counts_pct_balls[0]:.2f}%"},
                {"Shot": "1s", "Count": counts[1], "Pct (balls)": f"{counts_pct_balls[1]:.2f}%"},
                {"Shot": "2s", "Count": counts[2], "Pct (balls)": f"{counts_pct_balls[2]:.2f}%"},
                {"Shot": "3s", "Count": counts[3], "Pct (balls)": f"{counts_pct_balls[3]:.2f}%"},
                {"Shot": "4s", "Count": counts[4], "Pct (balls)": f"{counts_pct_balls[4]:.2f}%"},
                {"Shot": "6s", "Count": counts[6], "Pct (balls)": f"{counts_pct_balls[6]:.2f}%"},
            ])
            def style_scoring(df_table):
                sty = df_table.style.set_table_styles([
                    {'selector':'thead','props':[('background-color','#fff7e6'),('color','#000'),('font-weight','600')]},
                ]).format({"Count":"{:,}"})
                color_map = {"0s":"#f0f0f0","1s":"#cfe8ff","2s":"#cfe8ff","3s":"#cfe8ff","4s":"#fff3b0","6s":"#e6d0ff"}
                for i, v in enumerate(df_table['Shot']):
                    clr = color_map.get(v, "#ffffff")
                    sty = sty.set_properties(subset=pd.IndexSlice[i, :], **{"background-color": clr, "color":"#000"})
                return sty
    
            st.markdown("#### Scoring Shots")
            st.dataframe(style_scoring(scoring_table), use_container_width=True)
    
            # -------------------------
            # Wagon Wheel (8 sectors) - clock-accurate sector centers + proper mirroring for LHB
            # -------------------------
            sector_names = {
                1: "Third Man",
                2: "Point",
                3: "Covers",
                4: "Mid Off",
                5: "Mid On",
                6: "Mid Wicket",
                7: "Square Leg",
                8: "Fine Leg"
            }
    
            # Base angles for RHB per your clock instruction: Third Man centered at 11:15 (112.5°)
            # and then each sector moves 45° toward the left (counter-clockwise)
            base_angles = {
                1: 112.5,  # Third Man
                2: 157.5,  # Point
                3: 202.5,  # Covers
                4: 247.5,  # Mid Off
                5: 292.5,  # Mid On
                6: 337.5,  # Mid Wicket
                7: 22.5,   # Square Leg
                8: 67.5    # Fine Leg
            }
    
            def get_sector_angle_requested(zone, batting_style):
                """Return sector center in radians.
    
                IMPORTANT: For left-handers we must *mirror* the chart across vertical axis so that
                Third Man <-> Fine Leg, Point <-> Square Leg, Covers <-> Mid Wicket, Mid Off <-> Mid On.
                This is achieved by reflecting the angle across the vertical axis: angle -> (180 - angle) % 360.
                """
                angle = float(base_angles.get(int(zone), 0.0))
                if str(batting_style).strip().upper().startswith('L'):
                    angle = (180.0 - angle) % 360.0
                return math.radians(angle)
    
            def draw_cricket_field_with_run_totals_requested(final_df_local, batsman_name):
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.set_aspect('equal')
                ax.axis('off')
    
                # Field base
                ax.add_patch(Circle((0, 0), 1, fill=True, color='#228B22', alpha=1))
                ax.add_patch(Circle((0, 0), 1, fill=False, color='black', linewidth=3))
                ax.add_patch(Circle((0, 0), 0.5, fill=True, color='#66bb6a'))
                ax.add_patch(Circle((0, 0), 0.5, fill=False, color='white', linewidth=1))
    
                # pitch rectangle approx
                pitch_rect = Rectangle((-0.04, -0.08), 0.08, 0.16, color='tan', alpha=1, zorder=8)
                ax.add_patch(pitch_rect)
    
                # radial sector lines (8)
                angles = np.linspace(0, 2*np.pi, 9)[:-1]
                for angle in angles:
                    x = math.cos(angle)
                    y = math.sin(angle)
                    ax.plot([0, x], [0, y], color='white', alpha=0.25, linewidth=1)
    
                # prepare data (runs, fours, sixes by sector)
                tmp = final_df_local.copy()
                if wagon_zone_col in tmp.columns:
                    tmp['wagon_zone_int'] = pd.to_numeric(tmp[wagon_zone_col], errors='coerce').astype('Int64')
                else:
                    tmp['wagon_zone_int'] = pd.Series(dtype='Int64')
    
                runs_by_zone = tmp.groupby('wagon_zone_int')[run_col].sum().to_dict()
                fours_by_zone = tmp.groupby('wagon_zone_int')[run_col].apply(lambda s: int((s==4).sum())).to_dict()
                sixes_by_zone = tmp.groupby('wagon_zone_int')[run_col].apply(lambda s: int((s==6).sum())).to_dict()
                total_runs_in_wagon = sum(int(v) for v in runs_by_zone.values())
    
                # Title
                title_text = f"{batsman_name}'s Scoring Zones"
                plt.title(title_text, pad=20, color='white', size=14, fontweight='bold')
    
                # Place % runs and runs in each sector using sector centers
                for zone in range(1, 9):
                    batting_style_val = None
                    if bat_hand_col in tmp.columns and not tmp[bat_hand_col].dropna().empty:
                        batting_style_val = tmp[bat_hand_col].dropna().iloc[0]
                    angle_mid = get_sector_angle_requested(zone, batting_style_val)
                    x = 0.60 * math.cos(angle_mid)
                    y = 0.60 * math.sin(angle_mid)
                    runs = int(runs_by_zone.get(zone, 0))
                    pct = (runs / total_runs_in_wagon * 100) if total_runs_in_wagon > 0 else 0.0
                    pct_str = f"{pct:.2f}%"
    
                    # main labels
                    ax.text(x, y+0.03, pct_str, ha='center', va='center', color='white', fontweight='bold', fontsize=18)
                    ax.text(x, y-0.03, f"{runs} runs", ha='center', va='center', color='white', fontsize=10)
    
                    # fours & sixes below
                    fours = int(fours_by_zone.get(zone, 0))
                    sixes = int(sixes_by_zone.get(zone, 0))
                    ax.text(x, y-0.12, f"4s: {fours}  6s: {sixes}", ha='center', va='center', color='white', fontsize=9)
    
                    # sector name slightly farther out
                    sx = 0.80 * math.cos(angle_mid)
                    sy = 0.80 * math.sin(angle_mid)
                    ax.text(sx, sy, sector_names.get(zone, f"Sector {zone}"), ha='center', va='center', color='white', fontsize=8)
    
                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.2, 1.2)
                plt.tight_layout(pad=0)
                return fig
    
            # run the wagon wheel drawing if column exists
            if wagon_zone_col not in final_df.columns:
                st.info("wagonZone column not available for wagon wheel.")
            else:
                ww_df = final_df.copy()
                ww_df['wagon_zone_int'] = pd.to_numeric(ww_df[wagon_zone_col], errors='coerce').astype('Int64')
                grouped = ww_df.groupby('wagon_zone_int').agg(
                    runs = (run_col, 'sum'),
                    balls = (run_col, 'size'),
                    fours = (run_col, lambda s: int((s==4).sum())),
                    sixes = (run_col, lambda s: int((s==6).sum()))
                ).reset_index().rename(columns={'wagon_zone_int':'sector'})
                all_sectors = pd.DataFrame({"sector": list(range(1,9))})
                grouped = all_sectors.merge(grouped, on='sector', how='left').fillna(0)
                grouped[['runs','balls','fours','sixes']] = grouped[['runs','balls','fours','sixes']].astype(int)
                total_runs = grouped['runs'].sum()
                grouped['pct_runs'] = grouped['runs'].apply(lambda x: round((x / total_runs * 100) if total_runs>0 else 0.0,2))
    
                ww_display = grouped.copy()
                ww_display['Sector Name'] = ww_display['sector'].map({
                    1:"Third Man",2:"Point",3:"Covers",4:"Mid Off",5:"Mid On",6:"Mid Wicket",7:"Square Leg",8:"Fine Leg"
                })
                ww_display = ww_display[['sector','Sector Name','runs','pct_runs','fours','sixes','balls']].rename(columns={
                    'sector':'Sector','runs':'Runs','pct_runs':'Pct of Runs','fours':'4s','sixes':'6s','balls':'Balls'
                })
                ww_display['Pct of Runs'] = ww_display['Pct of Runs'].apply(lambda x: f"{x:.2f}%")
    
                # draw figure using requested style and mapping
                fig = draw_cricket_field_with_run_totals_requested(final_df, batsman_selected)
                safe_st_pyplot(fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
    
                # Label below wagon chart indicating RHB/LHB
                batting_style_display = None
                if bat_hand_col in final_df.columns and not final_df[bat_hand_col].dropna().empty:
                    batting_style_display = final_df[bat_hand_col].dropna().iloc[0]
                side_label = "LHB" if batting_style_display and str(batting_style_display).strip().upper().startswith('L') else "RHB"
                st.markdown(f"<div style='text-align:center; margin-top:6px;'><strong>{batsman_selected}'s Wagon Chart ({side_label})</strong></div>", unsafe_allow_html=True)
    
                # show table below wheel
                # st.dataframe(ww_display.style.set_table_styles([
                #     {"selector":"thead th", "props":[("background-color","#e6f2ff"),("font-weight","600")]},
                # ]), use_container_width=True)
    
            # -------------------------
            # Pitchmaps below Wagon: two heatmaps (dots vs scoring) - increased height
            # -------------------------
            run_grid = np.zeros((5,5), dtype=float)
            dot_grid = np.zeros((5,5), dtype=int)
    
            if line_col in final_df.columns and length_col in final_df.columns:
                plot_df = final_df[[line_col,length_col,run_col]].copy().dropna(subset=[line_col,length_col])
                for _, r in plot_df.iterrows():
                    li = line_map.get(r[line_col], None)
                    le = length_map.get(r[length_col], None)
                    if li is None or le is None:
                        continue
                    runs_here = int(r[run_col])
                    run_grid[le, li] += runs_here
                    if runs_here == 0:
                        dot_grid[le, li] += 1
                    
                # import base64
                # from io import BytesIO
                # from PIL import Image
                
                # # Pixel height for pitchmaps (change this value to whatever visible height you want)
                # HEIGHT_PITCHMAP_PX = 1600

                # Assuming dot_grid and run_grid are 5x5 numpy arrays already defined
                
                st.markdown("### Pitchmaps")
                c1, c2 = st.columns([1, 1])
                
                with c1:
                    st.markdown("**Dot Balls**")
                    fig1, ax1 = plt.subplots(figsize=(8, 14), dpi=150)  # Increased height from 10 to 12
                    im1 = ax1.imshow(dot_grid, origin='lower', cmap='Blues')
                    ax1.set_xticks(range(5))
                    ax1.set_yticks(range(5))
                    ax1.set_xticklabels(['Wide Out Off', 'Outside Off', 'On Stumps', 'Down Leg', 'Wide Down Leg'],
                                        rotation=45, ha='right')
                    ax1.set_yticklabels(['Short', 'Back of Length', 'Good', 'Full', 'Yorker'])
                    for i in range(5):
                        for j in range(5):
                            ax1.text(j, i, int(dot_grid[i, j]), ha='center', va='center', color='black', fontsize=12)
                    fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
                    plt.tight_layout(pad=3.0)
                    st.pyplot(fig1)
                
                with c2:
                    st.markdown("**Scoring Balls**")
                    fig2, ax2 = plt.subplots(figsize=(8, 14), dpi=150)  # Increased height from 10 to 12
                    im2 = ax2.imshow(run_grid, origin='lower', cmap='Reds')
                    ax2.set_xticks(range(5))
                    ax2.set_yticks(range(5))
                    ax2.set_xticklabels(['Wide Out Off', 'Outside Off', 'On Stumps', 'Down Leg', 'Wide Down Leg'],
                                        rotation=45, ha='right')
                    ax2.set_yticklabels(['Short', 'Back of Length', 'Good', 'Full', 'Yorker'])
                    for i in range(5):
                        for j in range(5):
                            ax2.text(j, i, int(run_grid[i, j]), ha='center', va='center', color='black', fontsize=12)
                    fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                    plt.tight_layout(pad=3.0)
                    st.pyplot(fig2)

            else:
                st.info("Pitchmap requires both 'line' and 'length' columns in dataset; skipping pitchmaps.")
    
    # ---------------------------
    # Bowler Analysis (match-level)
    # ---------------------------
    elif option == "Bowler Analysis":
        bowler_choices = sorted([x for x in temp_df[bowler_col].dropna().unique() if str(x).strip() not in ("","0")])
        if not bowler_choices:
            st.info("No bowlers found in this match.")
        else:
            bowler_selected = st.selectbox("Select Bowler", options=bowler_choices, index=0)
            filtered_df = temp_df[temp_df[bowler_col] == bowler_selected].copy()
    
            # Legal balls definition: both wide & noball == 0
            filtered_df['noball'] = pd.to_numeric(filtered_df.get('noball',0), errors='coerce').fillna(0).astype(int)
            filtered_df['wide'] = pd.to_numeric(filtered_df.get('wide',0), errors='coerce').fillna(0).astype(int)
            filtered_df['legal_ball'] = ((filtered_df['noball'] == 0) & (filtered_df['wide'] == 0)).astype(int)
    
            # runs conceded should be sum of score/batruns when byes & legbyes ==0
            if 'score' in filtered_df.columns:
                cond = (~filtered_df.get('byes', 0).astype(bool)) & (~filtered_df.get('legbyes',0).astype(bool))
                runs_given = int(filtered_df.loc[cond, 'score'].sum() if 'score' in filtered_df.columns else 0)
            elif 'batruns' in filtered_df.columns:
                cond = (~filtered_df.get('byes', 0).astype(bool)) & (~filtered_df.get('legbyes',0).astype(bool))
                runs_given = int(filtered_df.loc[cond, 'batruns'].sum())
            else:
                runs_given = int(filtered_df.get('bowlruns', filtered_df.get('total_runs', 0)).sum())
    
            balls_bowled = int(filtered_df['legal_ball'].sum())
            wickets = int(filtered_df['is_wkt'].sum()) if 'is_wkt' in filtered_df.columns else 0
            econ = (runs_given * 6.0 / balls_bowled) if balls_bowled>0 else float('nan')
            avg = (runs_given / wickets) if wickets>0 else float('nan')
            sr = (balls_bowled / wickets) if wickets>0 else float('nan')
    
            st.markdown(f"### Bowling Analysis for {bowler_selected}")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Runs conceded: {runs_given}")
                st.write(f"Balls: {balls_bowled}")
                st.write(f"Wickets: {wickets}")
            with col2:
                st.write(f"Econ: {econ:.2f}" if not np.isnan(econ) else "Econ: -")
                st.write(f"Avg: {avg:.2f}" if not np.isnan(avg) else "Avg: -")
                st.write(f"SR: {sr:.2f}" if not np.isnan(sr) else "SR: -")
    
            # pitchmap and heatmaps for bowler (reuse same mapping)
            if line_col in filtered_df.columns and length_col in filtered_df.columns:
                plot_df = filtered_df[[line_col,length_col,'batruns','score']].copy()
                plot_df['run_val'] = plot_df.get('batruns', plot_df.get('score', 0))
                run_grid_b = np.zeros((5,5), dtype=float)
                dot_grid_b = np.zeros((5,5), dtype=int)
                for _, r in plot_df.dropna(subset=[line_col,length_col]).iterrows():
                    li = line_map.get(r[line_col], None)
                    le = length_map.get(r[length_col], None)
                    if li is None or le is None:
                        continue
                    rv = 0
                    try:
                        rv = int(r['run_val'])
                    except:
                        rv = 0
                    run_grid_b[le, li] += rv
                    if rv == 0:
                        dot_grid_b[le, li] += 1
    
                st.markdown("### Bowler Pitchmaps")
                c1, c2 = st.columns(2)
                with c1:
                    st.write("Dot Balls")
                    fig1, ax1 = plt.subplots(figsize=(7, 12))
                    im1 = ax1.imshow(dot_grid_b, origin='lower', cmap='Blues')
                    ax1.set_xticks(range(5)); ax1.set_yticks(range(5))
                    ax1.set_xticklabels(['Wide Out Off','Outside Off','On Stumps','Down Leg','Wide Down Leg'], rotation=45, ha='right')
                    ax1.set_yticklabels(['Short','Back of Length','Good','Full','Yorker'])
                    for i in range(5):
                        for j in range(5):
                            ax1.text(j, i, int(dot_grid_b[i,j]), ha='center', va='center', color='black')
                    fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
                    safe_st_pyplot(fig1, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
                with c2:
                    st.write("Runs Conceded")
                    fig2, ax2 = plt.subplots(figsize=(7, 12))
                    im2 = ax2.imshow(run_grid_b, origin='lower', cmap='Reds')
                    ax2.set_xticks(range(5)); ax2.set_yticks(range(5))
                    ax2.set_xticklabels(['Wide Out Off','Outside Off','On Stumps','Down Leg','Wide Down Leg'], rotation=45, ha='right')
                    ax2.set_yticklabels(['Short','Back of Length','Good','Full','Yorker'])
                    for i in range(5):
                        for j in range(5):
                            ax2.text(j, i, int(run_grid_b[i,j]), ha='center', va='center', color='black')
                    fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                    safe_st_pyplot(fig2, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
    
    else:
        st.info("Choose a valid analysis option.")
# SEARCH FOR THIS LINE IN YOUR FILE:
# st.header("Strength and Weakness Analysis")
# 
# Then REPLACE the entire section from that line down to the next major section
# with the code below:
else:
    st.header("Strength and Weakness Analysis")
    pdf=df
    player_name = st.selectbox("Search for a player", idf['batsman'].unique())
    
    # Dropdown for Batting or Bowling selection
    option = st.selectbox("Select Role", ("Batting", "Bowling"))
    
    if option == "Batting":          
        # Apply the function to add the 'bowl_kind' column
        
        result_df = pd.DataFrame()
    
        final_df = pdf[pdf['batsman'] == player_name]
        result_df = pd.DataFrame()
        i = 0
        
        # Loop over pace and spin bowling types
        for bowl_kind in ['pace bowler', 'spin bowler']:
            temp_df = pdf[pdf['batsman'] == player_name]  # Filter data for the selected batsman
            
            # Filter for the specific 'bowl_kind'
            temp_df = temp_df[temp_df['bowl_kind'] == bowl_kind]
            
            # Apply the cumulative function (bcum)
            temp_df = cumulator(temp_df)
            
            # If the DataFrame is empty after applying `bcum`, skip this iteration
            if temp_df.empty:
                continue
            
            # Add the bowl_kind column
            temp_df['bowl_kind'] = bowl_kind
            
            # Reorder columns to make 'bowl_kind' the first column
            cols = temp_df.columns.tolist()
            new_order = ['bowl_kind'] + [col for col in cols if col != 'bowl_kind']
            temp_df = temp_df[new_order]
            
            # Concatenate results into result_df
            if i == 0:
                result_df = temp_df
                i += 1
            else:
                result_df = pd.concat([result_df, temp_df], ignore_index=True)
        
        # Display the final result_df
        result_df = result_df.drop(columns=['batsman', 'debut_year', 'final_year','hundreds','fifties','thirties','highest_score','matches'])
        result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
        columns_to_convert = ['RUNS']
        
        # Fill NaN values with 0
        # result_df[columns_to_convert] = result_df[columns_to_convert].fillna(0)
        
        # Convert the specified columns to integer type
        # result_df[columns_to_convert] = result_df[columns_to_convert].astype(int)
        result_df = round_up_floats(result_df)
        
        # Specify the desired order with 'bowl_kind' first
        cols = result_df.columns.tolist()
        # new_order = ['BOWL KIND', 'INNINGS'] + [col for col in cols if col not in ['BOWL KIND', 'INNINGS']]
        
        # Reindex the DataFrame with the new column order
        # result_df = result_df[new_order]
        result_df['BOWL KIND'] = result_df['BOWL KIND'].str.capitalize()
        st.markdown("### Performance Against Bowling Types (Pace vs Spin)")
        st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
        
        result_df = pd.DataFrame()
        i = 0
        allowed_bowling_styles=['Off-break', 'Left-arm medium fast', 'Right-arm medium fast',
       'Right-arm fast', 'Slow left-arm orthodox',
       'Left-arm fast',
       'Leg-break googly', 'Right-arm medium',
       'Left-arm medium',
       'Left-arm wrist spin',
       'Leg-break','Leg-spin',
       'Left-arm medium fast and slow left-arm orthodox',
       'Off-break and slow left-arm orthodox',
       'Right-arm medium fast and off-break']
        for bowling_style in allowed_bowling_styles:
            temp_df = pdf[pdf['batsman'] == player_name]  # Filter data for the selected batsman
            
            # Filter for the specific bowling style
            temp_df = temp_df[temp_df['bowling_style'] == bowling_style]
            
            # Apply the cumulative function (bcum)
            temp_df = cumulator(temp_df)
            
            # If the DataFrame is empty after applying `bcum`, skip this iteration
            if temp_df.empty:
                continue
            
            # Add the bowling style column
            temp_df['bowling_style'] = bowling_style
            
            # Reorder columns to make 'bowling_style' the first column
            cols = temp_df.columns.tolist()
            new_order = ['bowling_style'] + [col for col in cols if col != 'bowling_style']
            temp_df = temp_df[new_order]
            
            # Concatenate results into result_df
            if i == 0:
                result_df = temp_df
                i += 1
            else:
                result_df = pd.concat([result_df, temp_df], ignore_index=True)
        
        # Display the final result_df
        result_df = result_df.drop(columns=['batsman', 'debut_year', 'final_year','hundreds','fifties','thirties','highest_score','matches'])
        result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
        columns_to_convert = ['RUNS']

        # Fill NaN values with 0
        result_df[columns_to_convert] = result_df[columns_to_convert].fillna(0)

        # Convert the specified columns to integer type
        result_df[columns_to_convert] = result_df[columns_to_convert].astype(int)
        result_df = round_up_floats(result_df)
        cols = result_df.columns.tolist()

        # Specify the desired order with 'bowling_style' first
        new_order = ['BOWLING STYLE', 'INNINGS'] + [col for col in cols if col not in ['BOWLING STYLE','INNINGS',]]

        # Reindex the DataFrame with the new column order
        result_df = result_df[new_order]

        st.markdown("### Performance Against Bowling Styles")
        st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
        

        @st.cache_data
        def get_sector_angle(zone, batting_style):
            # Set base angle for RHB and mirror for LHB
            base_angles = {
                1: 45,   # Third Man
                2: 90,   # Point
                3: 135,  # Covers
                4: 180,  # Mid-off
                5: 225,  # Mid-on
                6: 270,  # Mid-wicket
                7: 315,  # Square leg
                8: 0     # Fine leg (360 degrees or 0 degrees)
            }
            angle = base_angles[zone]
            
            # Adjust for left-handed batsman by mirroring the sectors
            if batting_style == 'LHB':
                angle = (180 + angle) % 360
            
            return np.radians(angle)
        @st.cache_data
        def draw_cricket_field_with_run_totals(final_df):
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_aspect('equal')
            ax.axis('off')

            # Draw field elements
            boundary = plt.Circle((0, 0), 1, fill=True, color='#228B22', alpha=1)
            boundary_line = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=4)
            inner_circle = plt.Circle((0, 0), 0.5, fill=True, color='#90EE90')
            inner_circle_line = plt.Circle((0, 0), 0.5, fill=False, color='white', linewidth=1)
            
            ax.add_patch(boundary)
            ax.add_patch(boundary_line)
            ax.add_patch(inner_circle)
            ax.add_patch(inner_circle_line)
            
            # Draw sector lines
            angles = np.linspace(0, 2*np.pi, 9)[:-1]
            for angle in angles:
                x = np.cos(angle)
                y = np.sin(angle)
                ax.plot([0, x], [0, y], color='white', alpha=0.2, linewidth=1)
            
            # Draw pitch rectangle
            pitch_rect = plt.Rectangle((-0.04, -0.08), 0.08, 0.16, color='tan', alpha=1)
            ax.add_patch(pitch_rect)
            
            # Determine total runs per sector
            batting_style = final_df['batting_style'].iloc[0]  # Assume all rows have the same batting style
            total_runs = final_df.groupby('wagonZone')['batsman_runs'].sum()

            # Add title based on batting style
            title = "Right-handed Batsman" if batting_style == 'RHB' else "Left-handed Batsman"
            plt.title(title, pad=20, color='white', size=12, fontweight='bold')
            
            # Label each sector with the total runs scored
            for zone in range(1, 9):
                angle = get_sector_angle(zone, batting_style) + np.radians(22.5)
                x = 0.65 * np.cos(angle)  # Reduced radius for labels to be inside the sector
                y = 0.65 * np.sin(angle)
                
                # Get the total runs for the zone (default to 0 if not present)
                runs = total_runs.get(zone, 0)
                
                # Display runs in the sector
                ax.text(x, y, str(runs), ha='center', va='center', color='white', fontweight='bold', size=20)
            
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            plt.tight_layout(pad=0)
            
            return fig
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Wagon Chart**")
            # Display the wicket count grid
            fig = draw_cricket_field_with_run_totals(final_df)
            st.pyplot(fig, use_container_width=True)
            

        with col2:
             # Set up line and length mapping
            import plotly.graph_objects as go          
            line_positions = {
                'WIDE_OUTSIDE_OFFSTUMP': 0,
                'OUTSIDE_OFFSTUMP': 1,
                'ON_THE_STUMPS':2,
                'DOWN_LEG': 3,
                'WIDE_DOWN_LEG':4 
            }


            length_positions = {
                'SHORT': 0 ,
                'SHORT_OF_A_GOOD_LENGTH': 1,
                'GOOD_LENGTH': 2,
                'FULL': 3,
                'YORKER': 4,
                'FULL_TOSS': 4
            } 
            run_count_grid = np.zeros((5, 5))
            wicket_count_grid = np.zeros((5, 5))
            
            # Fill the grids based on final_df data
            for _, row in final_df.iterrows():
                line = row['line']
                length = row['length']
                runs = row['batsman_runs']
                is_wkt = row['is_wkt']
                
                # Identify the correct cell for run count and wicket count
                line_idx = line_positions.get(line, 2)  # Default to 'On Stumps' if line not found
                length_idx = length_positions.get(length, 2)  # Default to 'Good Length' if length not found
                
                # Update run counts and wicket counts
                run_count_grid[length_idx, line_idx] += runs
                
                if is_wkt == 1:
                    wicket_count_grid[length_idx, line_idx] += 1
            
            # Labels for line and length positions
            line_labels = ['Wide Outside Off', 'Outside Off', 'On Stumps', 'Outside Leg', 'Wide Outside Leg']
            length_labels = ['Short', 'Back of Length', 'Good Length', 'Full', 'Yorker']
            
            # Function to create heatmap figure for a 5x5 grid
            @st.cache_data
            def create_heatmap(grid, title, annotations):
                fig = go.Figure(
                    data=go.Heatmap(
                        z=grid,
                        colorscale='Reds',
                        colorbar=dict(title=title)
                    )
                )
                # Add black text annotations to show actual counts
                for i in range(5):
                    for j in range(5):
                        fig.add_annotation(
                            x=j, y=i,
                            text=f'{annotations[i, j]}',
                            showarrow=False,
                            font=dict(color="black", size=12)
                        )
                
                # Update layout for vertical orientation and labels
                fig.update_layout(
                    xaxis=dict(showgrid=False, tickvals=list(range(5)), ticktext=line_labels, title="Line"),
                    yaxis=dict(showgrid=False, tickvals=list(range(5)), ticktext=length_labels, title="Length"),
                    height=700, width=300  # Adjusted size for compact display
                )
                return fig
            
            

            st.markdown("**Runs Scored**")
            # Display the runs count grid
            st.plotly_chart(create_heatmap(run_count_grid, "Runs", run_count_grid), use_container_width=True)
         
        @st.cache_data
        def apply_length_offset(y_value, offset_range=(-0.95, 0.95), boundary=(-2, 10)):
            offset = np.random.uniform(offset_range[0], offset_range[1])
            if boundary[0] <= y_value + offset <= boundary[1]:
                return y_value + offset
            return y_value
        
        # Function to apply a small random offset to line while keeping length accurate
        @st.cache_data
        def apply_line_offset(x_value, offset_range=(-0.05, 0.05), boundary=(-0.5, 0.5)):
            offset = np.random.uniform(offset_range[0], offset_range[1])
            if boundary[0] <= x_value + offset <= boundary[1]:
                return x_value + offset
            return x_value
        
        # Define pitch zones and their respective boundaries
        zones = {
            'SHORT': (8, 10),
            'SHORT_OF_A_GOOD_LENGTH': (6, 8),
            'GOOD_LENGTH': (4, 6),
            'FULL': (2, 4),
            'YORKER': (0, 2),
            'FULL_TOSS': (-2, 0)
        }
        
        line_positions = {
            'WIDE_OUTSIDE_OFFSTUMP': 0.25,
            'OUTSIDE_OFFSTUMP': 0.15,
            'ON_THE_STUMPS': 0,
            'DOWN_LEG': -0.15,
            'WIDE_DOWN_LEG': -0.25
        }
        
        length_positions = {
            'SHORT': 9,
            'SHORT_OF_A_GOOD_LENGTH': 7,
            'GOOD_LENGTH': 5,
            'FULL': 3,
            'YORKER': 1,
            'FULL_TOSS': -1
        }
        
        # Function to create a 3D pitch map based on handedness
        @st.cache_data
        def create_pitch_map(data, handedness):
            fig = go.Figure()
        
            # Define stumps and bails positions
            stump_positions = [-0.05, 0, 0.05]
            stump_height = 0.3
            stump_thickness = 2
            bail_height = stump_height + 0.002
        
            # Add stumps and bails to the figure
            for x_pos in stump_positions:
                fig.add_trace(go.Scatter3d(
                    x=[x_pos, x_pos],
                    y=[0, 0],
                    z=[0, stump_height],
                    mode='lines',
                    line=dict(color='black', width=stump_thickness),
                    showlegend=False
                ))
        
            fig.add_trace(go.Scatter3d(
                x=[stump_positions[0], stump_positions[1]],
                y=[0, 0],
                z=[bail_height, bail_height],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))
            fig.add_trace(go.Scatter3d(
                x=[stump_positions[1], stump_positions[2]],
                y=[0, 0],
                z=[bail_height, bail_height],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))
        
            # Add pitch zones
            for zone_name, (y_min, y_max) in zones.items():
                fig.add_trace(go.Scatter3d(
                    x=[-0.5, 0.5, 0.5, -0.5, -0.5],
                    y=[y_min, y_min, y_max, y_max, y_min],
                    z=[0, 0, 0, 0, 0],
                    mode='lines+markers',
                    line=dict(color="gray", width=2),
                    marker=dict(size=0.1, opacity=0.2),
                    showlegend=False
                ))
        
            # Add length labels on the side of the pitch
            for length, y_position in length_positions.items():
                fig.add_trace(go.Scatter3d(
                    x=[0.6],  # Position length labels to the side of the pitch
                    y=[y_position],
                    z=[0],
                    mode='text',
                    text=[length],
                    textposition="middle right",
                    textfont=dict(size=10, color="black"),
                    showlegend=False
                ))
        
            # Set mirroring factor based on handedness
            mirror_factor = -1 if handedness == 'LHB' else 1 if handedness == 'RHB' else 0
        
            # Filter data to only include wicket balls
            wicket_data = data[data['bowler_wkt'] == 1]
        
            # Plot wicket balls
            for index, row in wicket_data.iterrows():
                if pd.isna(row['line']) or pd.isna(row['length']):
                    continue  # Skip missing data
                x_base = line_positions.get(row['line'], 0) * mirror_factor
                y_base = length_positions.get(row['length'], 5)
        
                # Apply offset to length while keeping line accurate
                x_pos = apply_line_offset(x_base, boundary=(-0.5, 0.5))
                y_pos = apply_length_offset(y_base, boundary=(-2, 10))
                z_pos = 0
        
                # Plot the wicket ball
                fig.add_trace(go.Scatter3d(
                    x=[x_pos],
                    y=[y_pos],
                    z=[z_pos],
                    mode='markers',
                    marker=dict(size=5, color='red', opacity=1),
                    hoverinfo="text",
                    text=f"Line: {row['line']}<br>Length: {row['length']}<br>Runs: {row['batsman_runs']} - Wicket"
                ))
        
            # Final layout settings
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title='X-axis', range=[-1, 1]),
                    yaxis=dict(title='Y-axis', range=[-2, 10]),
                    zaxis=dict(title='Z-axis (Height)', range=[0, 2]),
                ),
                width=1200,
                height=1000,
                showlegend=False
            )
        
            return fig


        bat_hand = final_df['batting_style'].iloc[0]
        # Display each plot in the respective column
        with col1:
            st.markdown("**Wickets vs Pace**")
            if pace_df.empty:
                st.write("No data available")
            else:
                st.plotly_chart(create_pitch_map(pace_df, bat_hand), use_container_width=True)
        
        with col2:
            st.markdown("**Wickets vs Spin**")
            if spin_df.empty:
                st.write("No data available")
            else:
                st.plotly_chart(create_pitch_map(spin_df, bat_hand), use_container_width=True)
                
                
    else:
        final_df=pdf[pdf['bowler']==player_name]  
        allowed_batting_styles = ['LHB', 'RHB']  # Define the two batting styles
        result_df = pd.DataFrame()
        temp_df = pdf[pdf['bowler'] == player_name]
        if temp_df.empty :
            st.markdown('Bowling stats do not exist')
        else:
            # Loop over left-hand and right-hand batting styles
            for bat_style in allowed_batting_styles:
                temp_df = pdf[pdf['bowler'] == player_name]  # Filter data for the selected bowler
                
                # Filter for the specific batting style
                temp_df = temp_df[temp_df['batting_style'] == bat_style]
                
                # Apply the cumulative function (bcum) for bowling
                temp_df = bowlerstat(temp_df)
                
                # If the DataFrame is empty after applying bcum, skip this iteration
                if temp_df.empty:
                    continue
                
                # Add the batting style as a column for later distinction
                temp_df['batting_style'] = bat_style
                
                # Concatenate results into result_df
                result_df = pd.concat([result_df, temp_df], ignore_index=True)
        
            # Drop unwanted columns from the result DataFrame
            result_df = result_df.drop(columns=['bowler'])
        
            # Standardize column names
            result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
            
            # Convert the relevant columns to integers and fill NaN values
            columns_to_convert = ['WKTS']
            result_df[columns_to_convert] = result_df[columns_to_convert].fillna(0).astype(int)
            result_df = round_up_floats(result_df)
            cols = result_df.columns.tolist()
              
              # Specify the desired order with 'phase' first
            new_order = ['BATTING STYLE'] + [col for col in cols if col not in 'BATTING STYLE']
            result_df = result_df[new_order]
        
            # Display the final table
            st.markdown("### Cumulative Bowling Performance Against Batting Styles")
            st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
        if not final_df.empty:     
            st.markdown("### Pitch Map vs Batting Styles")
            
            zones = {
                'SHORT': (8, 10),
                'SHORT_OF_A_GOOD_LENGTH': (6, 8),
                'GOOD_LENGTH': (4, 6),
                'FULL': (2, 4),
                'YORKER': (0, 2),
                'FULL_TOSS': (-2, 0)
            }

            line_positions = {
                'WIDE_OUTSIDE_OFFSTUMP': -0.25,
                'OUTSIDE_OFFSTUMP': -0.15,
                'ON_THE_STUMPS': 0,
                'DOWN_LEG': 0.15,
                'WIDE_DOWN_LEG': 0.25
            }


            length_positions = {
                'SHORT': 9,
                'SHORT_OF_A_GOOD_LENGTH': 7,
                'GOOD_LENGTH': 5,
                'FULL': 3,
                'YORKER': 1,
                'FULL_TOSS': -1
            } 
            
            # Function to apply a small random offset to length while keeping line accurate
            @st.cache_data
            def apply_length_offset(y_value, offset_range=(-0.95, 0.95), boundary=(-2, 10)):
                offset = np.random.uniform(offset_range[0], offset_range[1])
                if boundary[0] <= y_value + offset <= boundary[1]:
                    return y_value + offset
                return y_value
            @st.cache_data
            def apply_line_offset(x_value, offset_range=(-0.05, 0.05), boundary=(-0.5, 0.5)):
                offset = np.random.uniform(offset_range[0], offset_range[1])
                if boundary[0] <= x_value + offset <= boundary[1]:
                    return x_value + offset
                return x_value
            
            # Set up two columns for LHB and RHB views
            col1, col2 = st.columns(2)
            lhb_data = final_df[final_df['batting_style'] == 'LHB']
            rhb_data = final_df[final_df['batting_style'] == 'RHB']
            
            # Function to create a 3D pitch map based on handedness
            @st.cache_data
            def create_pitch_map(data, handedness):
                fig = go.Figure()
            
                # Define stumps and bails
                stump_positions = [-0.05, 0, 0.05]
                stump_height = 0.3
                stump_thickness = 2
                bail_height = stump_height + 0.002
            
                # Add stumps
                for x_pos in stump_positions:
                    fig.add_trace(go.Scatter3d(
                        x=[x_pos, x_pos],
                        y=[0, 0],
                        z=[0, stump_height],
                        mode='lines',
                        line=dict(color='black', width=stump_thickness),
                        showlegend=False
                    ))
            
                # Add bails
                fig.add_trace(go.Scatter3d(
                    x=[stump_positions[0], stump_positions[1]],
                    y=[0, 0],
                    z=[bail_height, bail_height],
                    mode='lines',
                    line=dict(color='black', width=2),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter3d(
                    x=[stump_positions[1], stump_positions[2]],
                    y=[0, 0],
                    z=[bail_height, bail_height],
                    mode='lines',
                    line=dict(color='black', width=2),
                    showlegend=False
                ))
            
                # Add pitch zones
                for zone_name, (y_min, y_max) in zones.items():
                    fig.add_trace(go.Scatter3d(
                        x=[-0.5, 0.5, 0.5, -0.5, -0.5],
                        y=[y_min, y_min, y_max, y_max, y_min],
                        z=[0, 0, 0, 0, 0],
                        mode='lines+markers',
                        line=dict(color="gray", width=2),
                        marker=dict(size=0.1, opacity=0.2),
                        showlegend=False
                    ))
            
                # Add length labels on the side of the pitch
                for length, y_position in length_positions.items():
                    fig.add_trace(go.Scatter3d(
                        x=[0.6],  # Adjust X position to be to the side of the pitch
                        y=[y_position],
                        z=[0],
                        mode='text',
                        text=[length],
                        textposition="middle right",
                        textfont=dict(size=10, color="black"),
                        showlegend=False
                    ))
            
                # Set mirroring factor based on handedness
                if handedness == 'LHB':
                    mirror_factor = -1
                elif handedness == 'RHB':
                    mirror_factor = 1
                else:
                    mirror_factor = 0  # Default case if handedness is neither "Left-hand bat" nor "Right-hand bat"
            
                # Separate the data into wicket and non-wicket balls
                wicket_data = data[data['bowler_wkt'] == 1]        
                # Plot wicket balls first
                for index, row in wicket_data.iterrows():
                    if pd.isna(row['line']) or pd.isna(row['length']):
                       continue  # Skip this row and move to the next one
                    # Determine base X and Y positions from line and length
                    x_base = line_positions.get(row['line'], 0) * mirror_factor
                    y_base = length_positions.get(row['length'], 5)
            
                    # Apply offset to length (y) while keeping line (x) accurate
                    x_pos = apply_line_offset(x_base, boundary=(-0.5, 0.5))
                    y_pos = apply_length_offset(y_base, boundary=(-2, 10))
                    z_pos = 0
            
                    # Set color and size for wickets
                    color = 'red'
                    size = 5
                    opacity = 1  # Set opacity to a single value
            
                    # Plot the wicket ball
                    fig.add_trace(go.Scatter3d(
                        x=[x_pos],
                        y=[y_pos],
                        z=[z_pos],
                        mode='markers',
                        marker=dict(size=size, color=color, opacity=opacity),
                        hoverinfo="text",
                        text=f"Line: {row['line']}<br>Length: {row['length']}<br>Runs: {row['batsman_runs']} - Wicket"

                    ))
                # Twinkle effect for wickets (already added in the wicket balls loop)
            
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(title='X-axis', range=[-1, 1]),
                        yaxis=dict(title='Y-axis', range=[-2, 10]),
                        zaxis=dict(title='Z-axis (Height)', range=[0, 2]),
                    ),
                    width=1200,
                    height=1000,
                    showlegend=False
                )
                return fig

            
            # Display each plot in the respective column
            with col1:
                st.markdown("**Wickets vs Left-Handed Batsmen**")
                if lhb_data.empty:
                    st.write("No data available")
                else:
                    st.plotly_chart(create_pitch_map(lhb_data, 'LHB'), key="LHB", use_container_width=True)
            
            with col2:
                st.markdown("**Wickets vs Right-Handed Batsmen**")
                if rhb_data.empty:
                    st.write("No data available")
                else:
                    st.plotly_chart(create_pitch_map(rhb_data, 'RHB'), key="RHB", use_container_width=True)

            import numpy as np
            import plotly.graph_objects as go
            import streamlit as st
            
            # Assuming final_df already exists with the following columns: 'line', 'length', 'batsman_runs', 'is_wkt'
            
            # Set up line and length mapping
            line_positions = {
                'WIDE_OUTSIDE_OFFSTUMP': 0,
                'OUTSIDE_OFFSTUMP': 1,
                'ON_THE_STUMPS':2,
                'DOWN_LEG': 3,
                'WIDE_DOWN_LEG':4 
            }


            length_positions = {
                'SHORT': 0 ,
                'SHORT_OF_A_GOOD_LENGTH': 1,
                'GOOD_LENGTH': 2,
                'FULL': 3,
                'YORKER': 4,
                'FULL_TOSS': 4
            }
            
            # Initialize 5x5 grids for wicket count and run accumulation
            wicket_count_grid = np.zeros((5, 5))
            run_count_grid_bowler = np.zeros((5, 5))
            
            # Fill the grids based on final_df data
            for _, row in final_df.iterrows():
                if pd.isna(row['line']) or pd.isna(row['length']):
                    continue  # Skip this row and move to the next one
                line = row['line']
                length = row['length']
                runs = row['batsman_runs']
                is_wkt = row['bowler_wkt']
            
                # Identify the correct cell for wicket count and run count
                line_idx = line_positions.get(line, 2)  # Default to 'On Stumps' if line not found
                length_idx = length_positions.get(length, 2)  # Default to 'Good Length' if length not found
            
                if is_wkt == 1:
                    # Update wicket count in the grid for the given line and length
                    wicket_count_grid[length_idx, line_idx] += 1
                
                # Update run count in the grid for the given line and length
                run_count_grid_bowler[length_idx, line_idx] += runs
            
            # Labels for line and length positions
            line_labels = ['Wide Outside Off', 'Outside Off', 'On Stumps', 'Outside Leg', 'Wide Outside Leg']
            length_labels = ['Short', 'Back of Length', 'Good Length', 'Full', 'Yorker']
            
            # Function to create heatmap figure for a 5x5 grid
            @st.cache_data
            def create_heatmap(grid, title, annotations):
                fig = go.Figure(
                    data=go.Heatmap(
                        z=grid,
                        colorscale='Reds',
                        colorbar=dict(title=f'{title}')
                    )
                )
                # Add black text annotations to show values
                for i in range(5):
                    for j in range(5):
                        fig.add_annotation(
                            x=j, y=i,
                            text=f'{annotations[i, j]}',
                            showarrow=False,
                            font=dict(color="black", size=12)
                        )
                
                # Update layout for vertical orientation and labels
                fig.update_layout(
                    xaxis=dict(showgrid=False, tickvals=list(range(5)), ticktext=line_labels, title="Line"),
                    yaxis=dict(showgrid=False, tickvals=list(range(5)), ticktext=length_labels, title="Length"),
                    height=700, width=300  # Adjusted size for compact display
                )
                return fig
            
            # Organize layouts in two columns to make them appear side-by-side
            col1, col2 = st.columns(2)
            
            # First Column - Wicket Heatmap
            with col1:
                st.markdown("**Wicket Distribution**")
                wicket_fig = create_heatmap(wicket_count_grid, "Wickets", wicket_count_grid)
                st.plotly_chart(wicket_fig, use_container_width=True)
            
            # Second Column - Run Distribution for Bowler
            with col2:
                st.markdown("**Runs Given**")
                run_fig_bowler = create_heatmap(run_count_grid_bowler, "Runs", run_count_grid_bowler)
                st.plotly_chart(run_fig_bowler, use_container_width=True)
        else:
            st.write("## No Bowling Data Available")
        
        # If user chooses Bowling role, provide a placeholder (or extend later)
        # else:
        #     st.header(f"Strength & Weakness — Bowling: {player_selected}")
        
        
