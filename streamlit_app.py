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
WAGON_ZONES = [
                    "Third Man",
                    "Point",
                    "Cover",
                    "Mid Off",
                    "Mid On",
                    "Mid Wicket",
                    "Square Leg",
                    "Fine Leg"
                ]
                
                # Semantic swaps for LHB
LHB_ZONE_SWAP = {
                    "Third Man": "Fine Leg",
                    "Fine Leg": "Third Man",
                
                    "Point": "Square Leg",
                    "Square Leg": "Point",
                
                    "Cover": "Mid Wicket",
                    "Mid Wicket": "Cover",
                
                    "Mid Off": "Mid On",
                    "Mid On": "Mid Off"
}
                
def draw_bowler_caught_dismissals_wagon(df_wagon, bowler_name, normalize_to_rhb=True):
    """
    Plotly wagon chart for caught dismissals FOR BOWLER.
    Clean, single-render, no duplication.
    """
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd
    import streamlit as st

    # ---------------- Defensive checks ----------------
    if not isinstance(df_wagon, pd.DataFrame) or df_wagon.empty:
        return

    if not {'dismissal', 'bowl', 'wagonX', 'wagonY'}.issubset(df_wagon.columns):
        st.warning("Required columns missing for caught dismissal wagon.")
        return

    # ---------------- Filter caught dismissals ----------------
    caught_df = df_wagon[
        (df_wagon['bowl'] == bowler_name) &
        (df_wagon['dismissal'].astype(str).str.lower().str.contains('caught', na=False))
    ].copy()

    if caught_df.empty:
        st.info(f"No caught dismissals for {bowler_name}.")
        return

    caught_df['wagonX'] = pd.to_numeric(caught_df['wagonX'], errors='coerce')
    caught_df['wagonY'] = pd.to_numeric(caught_df['wagonY'], errors='coerce')
    caught_df = caught_df.dropna(subset=['wagonX', 'wagonY'])
    if caught_df.empty:
        return

    # ---------------- Normalize coordinates ----------------
    center, radius = 184.0, 184.0
    x = (caught_df['wagonX'] - center) / radius
    y = - (caught_df['wagonY'] - center) / radius

    # Detect LHB
    is_lhb = False
    if 'bat_hand' in caught_df.columns:
        try:
            is_lhb = caught_df['bat_hand'].dropna().iloc[0].upper().startswith('L')
        except Exception:
            pass

    if (not normalize_to_rhb) and is_lhb:
        x = -x

    mask = np.sqrt(x**2 + y**2) <= 1
    x, y = x[mask], y[mask]
    caught_df = caught_df.loc[mask]

    if caught_df.empty:
        return

    # ---------------- Hover data ----------------
    hover_cols = [c for c in ['bat', 'bowl', 'line', 'length', 'shot', 'dismissal'] if c in caught_df.columns]
    customdata = caught_df[hover_cols].astype(str).values

    hovertemplate = (
        "<br>".join(
            [f"<b>{col}:</b> %{{customdata[{i}]}}" for i, col in enumerate(hover_cols)]
        ) + "<extra></extra>"
    )

    # ---------------- Build figure ----------------
    fig = go.Figure()

    def add_circle(x0, y0, x1, y1, **kw):
        if (not normalize_to_rhb) and is_lhb:
            x0, x1 = -x1, -x0
        fig.add_shape(type="circle", x0=x0, y0=y0, x1=x1, y1=y1, **kw)

    # Field
    add_circle(-1, -1, 1, 1, fillcolor="#228B22", line_color="black", layer="below")
    add_circle(-0.5, -0.5, 0.5, 0.5, fillcolor="#66bb6a", line_color="white", layer="below")

    # Pitch
    px0, px1 = (-0.04, 0.04)
    if (not normalize_to_rhb) and is_lhb:
        px0, px1 = -px1, -px0
    fig.add_shape(type="rect", x0=px0, y0=-0.08, x1=px1, y1=0.08,
                  fillcolor="tan", line_color=None)

    # Radials
    for ang in np.linspace(0, 2*np.pi, 8, endpoint=False):
        xe, ye = np.cos(ang), np.sin(ang)
        if (not normalize_to_rhb) and is_lhb:
            xe = -xe
        fig.add_trace(go.Scatter(
            x=[0, xe], y=[0, ye],
            mode="lines", line=dict(color="white", width=1),
            opacity=0.25, showlegend=False
        ))

    # Points (ONCE)
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(color="red", size=12, line=dict(color="black", width=1.5)),
        customdata=customdata,
        hovertemplate=hovertemplate,
        name="Caught Dismissals"
    ))

    fig.update_layout(
        xaxis=dict(range=[-1.2, 1.2], visible=False),
        yaxis=dict(range=[-1.2, 1.2], visible=False, scaleanchor="x"),
        height=700,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

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
    'YORKER': 4,
    'FULLTOSS': 5
}
# def unique_vals_union(col):
#     vals = []
#     for df in (pf, bdf):
#         if col in df.columns:
#             vals.extend(df[col].dropna().astype(str).str.strip().tolist())
#     vals = sorted({v for v in vals if str(v).strip() != ''})
#     return vals

# bowl_kinds_present = unique_vals_union('bowl_kind') # e.g. ['pace', 'spin']
# # Limit to pace/spin only
# bowl_kinds_present = [k for k in bowl_kinds_present if 'pace' in k.lower() or 'spin' in k.lower()]

# bowl_styles_present = unique_vals_union('bowl_style')

def display_pitchmaps_from_df(df_src, title_prefix):
    if df_src is None or df_src.empty:
        st.info(f"No deliveries to show for {title_prefix}")
        return

# ---------- robust map-lookup helpers ----------
def _norm_key(s):
    if s is None:
        return ''
    return str(s).strip().upper().replace(' ', '_').replace('-', '_')

def get_map_index(map_obj, raw_val):
    if raw_val is None:
        return None
    sval = str(raw_val).strip()
    if sval == '' or sval.lower() in ('nan', 'none'):
        return None

    if sval in map_obj:
        return int(map_obj[sval])
    s_norm = _norm_key(sval)
    for k in map_obj:
        try:
            if isinstance(k, str) and _norm_key(k) == s_norm:
                return int(map_obj[k])
        except Exception:
            continue
    for k in map_obj:
        try:
            if isinstance(k, str) and (k.lower() in sval.lower() or sval.lower() in k.lower()):
                return int(map_obj[k])
        except Exception:
            continue
    return None

# ---------- grids builder ----------
def build_pitch_grids(df_in, line_col_name='line', length_col_name='length', runs_col_candidates=('batruns', 'score'),
                      control_col='control', dismissal_col='dismissal'):
    if 'length_map' in globals() and isinstance(length_map, dict) and len(length_map) > 0:
        try:
            max_idx = max(int(v) for v in length_map.values())
            n_rows = max(5, max_idx + 1)
        except Exception:
            n_rows = 5
    else:
        n_rows = 5
        st.warning("length_map not found; defaulting to 5 rows.")

    length_vals = df_in.get(length_col_name, pd.Series()).dropna().astype(str).str.lower().unique()
    if any('full toss' in val for val in length_vals):
        n_rows = max(n_rows, 6)

    n_cols = 5

    count = np.zeros((n_rows, n_cols), dtype=int)
    bounds = np.zeros((n_rows, n_cols), dtype=int)
    dots = np.zeros((n_rows, n_cols), dtype=int)
    runs = np.zeros((n_rows, n_cols), dtype=float)
    wkt = np.zeros((n_rows, n_cols), dtype=int)
    ctrl_not = np.zeros((n_rows, n_cols), dtype=int)

    runs_col = None
    for c in runs_col_candidates:
        if c in df_in.columns:
            runs_col = c
            break
    if runs_col is None:
        runs_col = None # will use 0

    wkt_tokens = {'caught', 'bowled', 'stumped', 'lbw','leg before wicket','hit wicket'}
    # Pre-process dismissal column to handle NA safely (add this)
    dismissal_series = df_in.get(dismissal_col, pd.Series(dtype='object')).fillna('').astype(str).str.lower()
    for _, row in df_in.iterrows():
        li = get_map_index(line_map, row.get(line_col_name, None)) if 'line_map' in globals() else None
        le = get_map_index(length_map, row.get(length_col_name, None)) if 'length_map' in globals() else None
        if li is None or le is None:
            continue
        if not (0 <= le < n_rows and 0 <= li < n_cols):
            continue
        count[le, li] += 1
        rv = 0
        if runs_col:
            try:
                rv = int(row.get(runs_col, 0) or 0)
            except:
                rv = 0
        runs[le, li] += rv
        if rv >= 4:
            bounds[le, li] += 1
        if rv == 0:
            dots[le, li] += 1
        dval = dismissal_series.iloc[_]
        if any(tok in dval for tok in wkt_tokens):
            wkt[le, li] += 1
        cval = row.get(control_col, None)
        if cval is not None:
            if isinstance(cval, str) and 'not' in cval.lower():
                ctrl_not[le, li] += 1
            elif isinstance(cval, (int, float)) and float(cval) == 0:
                ctrl_not[le, li] += 1

    sr = np.full(count.shape, np.nan)
    ctrl_pct = np.full(count.shape, np.nan)
    for i in range(n_rows):
        for j in range(n_cols):
            if count[i, j] > 0:
                sr[i, j] = runs[i, j] / count[i, j] * 100.0
                ctrl_pct[i, j] = (ctrl_not[i, j] / count[i, j]) * 100.0

    return {
        'count': count, 'bounds': bounds, 'dots': dots,
        'runs': runs, 'sr': sr, 'ctrl_pct': ctrl_pct, 'wkt': wkt, 'n_rows': n_rows, 'n_cols': n_cols
    }

# --- helper: render matplotlib fig as fixed-pixel-height image in Streamlit ---
from PIL import Image
def display_pitchmaps_from_df(df_src, title_prefix):
    if df_src is None or df_src.empty:
        st.info(f"No deliveries to show for {title_prefix}")
        return

    grids = build_pitch_grids(df_src)

    bh_col_name = globals().get('bat_hand_col', 'bat_hand')
    is_lhb = False
    if bh_col_name in df_src.columns:
        hands = df_src[bh_col_name].dropna().astype(str).str.strip().unique()
        if any(h.upper().startswith('L') for h in hands):
            is_lhb = True

    def maybe_flip(arr):
        return np.fliplr(arr) if is_lhb else arr.copy()

    # count = maybe_flip(grids['count'])
    # bounds = maybe_flip(grids['bounds'])
    # dots = maybe_flip(grids['dots'])
    # sr = maybe_flip(grids['sr'])
    # ctrl = maybe_flip(grids['ctrl_pct'])
    # wkt = maybe_flip(grids['wkt'])
    # runs = maybe_flip(grids['runs'])

    # total = count.sum() if count.sum() > 0 else 1.0
    # perc = count.astype(float) / total * 100.0

    # xticks_base = ['Wide Out Off', 'Outside Off', 'On Stumps', 'Down Leg', 'Wide Down Leg']
    # xticks = xticks_base[::-1] if is_lhb else xticks_base

    # n_rows = grids['n_rows']
    # if n_rows >= 6:
    #     yticklabels = ['Short', 'Back of Length', 'Good', 'Full', 'Yorker', 'Full Toss'][:n_rows]
    # else:
    #     yticklabels = ['Short', 'Back of Length', 'Good', 'Full', 'Yorker'][:n_rows]

    # fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    # plt.suptitle(f"{player_selected} — {title_prefix}", fontsize=16, weight='bold')

    # plot_list = [
    #     (perc, '% of balls (heat)', 'Blues'),
    #     (bounds, 'Boundaries (count)', 'OrRd'),
    #     (dots, 'Dot balls (count)', 'Blues'),
    #     (sr, 'SR (runs/100 balls)', 'Reds'),
    #     (ctrl, 'False Shot % (not in control)', 'PuBu'),
    #     (runs, 'Runs (sum)', 'Reds')
    # ]

    # for ax_idx, (ax, (arr, ttl, cmap)) in enumerate(zip(axes.flat, plot_list)):
    #     safe_arr = np.nan_to_num(arr.astype(float), nan=0.0)
    #     flat = safe_arr.flatten()
    #     if np.all(flat == 0):
    #         vmin, vmax = 0, 1
    #     else:
    #         vmin = float(np.nanmin(flat))
    #         vmax = float(np.nanpercentile(flat, 95))
    #         if vmax <= vmin:
    #             vmax = vmin + 1.0

    #     im = ax.imshow(safe_arr, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    #     ax.set_title(ttl)
    #     ax.set_xticks(range(grids['n_cols'])); ax.set_yticks(range(grids['n_rows']))
    #     ax.set_xticklabels(xticks, rotation=45, ha='right')
    #     ax.set_yticklabels(yticklabels)

    #     ax.set_xticks(np.arange(-0.5, grids['n_cols'], 1), minor=True)
    #     ax.set_yticks(np.arange(-0.5, grids['n_rows'], 1), minor=True)
    #     ax.grid(which='minor', color='black', linewidth=0.6, alpha=0.95)
    #     ax.tick_params(which='minor', bottom=False, left=False)

    #     if ax_idx == 0:
    #         for i in range(grids['n_rows']):
    #             for j in range(grids['n_cols']):
    #                 w_count = int(wkt[i, j])
    #                 if w_count > 0:
    #                     w_text = f"{w_count} W" if w_count > 1 else 'W'
    #                     ax.text(j, i, w_text, ha='center', va='center', fontsize=14, color='gold', weight='bold',
    #                             bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))

    #     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # safe_fn = globals().get('safe_st_pyplot', None)
    # try:
    #     if callable(safe_fn):
    #         safe_fn(fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
    #     else:
    #         st.pyplot(fig)
    # except Exception:
    #     st.pyplot(fig)
    # finally:
    #     plt.close(fig)

# ---------- attempt to draw wagon chart using your existing function ----------
def draw_wagon_if_available(df_wagon, batter_name):
    if 'draw_cricket_field_with_run_totals_requested' in globals() and callable(globals()['draw_cricket_field_with_run_totals_requested']):
        try:
            fig_w = draw_cricket_field_with_run_totals_requested(df_wagon, batter_name)
            safe_fn = globals().get('safe_st_pyplot', None)
            if callable(safe_fn):
                safe_fn(fig_w, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
            else:
                st.pyplot(fig_w)
        except Exception as e:
            st.error(f"Wagon drawing function exists but raised: {e}")
    else:
        st.warning("Wagon chart function not found; please ensure `draw_cricket_field_with_run_totals_requested` is defined earlier.")

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
    st.image(img_resized, use_container_width=False, width=new_w)
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


st.set_page_config(page_title='IPL Performance Analysis Portal (Since IPL 2021)', layout='wide')

import streamlit as st
import streamlit.components.v1 as components

# st.markdown("""
# <style>

# /* ===== Global app background ===== */
# .stApp {
#     background-color: #0e1624;
# }

# /* ===== Sidebar ===== */
# [data-testid="stSidebar"] {
#     background: linear-gradient(180deg, #0b1220, #020617);
#     border-right: 1px solid #1e293b;
# }

# [data-testid="stSidebar"] * {
#     color: #e5e7eb !important;
# }

# /* ===== Headings ===== */
# h1, h2, h3, h4 {
#     color: #f8fafc;
# }

# /* ===== Body text ===== */
# p, span, label {
#     color: #cbd5e1;
# }

# /* ===== Inputs ===== */
# .stSelectbox div[data-baseweb="select"],
# .stNumberInput input,
# .stSlider {
#     background-color: #121c2e !important;
#     color: #e5e7eb !important;
#     border-radius: 8px;
#     border: 1px solid #1e293b;
# }

# /* ===== Dataframes ===== */
# [data-testid="stDataFrame"] {
#     background-color: #121c2e;
#     border-radius: 12px;
#     border: 1px solid #1e293b;
# }

# /* ===== Plot containers (CRITICAL FIX) ===== */
# div[data-testid="stPlotlyChart"],
# div[data-testid="stPyplot"] {
#     background-color: #121c2e;
#     border-radius: 14px;
#     padding: 14px;
#     border: 1px solid #1e293b;
#     box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
# }

# /* ===== Section dividers ===== */
# hr {
#     border: none;
#     border-top: 1px solid #1e293b;
#     margin: 24px 0;
# }

# </style>
# """, unsafe_allow_html=True)
# import matplotlib.pyplot as plt

# plt.rcParams.update({
#     "figure.facecolor": "#121c2e",
#     "axes.facecolor": "#121c2e",
#     "text.color": "#e5e7eb",
#     "axes.labelcolor": "#e5e7eb",
#     "xtick.color": "#cbd5e1",
#     "ytick.color": "#cbd5e1",
# })

# st.markdown("""
# <style>

# /* =========================
#    GLOBAL
# ========================= */
# .stApp {
#     background-color: #caf0f8;
#     color: #0f172a;
# }

# /* Remove default Streamlit padding blocks */
# .block-container {
#     padding-top: 1.5rem;
# }

# /* =========================
#    SIDEBAR
# ========================= */
# [data-testid="stSidebar"] {
#     background: linear-gradient(
#         180deg,
#         #023047 0%,
#         #022738 100%
#     );
#     border-right: 1px solid rgba(255,255,255,0.08);
# }

# /* Sidebar text – force visibility */
# [data-testid="stSidebar"] * {
#     color: #e5f4fb !important;
#     font-weight: 500;
# }

# /* Sidebar section headers */
# [data-testid="stSidebar"] h1,
# [data-testid="stSidebar"] h2,
# [data-testid="stSidebar"] h3 {
#     color: #ffffff !important;
#     font-weight: 700;
# }

# /* =========================
#    INPUT WIDGETS (MAIN + SIDEBAR)
# ========================= */
# .stSelectbox div[data-baseweb="select"],
# .stMultiSelect div[data-baseweb="select"],
# .stNumberInput input,
# .stTextInput input,
# .stSlider > div {
#     background: rgba(255, 255, 255, 0.55) !important;
#     backdrop-filter: blur(10px);
#     border-radius: 12px;
#     border: 1px solid rgba(2, 48, 71, 0.15);
#     color: #0f172a !important;
# }

# /* Dropdown text */
# .stSelectbox span {
#     color: #0f172a !important;
# }

# /* =========================
#    RADIO / CHECKBOX
# ========================= */
# .stRadio label,
# .stCheckbox label {
#     color: #e5f4fb !important;
# }

# /* =========================
#    METRIC CARDS
# ========================= */
# [data-testid="stMetric"] {
#     background: rgba(255,255,255,0.65);
#     border-radius: 14px;
#     padding: 14px;
#     border: 1px solid rgba(2, 48, 71, 0.15);
#     backdrop-filter: blur(10px);
# }

# /* =========================
#    DATAFRAMES & PLOTS
# ========================= */
# [data-testid="stDataFrame"],
# div[data-testid="stPlotlyChart"],
# div[data-testid="stPyplot"] {
#     background: rgba(255,255,255,0.7);
#     border-radius: 16px;
#     padding: 16px;
#     border: 1px solid rgba(2, 48, 71, 0.18);
#     box-shadow: 0 8px 24px rgba(2,48,71,0.12);
# }

# /* =========================
#    HEADINGS
# ========================= */
# h1, h2 {
#     color: #0f172a;
#     font-weight: 800;
# }

# h3, h4 {
#     color: #0f172a;
#     font-weight: 700;
# }

# /* =========================
#    ORANGE ACCENTS
# ========================= */
# .orange-accent {
#     color: #fb8500;
#     font-weight: 700;
# }

# /* =========================
#    DIVIDERS
# ========================= */
# hr {
#     border: none;
#     border-top: 1px solid rgba(2,48,71,0.2);
#     margin: 24px 0;
# }

# </style>
# """, unsafe_allow_html=True)


st.markdown("""
<style>

/* ===== APP BACKGROUND ===== */
.stApp {
    background-color: #f5f7fa;
    color: #0f172a;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background-color: #0b2545;
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * {
    color: #e6edf5 !important;
    font-weight: 500;
}

/* ===== BANNER ===== */
.app-banner {
    background: linear-gradient(135deg, #081c2f, #0b2545);
    border-radius: 18px;
    padding: 28px 32px;
    color: #ffffff;
}

/* ===== INPUTS ===== */
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"],
.stNumberInput input,
.stTextInput input,
.stSlider > div {
    background-color: #fbfdff !important;
    border: 1px solid #dbe2ea;
    border-radius: 10px;
    color: #0f172a !important;
}

/* ===== METRIC CARDS ===== */
[data-testid="stMetric"] {
    background-color: #fbfdff;
    border-radius: 14px;
    padding: 14px;
    border: 1px solid #dbe2ea;
}

/* ===== TABLES & PLOTS ===== */
[data-testid="stDataFrame"],
div[data-testid="stPlotlyChart"],
div[data-testid="stPyplot"] {
    background-color: #fbfdff;
    border-radius: 16px;
    padding: 16px;
    border: 1px solid #dbe2ea;
    box-shadow: 0 6px 18px rgba(15,23,42,0.08);
}

/* ===== HEADINGS ===== */
h1, h2 {
    font-weight: 800;
    color: #0f172a;
}

h3, h4 {
    font-weight: 700;
    color: #0f172a;
}

/* ===== ORANGE ACCENT ===== */
.orange-accent {
    color: #f08a24;
    font-weight: 700;
}

/* ===== DIVIDERS ===== */
hr {
    border: none;
    border-top: 1px solid #dbe2ea;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* ===== Sidebar YEAR / SLIDER LABELS ===== */
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSlider span,
[data-testid="stSidebar"] [data-baseweb="slider"] span {
    color: #f08a24 !important;
    font-weight: 600;
}

/* Slider tick values (years) */
[data-testid="stSidebar"] [data-baseweb="slider"] div {
    color: #f08a24 !important;
}

/* Slider value bubble (if visible) */
[data-testid="stSidebar"] [role="slider"] {
    color: #f08a24 !important;
}

</style>
""", unsafe_allow_html=True)


components.html(
    """
    <div style="
        width: 100%;
        background: linear-gradient(135deg, #0A2540 0%, #0f2f55 60%, #0A2540 100%);
        padding: 26px 32px;
        border-radius: 14px;
        box-shadow: 0 12px 28px rgba(0,0,0,0.22);
        font-family: Inter, Arial, sans-serif;
        box-sizing: border-box;
    ">

        <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:20px;">

            <!-- LEFT: BRAND -->
            <div>
                <div style="
                    font-size: 40px;
                    font-weight: 800;
                    color: #F8FAFC;
                    letter-spacing: 0.4px;
                    line-height: 1.05;
                ">
                    DeepCrease
                </div>

                <div style="
                    font-size: 15px;
                    font-weight: 600;
                    color: #2DD4BF;
                    margin-top: 6px;
                    letter-spacing: 0.4px;
                ">
                    Contextual Cricket Intelligence Engine
                </div>

                <div style="
                    margin-top: 10px;
                    font-size: 17px;
                    font-weight: 500;
                    color: #FACC15;
                    letter-spacing: 0.3px;
                ">
                    T20 Performance Analysis Portal <span style="opacity:0.85;">(Since 2021)</span>
                </div>
            </div>

            <!-- RIGHT: NAME + LINKS -->
            <div style="text-align:right; margin-top:6px;">
                <div style="
                    font-size:13px;
                    font-weight:600;
                    color:#E5E7EB;
                ">
                    Omkar Walunj
                </div>

                <a href="https://www.linkedin.com/in/omkar-walunj-8256a4280/"
                   target="_blank"
                   style="
                       display:block;
                       margin-top:4px;
                       text-decoration:underline;
                       font-size:12px;
                       font-weight:500;
                       color:#94A3B8;
                   ">
                    LinkedIn
                </a>

                <a href="https://substack.com/@theunseengame"
                   target="_blank"
                   style="
                       display:block;
                       margin-top:2px;
                       text-decoration:underline;
                       font-size:12px;
                       font-weight:500;
                       color:#94A3B8;
                   ">
                    Substack
                </a>
            </div>

        </div>
    </div>
    """,
    height=155,
    scrolling=False
)
st.markdown("""
<style>
/* ===== Sidebar Background ===== */
[data-testid="stSidebar"] {
    background-color: #F8FAFC;
}

/* Sidebar text (ensure readability) */
[data-testid="stSidebar"] * {
    color: #0b2545 !important;   /* deep navy text for contrast */
    font-weight: 500;
}

/* Sidebar headers */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #023047 !important;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)
def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# with st.sidebar:
#     st.markdown("### Select Years")

#     year_range = st.slider(
#         "Select year range",
#         min_value=2021,
#         max_value=2026,
#         value=(2021, 2026),
#         key="year_range_sidebar"
#     )

#     year_html = (
#         "<div style='display:flex;justify-content:center;align-items:center;"
#         "gap:10px;margin-top:8px;font-weight:700;'>"
#         f"<span style='background:#f08a24;color:#0b2545;"
#         "padding:4px 10px;border-radius:14px;font-size:13px;'>"
#         f"{year_range[0]}</span>"
#         "<span style='color:#f1f6fa;'>&rarr;</span>"
#         f"<span style='background:#f08a24;color:#0b2545;"
#         "padding:4px 10px;border-radius:14px;font-size:13px;'>"
#         f"{year_range[1]}</span>"
#         "</div>"
#     )

#     st.markdown(year_html, unsafe_allow_html=True)


import os
import glob
import re
import hashlib
from io import BytesIO
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

DATASETS_DIR = "Datasets"
CACHE_DIR = ".cache_data"
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

TOURNAMENTS = {
    "IPL": "IPL_APP_IPL",
    "CPL": "IPL_APP_CPL",
    "ILT20": "IPL_APP_ILT20",
    "LPL": "IPL_APP_LPL",
    "MLC": "IPL_APP_MLC",
    "SA20": "IPL_APP_SA20",
    "Super Smash": "IPL_APP_SUPER_SMASH",
    "T20 Blast": "IPL_APP_T20_BLAST",
    "T20I": "IPL_APP_T20I",
    "BBL": "IPL_APP_BBL",
}

# ────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────

def _hash_args(tournaments, years, usecols, file_signatures):
    """Create a deterministic cache key that includes file signatures to avoid stale cache."""
    tpart = "|".join(sorted(tournaments)) if tournaments else "none"
    key = tpart + "|" + f"{min(years)}-{max(years)}"
    if usecols:
        key += "|" + ",".join(sorted(usecols))
    # include file signatures (path, mtime, size) for cache-busting when files change
    if file_signatures:
        sig_parts = []
        for s in file_signatures:
            # s is a tuple like (tournament, path, mtime, size)
            sig_parts.append(f"{s[0]}::{s[1]}::{s[2]}::{s[3]}")
        key += "|" + "|".join(sig_parts)
    return hashlib.md5(key.encode()).hexdigest()


def _strict_file_for_tournament(token: str):
    """
    STRICT resolver:
    - Match token as a separate segment in filename (not arbitrary substring).
    - Prefer parquet > csv > excel.
    - If multiple candidates remain, pick the most recently-modified one.
    """
    if not token:
        return None

    token = token.lower()
    files = glob.glob(os.path.join(DATASETS_DIR, "*"))
    files = [f for f in files if os.path.isfile(f)]

    # token boundary regex: token must be preceded/followed by non-alnum or start/end
    pattern = re.compile(r'(^|[^a-z0-9])' + re.escape(token) + r'([^a-z0-9]|$)', flags=re.IGNORECASE)

    strict_matches = []
    for f in files:
        name = os.path.basename(f).lower()
        if pattern.search(name):
            strict_matches.append(f)

    valid = strict_matches or []
    # fallback: if no strict matches, allow substring matches (lower priority)
    if not valid:
        for f in files:
            name = os.path.basename(f).lower()
            if token in name:
                valid.append(f)

    if not valid:
        return None

    # Prefer parquet > csv > excel, and among same ext prefer most recent (mtime)
    def sort_key(fpath):
        ext = os.path.splitext(fpath)[1].lower()
        priority = 0 if ext == ".parquet" else 1 if ext == ".csv" else 2
        # Use negative mtime so newest is first
        mtime = -os.path.getmtime(fpath)
        return (priority, mtime)

    valid.sort(key=sort_key)
    return valid[0]


def _detect_year_column(df):
    # Priority: explicit year/season columns
    for c in df.columns:
        lc = c.lower()
        if lc == "year" or lc == "season" or lc.endswith("_year"):
            return c

    # next: any column that contains 'year'
    for c in df.columns:
        if "year" in c.lower():
            return c

    # next: obvious date columns
    for c in df.columns:
        if "date" in c.lower() or "match_date" in c.lower() or "start" in c.lower():
            return c

    return None


def _extract_years(series):
    # numeric
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("Int64")

    # datetime-like parsing
    dt = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    if dt.notna().any():
        return dt.dt.year.astype("Int64")

    # strict 4-digit regex (captures 19xx/20xx)
    s = series.astype(str).str.extract(r"\b((?:19|20)\d{2})\b")[0]
    if s.notna().any():
        return s.astype("Int64")

    return None


# ────────────────────────────────────────────────
# FAST LOADER
# ────────────────────────────────────────────────

@st.cache_data(ttl=24 * 3600)
def load_filtered_data_fast(selected_tournaments, selected_years, usecols=None, file_signatures=None):
    """
    Load data for the given tournaments and years.
    - If selected_tournaments is empty -> return empty DataFrame.
    - Filters by years using detected year/date columns per file.
    This function is cached keyed on (selected_tournaments, selected_years, usecols, file_signatures)
    so that changes to the actual data files break the cache.
    """
    if not selected_tournaments:
        return pd.DataFrame()

    # make inputs deterministic for cache key (file_signatures already passed by caller)
    cache_key = _hash_args(tuple(selected_tournaments), selected_years, usecols, file_signatures)
    cache_path = os.path.join(CACHE_DIR, f"merged_{cache_key}.parquet")

    # If a file-based cache exists for this exact signature, return it
    if os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            # ignore and rebuild
            pass

    frames = []

    for t in selected_tournaments:
        token = TOURNAMENTS.get(t, t).lower()
        path = _strict_file_for_tournament(token)

        if path is None:
            # caller will warn; skip this tournament
            continue

        ext = os.path.splitext(path)[1].lower()

        try:
            if ext == ".parquet":
                df = pd.read_parquet(path, columns=usecols)
            elif ext == ".csv":
                df = pd.read_csv(path, usecols=usecols, low_memory=False)
            else:
                df = pd.read_excel(path, usecols=usecols)
        except Exception:
            # skip unreadable file
            continue

        if df is None or df.empty:
            continue

        # detect year column and filter by selected years if possible
        year_col = _detect_year_column(df)
        if year_col:
            yrs = _extract_years(df[year_col])
            if yrs is not None:
                mask = yrs.isin(selected_years).fillna(False)
                df = df.loc[mask]

        # explicitly set tournament label (friendly name e.g., "IPL")
        df["tournament"] = t

        if df.empty:
            continue

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True, sort=False)

    # Ensure tournament column exists
    if "tournament" not in merged.columns:
        merged["tournament"] = np.nan

    # Try to persist to parquet cache
    try:
        merged.to_parquet(cache_path, index=False)
    except Exception:
        pass

    return merged


# ────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ────────────────────────────────────────────────
# ────────────────────────────────────────────────
st.sidebar.header("Select Years")

if "year_range" not in st.session_state:
    st.session_state.year_range = (2021, 2026)

years = st.sidebar.slider(
    "Select year range",
    min_value=2021,
    max_value=2026,
    value=st.session_state.year_range,
    step=1,
    key="year_slider_key",
    label_visibility="visible"
)


st.session_state.year_range = years
selected_years = list(range(years[0], years[1] + 1))

# 🔥 CLEAR, VISIBLE YEAR DISPLAY (THIS SOLVES IT)
st.sidebar.markdown(
    f"""
    <div style="
        margin-top:6px;
        text-align:center;
        font-weight:700;
        color:#f08a24;
        font-size:14px;
    ">
        {years[0]} &nbsp;–&nbsp; {years[1]}
    </div>
    """,
    unsafe_allow_html=True
)


st.sidebar.header("Select Tournaments")

if "selected_tournaments" not in st.session_state:
    st.session_state.selected_tournaments = []

selected_tournaments = st.sidebar.multiselect(
    "Choose tournaments to load (select one or more)",
    options=list(TOURNAMENTS.keys()),
    default=st.session_state.selected_tournaments,
    key="tournament_select_key"
)

st.session_state.selected_tournaments = selected_tournaments

# ────────────────────────────────────────────────
# LOAD DATA
# ────────────────────────────────────────────────

usecols = None

# 1) If nothing selected — show message and stop execution (prevent downstream KeyError)
if not selected_tournaments:
    st.info("Please choose tournament(s) from the sidebar to load data.")
    st.stop()   # ⛔ HARD STOP — nothing below runs

# 2) Resolve file paths & build file_signatures that will be passed into the cached loader
resolved_files = []
missing = []
for t in selected_tournaments:
    token = TOURNAMENTS.get(t, t).lower()
    path = _strict_file_for_tournament(token)
    if path is None:
        missing.append(t)
        resolved_files.append((t, None, None, None))
    else:
        try:
            mtime = os.path.getmtime(path)
            size = os.path.getsize(path)
        except Exception:
            mtime, size = None, None
        resolved_files.append((t, path, mtime, size))

# if any missing, warn user — but allow loading other tournaments
if missing:
    st.warning(f"No dataset found for: {', '.join(missing)}. Check files in {os.path.abspath(DATASETS_DIR)}.")

# if all selected tournaments are missing, stop — nothing to load
if all(r[1] is None for r in resolved_files):
    st.error("No data files were found for any selected tournaments. Please add dataset files or change selection.")
    st.stop()

# Show debug info of resolved files (helpful to confirm which file each tournament maps to)
# with st.expander("Debug: Files resolved for each selected tournament (remove in production)"):
#     for t, p, m, s in resolved_files:
#         st.write({
#             "tournament": t,
#             "path": p,
#             "mtime": datetime.fromtimestamp(m).isoformat() if m else None,
#             "size_bytes": s
#         })

file_signatures = tuple(resolved_files)

# 3) Load data (the loader is cached and keyed on file_signatures so changes to files break cache)
with st.spinner("Loading data…"):
    df = load_filtered_data_fast(selected_tournaments, selected_years, usecols, file_signatures=file_signatures)

# 4) Defensive re-filter by tournament (ensures downstream code only sees selected tournaments)
if df is None or df.empty:
    st.warning("No data loaded for the chosen tournament(s) / year range.")
    st.stop()

# ensure tournament column exists and filter
if "tournament" not in df.columns:
    st.error("Loaded data does not contain a 'tournament' column. Loader expected to add it. Aborting.")
    st.stop()

# Re-filter to selected tournaments (defensive)
df = df[df["tournament"].isin(selected_tournaments)].copy()

if df.empty:
    st.warning("After filtering by selected tournaments and years, no rows remain.")
    st.stop()

# Inform user of successful load and summary
# st.success(
#     f"Loaded {len(df):,} rows from "
#     f"{len(df['tournament'].unique()):,} tournament(s), "
#     f"{len(selected_years)} year(s)."
# )

DF_gen = df

# Debugging: show unique tournaments & a small sample per tournament
# with st.expander("Debug: data overview (per-tournament samples)"):
#     st.write("Tournaments present in DF_gen:", DF_gen["tournament"].unique().tolist())
#     for t in sorted(DF_gen["tournament"].unique()):
#         st.write(f"--- Sample rows for tournament: {t} ---")
#         sample_cols = DF_gen.columns.tolist()[:12]  # limit column display
#         st.dataframe(DF_gen[DF_gen["tournament"] == t][sample_cols].head(5))

# # Example: show unique batters (what you did)
# if "bat" in DF_gen.columns:
#     st.write("Unique batters in loaded DF_gen (first 50):", list(DF_gen["bat"].unique()[:50]))
# else:
#     # try common column names that represent batter/player
#     player_cols = [c for c in DF_gen.columns if c.lower() in ("player","batter","batsman","bat")]
#     if player_cols:
#         col = player_cols[0]
#         st.write(f"Unique players detected in column '{col}' (first 50):", list(DF_gen[col].unique()[:50]))
#     else:
#         st.write("No obvious batter/player column found in DF_gen. Columns: ", DF_gen.columns.tolist())

# import os
# import glob
# import re
# import hashlib
# from io import BytesIO
# import pandas as pd
# import numpy as np
# import streamlit as st
# from datetime import datetime

# # ────────────────────────────────────────────────
# # CONFIG
# # ────────────────────────────────────────────────

# DATASETS_DIR = "Datasets"
# CACHE_DIR = ".cache_data"
# os.makedirs(DATASETS_DIR, exist_ok=True)
# os.makedirs(CACHE_DIR, exist_ok=True)

# TOURNAMENTS = {
#     "IPL": "IPL_APP_IPL",
#     "CPL": "IPL_APP_CPL",
#     "ILT20": "IPL_APP_ILT20",
#     "LPL": "IPL_APP_LPL",
#     "MLC": "IPL_APP_MLC",
#     "SA20": "IPL_APP_SA20",
#     "Super Smash": "IPL_APP_SUPER_SMASH",
#     "T20 Blast": "IPL_APP_T20_BLAST",
#     "T20I": "IPL_APP_T20I",
#     "BBL": "IPL_APP_BBL",
# }
# # TOURNAMENTS = {
# #     "IPL": "IPL_APP_IPL",
# #     "CPL": "IPL_APP_CPL",
# #     "ILT20": "ilt20",
# #     "LPL": "lpl",
# #     "MLC": "mlc",
# #     "SA20": "sa20",
# #     "Super Smash": "super_smash",
# #     "T20 Blast": "t20_blast",
# #     "T20I": "t20i",
# #     "BBL": "bbl",
# # }
# # ────────────────────────────────────────────────
# # HELPERS
# # ────────────────────────────────────────────────

# def _hash_args(tournaments, years, usecols):
#     # create a deterministic cache key
#     tpart = "|".join(sorted(tournaments)) if tournaments else "none"
#     key = tpart + "|" + f"{min(years)}-{max(years)}"
#     if usecols:
#         key += "|" + ",".join(sorted(usecols))
#     return hashlib.md5(key.encode()).hexdigest()


# def _strict_file_for_tournament(token: str):
#     """
#     STRICT resolver:
#     - Match token as a separate segment in filename (not arbitrary substring).
#     - Prefer parquet > csv > excel.
#     - If multiple candidates remain, pick the most recently-modified one.
#     """
#     token = token.lower()
#     files = glob.glob(os.path.join(DATASETS_DIR, "*"))
#     files = [f for f in files if os.path.isfile(f)]

#     # token boundary regex: token must be preceded/followed by non-alnum or start/end
#     pattern = re.compile(r'(^|[^a-z0-9])' + re.escape(token) + r'([^a-z0-9]|$)', flags=re.IGNORECASE)

#     valid = []
#     for f in files:
#         name = os.path.basename(f).lower()
#         if pattern.search(name):
#             valid.append(f)

#     # Fallback: if no strict matches, also allow substring matches (but keep them as fallback)
#     if not valid:
#         for f in files:
#             name = os.path.basename(f).lower()
#             if token in name:
#                 valid.append(f)

#     if not valid:
#         return None

#     # Prefer parquet > csv > excel, and among same ext prefer most recent
#     def sort_key(fpath):
#         ext = os.path.splitext(fpath)[1].lower()
#         priority = 0 if ext == ".parquet" else 1 if ext == ".csv" else 2
#         mtime = -os.path.getmtime(fpath)  # negative so more recent sorts first
#         return (priority, mtime)

#     valid.sort(key=sort_key)
#     return valid[0]


# def _detect_year_column(df):
#     # Priority: explicit year/season columns
#     for c in df.columns:
#         lc = c.lower()
#         if lc == "year" or lc == "season" or lc.endswith("_year"):
#             return c

#     # next: any column that contains 'year'
#     for c in df.columns:
#         if "year" in c.lower():
#             return c

#     # next: obvious date columns
#     for c in df.columns:
#         if "date" in c.lower() or "match_date" in c.lower() or "start" in c.lower():
#             return c

#     return None


# def _extract_years(series):
#     # numeric
#     if pd.api.types.is_numeric_dtype(series):
#         return series.astype("Int64")

#     # datetime-like parsing
#     dt = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
#     if dt.notna().any():
#         return dt.dt.year.astype("Int64")

#     # strict 4-digit regex (captures 19xx/20xx)
#     s = series.astype(str).str.extract(r"\b((?:19|20)\d{2})\b")[0]
#     if s.notna().any():
#         return s.astype("Int64")

#     return None


# # ────────────────────────────────────────────────
# # FAST LOADER
# # ────────────────────────────────────────────────

# @st.cache_data(ttl=24 * 3600)
# def load_filtered_data_fast(selected_tournaments, selected_years, usecols=None):
#     """
#     Load data for the given tournaments and years.
#     - If selected_tournaments is empty -> return empty DataFrame.
#     - Filters by years using detected year/date columns per file.
#     """
#     if not selected_tournaments:
#         return pd.DataFrame()

#     # make inputs deterministic for cache key
#     cache_key = _hash_args(tuple(selected_tournaments), selected_years, usecols)
#     cache_path = os.path.join(CACHE_DIR, f"merged_{cache_key}.parquet")

#     if os.path.exists(cache_path):
#         try:
#             return pd.read_parquet(cache_path)
#         except Exception:
#             # ignore and continue to rebuild
#             pass

#     frames = []

#     for t in selected_tournaments:
#         token = TOURNAMENTS.get(t, t).lower()
#         path = _strict_file_for_tournament(token)

#         if path is None:
#             # don't st.warn inside cached function (streamlit warnings inside cached functions can be flaky)
#             # instead return a sentinel by continuing and let caller show warnings
#             continue

#         ext = os.path.splitext(path)[1].lower()

#         try:
#             if ext == ".parquet":
#                 df = pd.read_parquet(path, columns=usecols)
#             elif ext == ".csv":
#                 df = pd.read_csv(path, usecols=usecols, low_memory=False)
#             else:
#                 # try excel; read_excel can fail if file is not excel — catch below
#                 df = pd.read_excel(path, usecols=usecols)
#         except Exception:
#             # skip bad file
#             continue

#         if df is None or df.empty:
#             continue

#         # Detect and filter by year (if possible)
#         year_col = _detect_year_column(df)
#         if year_col:
#             yrs = _extract_years(df[year_col])
#             if yrs is not None:
#                 # only keep rows whose extracted year is in selected_years
#                 mask = yrs.isin(selected_years).fillna(False)
#                 df = df.loc[mask]

#         # Add tournament column (explicit)
#         df["tournament"] = t

#         if df.empty:
#             continue

#         frames.append(df)

#     if not frames:
#         return pd.DataFrame()

#     merged = pd.concat(frames, ignore_index=True, sort=False)

#     # As an extra safety: if the merged frame lacks a 'tournament' column (shouldn't happen), set it
#     if "tournament" not in merged.columns:
#         merged["tournament"] = np.nan

#     # Cache to parquet if possible
#     try:
#         merged.to_parquet(cache_path, index=False)
#     except Exception:
#         pass

#     return merged


# # ────────────────────────────────────────────────
# # SIDEBAR CONTROLS
# # ────────────────────────────────────────────────

# st.sidebar.header("Select Years")

# # keep year range in session state but default without changing tournaments
# if "year_range" not in st.session_state:
#     st.session_state.year_range = (2021, 2026)

# years = st.sidebar.slider(
#     "Select year range",
#     2021, 2026,
#     st.session_state.year_range,
#     step=1,
#     key="year_slider_key"
# )

# st.session_state.year_range = years
# selected_years = list(range(years[0], years[1] + 1))

# st.sidebar.header("Select Tournaments")

# # NO default tournament selected — user has to choose explicitly
# if "selected_tournaments" not in st.session_state:
#     st.session_state.selected_tournaments = []

# selected_tournaments = st.sidebar.multiselect(
#     "Choose tournaments to load (select one or more)",
#     options=list(TOURNAMENTS.keys()),
#     default=st.session_state.selected_tournaments,
#     key="tournament_select_key"
# )

# st.session_state.selected_tournaments = selected_tournaments

# # ────────────────────────────────────────────────
# # LOAD DATA
# # ────────────────────────────────────────────────

# usecols = None

# # If user hasn't selected any tournaments, prompt them and don't try to load any data.
# if not selected_tournaments:
#     st.info("Please choose tournament(s) from the sidebar to load data.")
#     st.stop()   # ⛔ HARD STOP — nothing below runs

# else:
#     # show explicit warnings for missing datasets outside cached function
#     missing = []
#     for t in selected_tournaments:
#         token = TOURNAMENTS.get(t, t).lower()
#         if _strict_file_for_tournament(token) is None:
#             missing.append(t)
#     if missing:
#         st.warning(f"No dataset found for: {', '.join(missing)}. Check files in {DATASETS_DIR}.")

#     with st.spinner("Loading data…"):
#         df = load_filtered_data_fast(selected_tournaments, selected_years, usecols)
#         df = df[df["tournament"].isin(selected_tournaments)].copy()

#     if df.empty:
#         st.warning("No data loaded for the chosen tournament(s) / year range.")
#     else:
#         st.success(
#             f"Loaded {len(df):,} rows from "
#             f"{len(df['tournament'].unique()):,} tournament(s), "
#             f"{len(selected_years)} year(s)."
#         )

# DF_gen = df
# st.write(DF_gen.bat.unique())


# import os
# import glob
# import hashlib
# from io import BytesIO
# import requests
# import pandas as pd
# import numpy as np
# import streamlit as st
# from datetime import datetime

# # directory where dataset files live
# DATASETS_DIR = "Datasets"
# os.makedirs(DATASETS_DIR, exist_ok=True)

# # small user-friendly mapping you can keep (we'll attempt to resolve to real files)
# TOURNAMENTS = {
#     "IPL": "ipl",  # will match filenames containing 'ipl'
#     "CPL": "cpl",
#     "ILT20": "ilt20",
#     "LPL": "lpl",
#     "MLC": "mlc",
#     "SA20": "sa20",
#     "Super Smash": "super_smash",
#     "T20 Blast": "t20_blast",
#     "T20I": "t20i",
#     "BBL": "bbl",  # Added BBL
# }

# CACHE_DIR = "./.cache_data"
# os.makedirs(CACHE_DIR, exist_ok=True)


# def _hash_args(selected_tournaments, selected_years, usecols):
#     key = "|".join(sorted(selected_tournaments)) + "|" + f"{min(selected_years)}-{max(selected_years)}"
#     if usecols:
#         key += "|" + ",".join(sorted(usecols))
#     return hashlib.md5(key.encode()).hexdigest()


# def _find_best_file_for_tournament(tournament_key):
#     lowered = tournament_key.lower()
#     candidates = glob.glob(os.path.join(DATASETS_DIR, "*"))
#     candidates = [p for p in candidates if os.path.isfile(p)]
#     scored = []
#     for p in candidates:
#         name = os.path.basename(p).lower()
#         score = 0
#         if lowered in name:
#             score += 10
#         tokens = [t for t in "".join(ch if ch.isalnum() else " " for ch in name).split() if t]
#         if lowered in tokens:
#             score += 5
#         ext = os.path.splitext(p)[1].lower()
#         if ext == ".parquet":
#             score += 4
#         elif ext == ".csv":
#             score += 2
#         elif ext in (".xlsx", ".xls"):
#             score += 1
#         try:
#             mtime = os.path.getmtime(p)
#             age_score = int((datetime.now().timestamp() - mtime) // (24 * 3600))
#             score += max(0, 2 - min(age_score, 2))
#         except Exception:
#             pass
#         if score > 0:
#             scored.append((score, p))
#     if not scored:
#         return None
#     scored.sort(reverse=True, key=lambda x: (x[0], os.path.getmtime(x[1])))
#     return scored[0][1]


# def _detect_year_column(df):
#     if df is None or df.shape[1] == 0:
#         return None
#     col_candidates = [c for c in df.columns if 'year' in c.lower() or 'season' in c.lower()]
#     if not col_candidates:
#         return None
#     for pref in ['year', 'season']:
#         for c in col_candidates:
#             if pref == c.lower():
#                 return c
#     return col_candidates[0]


# def _extract_years_series(series):
#     try:
#         s = pd.to_numeric(series.astype(str).str[:4], errors='coerce').fillna(np.nan)
#         if s.notna().sum() > 0:
#             return s.astype('Int64')
#     except Exception:
#         pass
#     try:
#         sdt = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
#         if sdt.notna().sum() > 0:
#             return sdt.dt.year.astype('Int64')
#     except Exception:
#         pass
#     return None


# @st.cache_data(ttl=24 * 3600, show_spinner="Loading selected data (may take a moment for large leagues)…")
# def load_filtered_data_fast(selected_tournaments, selected_years, usecols=None, csv_chunksize=250_000):
#     if not selected_tournaments:
#         return pd.DataFrame()
#     cache_hash = _hash_args(selected_tournaments, selected_years, usecols=usecols)
#     cache_path = os.path.join(CACHE_DIR, f"merged_{cache_hash}.parquet")
#     if os.path.exists(cache_path):
#         try:
#             df_cached = pd.read_parquet(cache_path)
#             return df_cached
#         except Exception:
#             pass
#     parts = []
#     first_columns = None
#     for t in selected_tournaments:
#         mapped = TOURNAMENTS.get(t, None)
#         source_candidate = None
#         if mapped:
#             maybe = os.path.join(DATASETS_DIR, mapped) if not os.path.isabs(mapped) else mapped
#             if os.path.exists(maybe):
#                 source_candidate = maybe
#             else:
#                 found = _find_best_file_for_tournament(mapped)
#                 if found:
#                     source_candidate = found
#         if source_candidate is None:
#             source_candidate = _find_best_file_for_tournament(t)
#         if source_candidate is None:
#             st.warning(f"No dataset file found for tournament '{t}' in '{DATASETS_DIR}'. Skipping.")
#             continue
#         ext = os.path.splitext(source_candidate)[1].lower()
#         try:
#             if ext == ".parquet":
#                 df_temp = pd.read_parquet(source_candidate, columns=usecols)
#                 year_col = _detect_year_column(df_temp)
#                 if year_col:
#                     yrs = _extract_years_series(df_temp[year_col])
#                     if yrs is not None:
#                         df_temp = df_temp[yrs.isin(selected_years).fillna(False).values]
#                 if df_temp is None or df_temp.shape[0] == 0:
#                     continue
#             elif ext == ".csv":
#                 collected = []
#                 reader = pd.read_csv(source_candidate, usecols=usecols, chunksize=csv_chunksize, low_memory=True)
#                 for chunk in reader:
#                     year_col = _detect_year_column(chunk)
#                     if year_col:
#                         yrs = _extract_years_series(chunk[year_col])
#                         if yrs is not None:
#                             chunk = chunk[yrs.isin(selected_years).fillna(False).values]
#                     if not chunk.empty:
#                         collected.append(chunk)
#                 df_temp = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame(columns=usecols or [])
#                 if df_temp.shape[0] == 0:
#                     continue
#             elif ext in (".xlsx", ".xls"):
#                 df_temp = pd.read_excel(source_candidate, usecols=usecols)
#                 year_col = _detect_year_column(df_temp)
#                 if year_col:
#                     yrs = _extract_years_series(df_temp[year_col])
#                     if yrs is not None:
#                         df_temp = df_temp[yrs.isin(selected_years).fillna(False).values]
#                 try:
#                     parquet_equiv = os.path.splitext(source_candidate)[0] + ".parquet"
#                     df_temp.to_parquet(parquet_equiv, index=False)
#                 except Exception:
#                     pass
#                 if df_temp.shape[0] == 0:
#                     continue
#             else:
#                 try:
#                     df_temp = pd.read_parquet(source_candidate, columns=usecols)
#                 except Exception:
#                     collected = []
#                     reader = pd.read_csv(source_candidate, usecols=usecols, chunksize=csv_chunksize, low_memory=True)
#                     for chunk in reader:
#                         year_col = _detect_year_column(chunk)
#                         if year_col:
#                             yrs = _extract_years_series(chunk[year_col])
#                             if yrs is not None:
#                                 chunk = chunk[yrs.isin(selected_years).fillna(False).values]
#                         if not chunk.empty:
#                             collected.append(chunk)
#                     df_temp = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame(columns=usecols or [])
#                     if df_temp.shape[0] == 0:
#                         continue
#             df_temp['tournament'] = t
#             if first_columns is None:
#                 first_columns = df_temp.columns.tolist()
#             else:
#                 common = [c for c in first_columns if c in df_temp.columns]
#                 others = [c for c in df_temp.columns if c not in common]
#                 df_temp = df_temp[common + others]
#             parts.append(df_temp)
#         except Exception as exc:
#             st.warning(f"Failed to read {source_candidate} for '{t}': {str(exc)[:200]}")
#             continue
#     if not parts:
#         return pd.DataFrame()
#     merged = pd.concat(parts, ignore_index=True, sort=False)
#     year_col = _detect_year_column(merged)
#     if year_col:
#         yrs = _extract_years_series(merged[year_col])
#         if yrs is not None:
#             merged = merged[yrs.isin(selected_years).fillna(False).values]
#     try:
#         merged.to_parquet(cache_path, index=False)
#     except Exception:
#         pass
#     return merged


# # --------------------
# # USAGE in app
# # --------------------
# # ────────────────────────────────────────────────
# # Sidebar Controls with Full Persistence
# # ────────────────────────────────────────────────

# st.sidebar.header("Select Years")

# # Initialize year slider state if not present
# if "year_range" not in st.session_state:
#     st.session_state.year_range = (2021, 2026)

# years = st.sidebar.slider(
#     "Select year range",
#     min_value=2021,
#     max_value=2026,
#     value=st.session_state.year_range,
#     step=1,
#     key="year_slider_key"  # Unique key binds widget to state
# )

# # Always sync slider change back to session state
# st.session_state.year_range = years
# selected_years = list(range(years[0], years[1] + 1))
# st.sidebar.write(f"Selected years: {', '.join(map(str, selected_years))}")

# st.sidebar.header("Select Tournaments")
# all_tournaments = list(TOURNAMENTS.keys())

# # Initialize tournament selection state if not present
# if "selected_tournaments" not in st.session_state:
#     st.session_state.selected_tournaments = ["IPL"]

# selected_tournaments = st.sidebar.multiselect(
#     "Choose tournaments to load",
#     options=all_tournaments,
#     default=st.session_state.selected_tournaments,
#     key="tournament_select_key"  # Critical key for persistence
# )

# # Force sync selected value back to session state (handles edge cases)
# st.session_state.selected_tournaments = selected_tournaments

# # Debug (optional - remove later)
# # st.sidebar.write("Current session tournaments:", st.session_state.selected_tournaments)

# # ────────────────────────────────────────────────
# # Load data only when needed
# # ────────────────────────────────────────────────
# usecols = None  # change to specific columns if you want speedup

# if selected_tournaments:  # only load if at least one tournament selected
#     with st.spinner("Loading data (fast path) — this should be quick if parquet exists..."):
#         df = load_filtered_data_fast(selected_tournaments, selected_years, usecols=usecols)
    
#     if df is None or df.empty:
#         st.warning("No data loaded. Check selected tournaments, the Datasets folder, or the year range.")
#     else:
#         st.success(f"Loaded {len(df):,} rows from {len(selected_tournaments)} tournament(s) and {len(selected_years)} year(s).")
#         st.sidebar.success("Data loaded successfully!")
# else:
#     st.info("Please select at least one tournament to load data.")
#     df = pd.DataFrame()

# DF_gen = df
# ===========================
# Fast loader (parquet-first)
# Paste this block in place of your old loader code
# ===========================
# =========================
# FAST data loader (paste to replace old load block)
# =========================
# import os
# import glob
# import hashlib
# from io import BytesIO
# import requests
# import pandas as pd
# import numpy as np
# import streamlit as st
# from datetime import datetime
# # directory where dataset files live
# DATASETS_DIR = "Datasets"
# os.makedirs(DATASETS_DIR, exist_ok=True)
# # small user-friendly mapping you can keep (we'll attempt to resolve to real files)
# # keep keys that you use in UI (you can expand). If you already had TOURNAMENTS,
# # keep the same keys when calling the loader.
# TOURNAMENTS = {
#     "IPL": "ipl", # will match filenames containing 'ipl'
#     "CPL": "cpl",
#     "ILT20": "ilt20",
#     "LPL": "lpl",
#     "MLC": "mlc",
#     "SA20": "sa20",
#     "Super Smash": "super_smash",
#     "T20 Blast": "t20_blast",
#     "T20I": "t20i",
#     "BBL": "bbl",  # Added BBL - assuming filenames containing 'bbl'
# }
# CACHE_DIR = "./.cache_data"
# os.makedirs(CACHE_DIR, exist_ok=True)
# def _hash_args(selected_tournaments, selected_years, usecols):
#     key = "|".join(sorted(selected_tournaments)) + "|" + f"{min(selected_years)}-{max(selected_years)}"
#     if usecols:
#         key += "|" + ",".join(sorted(usecols))
#     return hashlib.md5(key.encode()).hexdigest()
# def _find_best_file_for_tournament(tournament_key):
#     """
#     Try to resolve a tournament key to an actual file path under DATASETS_DIR.
#     Preference order: parquet -> csv -> xlsx -> others. Match by substring (case-insensitive).
#     """
#     lowered = tournament_key.lower()
#     candidates = glob.glob(os.path.join(DATASETS_DIR, "*"))
#     # keep only files
#     candidates = [p for p in candidates if os.path.isfile(p)]
#     # rank candidates: contains lowered token in basename? give score, prefer parquet
#     scored = []
#     for p in candidates:
#         name = os.path.basename(p).lower()
#         score = 0
#         if lowered in name:
#             score += 10
#         # prefer exact tokens split by non-alnum
#         tokens = [t for t in "".join(ch if ch.isalnum() else " " for ch in name).split() if t]
#         if lowered in tokens:
#             score += 5
#         ext = os.path.splitext(p)[1].lower()
#         if ext == ".parquet":
#             score += 4
#         elif ext == ".csv":
#             score += 2
#         elif ext in (".xlsx", ".xls"):
#             score += 1
#         # prefer recently updated files slightly
#         try:
#             mtime = os.path.getmtime(p)
#             age_score = int((datetime.now().timestamp() - mtime) // (24*3600)) # days old
#             score += max(0, 2 - min(age_score, 2)) # prefer newer files
#         except Exception:
#             pass
#         if score > 0:
#             scored.append((score, p))
#     if not scored:
#         return None
#     scored.sort(reverse=True, key=lambda x: (x[0], os.path.getmtime(x[1])))
#     return scored[0][1]
# def _detect_year_column(df):
#     """Return column name that likely holds year info, or None."""
#     if df is None or df.shape[1] == 0:
#         return None
#     col_candidates = [c for c in df.columns if 'year' in c.lower() or 'season' in c.lower()]
#     if not col_candidates:
#         return None
#     # prefer exact 'year' or 'season'
#     for pref in ['year','season']:
#         for c in col_candidates:
#             if pref == c.lower():
#                 return c
#     return col_candidates[0]
# def _extract_years_series(series):
#     """Try to coerce a pandas Series to integer year values; returns integer Series or None."""
#     try:
#         # if already numeric-ish
#         s = pd.to_numeric(series.astype(str).str[:4], errors='coerce').fillna(np.nan)
#         if s.notna().sum() > 0:
#             return s.astype('Int64')
#     except Exception:
#         pass
#     # try parse datetimes
#     try:
#         sdt = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
#         if sdt.notna().sum() > 0:
#             return sdt.dt.year.astype('Int64')
#     except Exception:
#         pass
#     return None
# @st.cache_data(ttl=24*3600, show_spinner="Loading selected data (may take a moment for large leagues)…")
# def load_filtered_data_fast(selected_tournaments, selected_years, usecols=None, csv_chunksize=250_000):
#     """
#     Fast loader that:
#       - resolves tournament keys to files under DATASETS_DIR
#       - reads parquet quickly when available
#       - reads csv in chunks and filters by year on the fly
#       - caches merged result to a parquet file for instant reuse next time
#     Args:
#       selected_tournaments: list of strings (keys from TOURNAMENTS or similar)
#       selected_years: list[int]
#       usecols: optional list of needed columns to speed reads (recommended)
#       csv_chunksize: chunk size for csv streaming
#     Returns:
#       pandas.DataFrame
#     """
#     if not selected_tournaments:
#         return pd.DataFrame()
#     cache_hash = _hash_args(selected_tournaments, selected_years, usecols=usecols)
#     cache_path = os.path.join(CACHE_DIR, f"merged_{cache_hash}.parquet")
#     # return cached if exists
#     if os.path.exists(cache_path):
#         try:
#             df_cached = pd.read_parquet(cache_path)
#             return df_cached
#         except Exception:
#             # try to rebuild cache if reading fails
#             pass
#     parts = []
#     first_columns = None
#     for t in selected_tournaments:
#         # Try direct mapping first (if user provided exact path in TOURNAMENTS earlier)
#         mapped = TOURNAMENTS.get(t, None)
#         source_candidate = None
#         if mapped:
#             # if mapped looks like a full path and exists, use it
#             maybe = os.path.join(DATASETS_DIR, mapped) if not os.path.isabs(mapped) else mapped
#             if os.path.exists(maybe):
#                 source_candidate = maybe
#             else:
#                 # try to find by substring
#                 found = _find_best_file_for_tournament(mapped)
#                 if found:
#                     source_candidate = found
#         # if not found by mapping, try fuzzy search by tournament name itself
#         if source_candidate is None:
#             source_candidate = _find_best_file_for_tournament(t)
#         if source_candidate is None:
#             # warn but continue; maybe user selected a tournament that isn't present locally
#             st.warning(f"No dataset file found for tournament '{t}' in '{DATASETS_DIR}'. Skipping.")
#             continue
#         ext = os.path.splitext(source_candidate)[1].lower()
#         try:
#             if ext == ".parquet":
#                 # very fast read; filter by year if possible
#                 df_temp = pd.read_parquet(source_candidate, columns=usecols)
#                 year_col = _detect_year_column(df_temp)
#                 if year_col:
#                     yrs = _extract_years_series(df_temp[year_col])
#                     if yrs is not None:
#                         df_temp = df_temp[yrs.isin(selected_years).fillna(False).values]
#                 if df_temp is None or df_temp.shape[0] == 0:
#                     continue
#             elif ext == ".csv":
#                 # chunked read with on-the-fly year filtering
#                 collected = []
#                 reader = pd.read_csv(source_candidate, usecols=usecols, chunksize=csv_chunksize, low_memory=True)
#                 for chunk in reader:
#                     year_col = _detect_year_column(chunk)
#                     if year_col:
#                         yrs = _extract_years_series(chunk[year_col])
#                         if yrs is not None:
#                             chunk = chunk[yrs.isin(selected_years).fillna(False).values]
#                     if not chunk.empty:
#                         collected.append(chunk)
#                 df_temp = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame(columns=usecols or [])
#                 if df_temp.shape[0] == 0:
#                     continue
#             elif ext in (".xlsx", ".xls"):
#                 # read excel (slow) but convert to parquet for future runs
#                 df_temp = pd.read_excel(source_candidate, usecols=usecols)
#                 year_col = _detect_year_column(df_temp)
#                 if year_col:
#                     yrs = _extract_years_series(df_temp[year_col])
#                     if yrs is not None:
#                         df_temp = df_temp[yrs.isin(selected_years).fillna(False).values]
#                 # attempt to cache a parquet copy next to file
#                 try:
#                     parquet_equiv = os.path.splitext(source_candidate)[0] + ".parquet"
#                     df_temp.to_parquet(parquet_equiv, index=False)
#                 except Exception:
#                     pass
#                 if df_temp.shape[0] == 0:
#                     continue
#             else:
#                 # unknown extension: attempt read_parquet first, else csv
#                 try:
#                     df_temp = pd.read_parquet(source_candidate, columns=usecols)
#                 except Exception:
#                     # fallback to csv
#                     reader = pd.read_csv(source_candidate, usecols=usecols, chunksize=csv_chunksize, low_memory=True)
#                     collected = []
#                     for chunk in reader:
#                         year_col = _detect_year_column(chunk)
#                         if year_col:
#                             yrs = _extract_years_series(chunk[year_col])
#                             if yrs is not None:
#                                 chunk = chunk[yrs.isin(selected_years).fillna(False).values]
#                         if not chunk.empty:
#                             collected.append(chunk)
#                     df_temp = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame(columns=usecols or [])
#                     if df_temp.shape[0] == 0:
#                         continue
#             # add tournament tag
#             df_temp['tournament'] = t
#             # align columns to first seen frame for consistent concat (keeps memory lower)
#             if first_columns is None:
#                 first_columns = df_temp.columns.tolist()
#             else:
#                 # keep common columns first, then the rest
#                 common = [c for c in first_columns if c in df_temp.columns]
#                 others = [c for c in df_temp.columns if c not in common]
#                 df_temp = df_temp[common + others]
#             parts.append(df_temp)
#         except Exception as exc:
#             st.warning(f"Failed to read {source_candidate} for '{t}': {str(exc)[:200]}")
#             continue
#     if not parts:
#         return pd.DataFrame()
#     merged = pd.concat(parts, ignore_index=True, sort=False)
#     # final year filter as safeguard (in case some files lacked year-column detection earlier)
#     year_col = _detect_year_column(merged)
#     if year_col:
#         yrs = _extract_years_series(merged[year_col])
#         if yrs is not None:
#             merged = merged[yrs.isin(selected_years).fillna(False).values]
#     # save cache for this exact selection (fast next time)
#     try:
#         merged.to_parquet(cache_path, index=False)
#     except Exception:
#         pass
#     return merged
# # --------------------
# # USAGE in app (replace your old load code with this)
# # --------------------
# # Sidebar controls (you already have these; keep them)
# st.sidebar.header("Select Years")
# years = st.sidebar.slider("Select year range", min_value=2021, max_value=2026, value=(2021, 2026), step=1)
# selected_years = list(range(years[0], years[1] + 1))
# st.sidebar.write(f"Selected years: {', '.join(map(str, selected_years))}")
# st.sidebar.header("Select Tournaments")
# all_tournaments = list(TOURNAMENTS.keys())
# selected_tournaments = st.sidebar.multiselect("Choose tournaments to load", options=all_tournaments, default=["IPL"])
# # call loader (optional: set usecols to speed up drastically if you only need a handful of columns)
# # e.g. usecols = ['p_match','inns','bat','bowler','batruns','length','line','year','bat_hand','wagonZone','score','dismissal']
# usecols = None # <-- set to a short list if you only need specific columns
# with st.spinner("Loading data (fast path) — this should be quick if parquet exists..."):
#     df = load_filtered_data_fast(selected_tournaments, selected_years, usecols=usecols)
# if df is None or df.empty:
#     st.warning("No data loaded. Check selected tournaments, the Datasets folder, or the year range.")
# else:
#     st.success(f"Loaded {len(df):,} rows from {len(selected_tournaments)} tournament(s) and {len(selected_years)} year(s).")
#     st.sidebar.write("Data loaded successfully!")
# DF_gen = df

# import os
# import glob
# import hashlib
# from io import BytesIO
# import requests
# import pandas as pd
# import numpy as np
# import streamlit as st
# from datetime import datetime

# # directory where dataset files live
# DATASETS_DIR = "Datasets"
# os.makedirs(DATASETS_DIR, exist_ok=True)

# # small user-friendly mapping you can keep (we'll attempt to resolve to real files)
# TOURNAMENTS = {
#     "IPL": "ipl",  # will match filenames containing 'ipl'
#     "CPL": "cpl",
#     "ILT20": "ilt20",
#     "LPL": "lpl",
#     "MLC": "mlc",
#     "SA20": "sa20",
#     "Super Smash": "super_smash",
#     "T20 Blast": "t20_blast",
#     "T20I": "t20i",
#     "BBL": "bbl",  # Added BBL
# }

# CACHE_DIR = "./.cache_data"
# os.makedirs(CACHE_DIR, exist_ok=True)


# def _hash_args(selected_tournaments, selected_years, usecols):
#     key = "|".join(sorted(selected_tournaments)) + "|" + f"{min(selected_years)}-{max(selected_years)}"
#     if usecols:
#         key += "|" + ",".join(sorted(usecols))
#     return hashlib.md5(key.encode()).hexdigest()


# def _find_best_file_for_tournament(tournament_key):
#     lowered = tournament_key.lower()
#     candidates = glob.glob(os.path.join(DATASETS_DIR, "*"))
#     candidates = [p for p in candidates if os.path.isfile(p)]
#     scored = []
#     for p in candidates:
#         name = os.path.basename(p).lower()
#         score = 0
#         if lowered in name:
#             score += 10
#         tokens = [t for t in "".join(ch if ch.isalnum() else " " for ch in name).split() if t]
#         if lowered in tokens:
#             score += 5
#         ext = os.path.splitext(p)[1].lower()
#         if ext == ".parquet":
#             score += 4
#         elif ext == ".csv":
#             score += 2
#         elif ext in (".xlsx", ".xls"):
#             score += 1
#         try:
#             mtime = os.path.getmtime(p)
#             age_score = int((datetime.now().timestamp() - mtime) // (24 * 3600))
#             score += max(0, 2 - min(age_score, 2))
#         except Exception:
#             pass
#         if score > 0:
#             scored.append((score, p))
#     if not scored:
#         return None
#     scored.sort(reverse=True, key=lambda x: (x[0], os.path.getmtime(x[1])))
#     return scored[0][1]


# def _detect_year_column(df):
#     if df is None or df.shape[1] == 0:
#         return None
#     col_candidates = [c for c in df.columns if 'year' in c.lower() or 'season' in c.lower()]
#     if not col_candidates:
#         return None
#     for pref in ['year', 'season']:
#         for c in col_candidates:
#             if pref == c.lower():
#                 return c
#     return col_candidates[0]


# def _extract_years_series(series):
#     try:
#         s = pd.to_numeric(series.astype(str).str[:4], errors='coerce').fillna(np.nan)
#         if s.notna().sum() > 0:
#             return s.astype('Int64')
#     except Exception:
#         pass
#     try:
#         sdt = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
#         if sdt.notna().sum() > 0:
#             return sdt.dt.year.astype('Int64')
#     except Exception:
#         pass
#     return None


# @st.cache_data(ttl=24 * 3600, show_spinner="Loading selected data (may take a moment for large leagues)…")
# def load_filtered_data_fast(selected_tournaments, selected_years, usecols=None, csv_chunksize=250_000):
#     if not selected_tournaments:
#         return pd.DataFrame()
#     cache_hash = _hash_args(selected_tournaments, selected_years, usecols=usecols)
#     cache_path = os.path.join(CACHE_DIR, f"merged_{cache_hash}.parquet")
#     if os.path.exists(cache_path):
#         try:
#             df_cached = pd.read_parquet(cache_path)
#             return df_cached
#         except Exception:
#             pass
#     parts = []
#     first_columns = None
#     for t in selected_tournaments:
#         mapped = TOURNAMENTS.get(t, None)
#         source_candidate = None
#         if mapped:
#             maybe = os.path.join(DATASETS_DIR, mapped) if not os.path.isabs(mapped) else mapped
#             if os.path.exists(maybe):
#                 source_candidate = maybe
#             else:
#                 found = _find_best_file_for_tournament(mapped)
#                 if found:
#                     source_candidate = found
#         if source_candidate is None:
#             source_candidate = _find_best_file_for_tournament(t)
#         if source_candidate is None:
#             st.warning(f"No dataset file found for tournament '{t}' in '{DATASETS_DIR}'. Skipping.")
#             continue
#         ext = os.path.splitext(source_candidate)[1].lower()
#         try:
#             if ext == ".parquet":
#                 df_temp = pd.read_parquet(source_candidate, columns=usecols)
#                 year_col = _detect_year_column(df_temp)
#                 if year_col:
#                     yrs = _extract_years_series(df_temp[year_col])
#                     if yrs is not None:
#                         df_temp = df_temp[yrs.isin(selected_years).fillna(False).values]
#                 if df_temp is None or df_temp.shape[0] == 0:
#                     continue
#             elif ext == ".csv":
#                 collected = []
#                 reader = pd.read_csv(source_candidate, usecols=usecols, chunksize=csv_chunksize, low_memory=True)
#                 for chunk in reader:
#                     year_col = _detect_year_column(chunk)
#                     if year_col:
#                         yrs = _extract_years_series(chunk[year_col])
#                         if yrs is not None:
#                             chunk = chunk[yrs.isin(selected_years).fillna(False).values]
#                     if not chunk.empty:
#                         collected.append(chunk)
#                 df_temp = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame(columns=usecols or [])
#                 if df_temp.shape[0] == 0:
#                     continue
#             elif ext in (".xlsx", ".xls"):
#                 df_temp = pd.read_excel(source_candidate, usecols=usecols)
#                 year_col = _detect_year_column(df_temp)
#                 if year_col:
#                     yrs = _extract_years_series(df_temp[year_col])
#                     if yrs is not None:
#                         df_temp = df_temp[yrs.isin(selected_years).fillna(False).values]
#                 try:
#                     parquet_equiv = os.path.splitext(source_candidate)[0] + ".parquet"
#                     df_temp.to_parquet(parquet_equiv, index=False)
#                 except Exception:
#                     pass
#                 if df_temp.shape[0] == 0:
#                     continue
#             else:
#                 try:
#                     df_temp = pd.read_parquet(source_candidate, columns=usecols)
#                 except Exception:
#                     collected = []
#                     reader = pd.read_csv(source_candidate, usecols=usecols, chunksize=csv_chunksize, low_memory=True)
#                     for chunk in reader:
#                         year_col = _detect_year_column(chunk)
#                         if year_col:
#                             yrs = _extract_years_series(chunk[year_col])
#                             if yrs is not None:
#                                 chunk = chunk[yrs.isin(selected_years).fillna(False).values]
#                         if not chunk.empty:
#                             collected.append(chunk)
#                     df_temp = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame(columns=usecols or [])
#                     if df_temp.shape[0] == 0:
#                         continue
#             df_temp['tournament'] = t
#             if first_columns is None:
#                 first_columns = df_temp.columns.tolist()
#             else:
#                 common = [c for c in first_columns if c in df_temp.columns]
#                 others = [c for c in df_temp.columns if c not in common]
#                 df_temp = df_temp[common + others]
#             parts.append(df_temp)
#         except Exception as exc:
#             st.warning(f"Failed to read {source_candidate} for '{t}': {str(exc)[:200]}")
#             continue
#     if not parts:
#         return pd.DataFrame()
#     merged = pd.concat(parts, ignore_index=True, sort=False)
#     year_col = _detect_year_column(merged)
#     if year_col:
#         yrs = _extract_years_series(merged[year_col])
#         if yrs is not None:
#             merged = merged[yrs.isin(selected_years).fillna(False).values]
#     try:
#         merged.to_parquet(cache_path, index=False)
#     except Exception:
#         pass
#     return merged


# # --------------------
# # USAGE in app
# # --------------------
# # ────────────────────────────────────────────────
# # Sidebar Controls with Full Persistence
# # ────────────────────────────────────────────────

# st.sidebar.header("Select Years")

# # Initialize year slider state if not present
# if "year_range" not in st.session_state:
#     st.session_state.year_range = (2021, 2026)

# years = st.sidebar.slider(
#     "Select year range",
#     min_value=2021,
#     max_value=2026,
#     value=st.session_state.year_range,
#     step=1,
#     key="year_slider_key"  # Unique key binds widget to state
# )

# # Always sync slider change back to session state
# st.session_state.year_range = years
# selected_years = list(range(years[0], years[1] + 1))
# st.sidebar.write(f"Selected years: {', '.join(map(str, selected_years))}")

# st.sidebar.header("Select Tournaments")
# all_tournaments = list(TOURNAMENTS.keys())

# # Initialize tournament selection state if not present
# if "selected_tournaments" not in st.session_state:
#     st.session_state.selected_tournaments = ["IPL"]

# selected_tournaments = st.sidebar.multiselect(
#     "Choose tournaments to load",
#     options=all_tournaments,
#     default=st.session_state.selected_tournaments,
#     key="tournament_select_key"  # Critical key for persistence
# )

# # Force sync selected value back to session state (handles edge cases)
# st.session_state.selected_tournaments = selected_tournaments

# # Debug (optional - remove later)
# # st.sidebar.write("Current session tournaments:", st.session_state.selected_tournaments)

# # ────────────────────────────────────────────────
# # Load data only when needed
# # ────────────────────────────────────────────────
# usecols = None  # change to specific columns if you want speedup

# if selected_tournaments:  # only load if at least one tournament selected
#     with st.spinner("Loading data (fast path) — this should be quick if parquet exists..."):
#         df = load_filtered_data_fast(selected_tournaments, selected_years, usecols=usecols)
    
#     if df is None or df.empty:
#         st.warning("No data loaded. Check selected tournaments, the Datasets folder, or the year range.")
#     else:
#         st.success(f"Loaded {len(df):,} rows from {len(selected_tournaments)} tournament(s) and {len(selected_years)} year(s).")
#         st.sidebar.success("Data loaded successfully!")
# else:
#     st.info("Please select at least one tournament to load data.")
#     df = pd.DataFrame()

# DF_gen = df

# =========================
# End loader
# =========================

# @st.cache_data
# def load_data():
#     path = "Datasets/ipl_bbb_21_25_2.xlsx"
#     df = pd.read_excel(path)

#     return df

# # df = load_data()    
# import streamlit as st
# import pandas as pd
# import requests
# from io import BytesIO

# # ────────────────────────────────────────────────
# # Tournament → file path or direct Dropbox URL
# # ────────────────────────────────────────────────
# TOURNAMENTS = {
#     "IPL": "Datasets/ipl_bbb_21_25_2.xlsx",               # your local .xlsx file
#     "CPL": "Datasets/IPL_APP_CPL.csv",
#     "ILT20": "Datasets/IPL_APP_ILT20.csv",
#     "LPL": "Datasets/IPL_APP_LPL.csv",
#     "MLC": "Datasets/IPL_APP_MLC.csv",
#     "SA20": "Datasets/IPL_APP_SA20.csv",
#     "Super Smash": "Datasets/IPL_APP_SuperSmash.csv",
    
#     # Large files from Dropbox (direct dl=1 links)
#     "T20 Blast": "https://www.dropbox.com/scl/fi/hyo26qc396k76lmawvt9i/IPL_APP_T20I_2.csv?rlkey=bc1rzwx1k64qwkkq9xxk6hpxc&st=ih914nfa&dl=1",
#     "T20I": "https://www.dropbox.com/scl/fi/pzxfy9bqoqtaiknli5oi2/IPL_APP_T20I.csv?rlkey=8xl4wb37yuq1ej85w122ldlxk&st=4m8xk916&dl=1",
# }

# # ────────────────────────────────────────────────
# # Year range slider (2021–2026)
# # ────────────────────────────────────────────────
# st.sidebar.header("Select Years")
# years = st.sidebar.slider(
#     "Select year range",
#     min_value=2021,
#     max_value=2026,
#     value=(2021, 2026),  # default full range
#     step=1
# )
# selected_years = list(range(years[0], years[1] + 1))
# st.sidebar.write(f"Selected years: {', '.join(map(str, selected_years))}")

# # ────────────────────────────────────────────────
# # Tournament multi-select
# # ────────────────────────────────────────────────
# st.sidebar.header("Select Tournaments")
# all_tournaments = list(TOURNAMENTS.keys())
# selected_tournaments = st.sidebar.multiselect(
#     "Choose tournaments to load",
#     options=all_tournaments,
#     default=["IPL"]  # start with IPL for speed
# )

# # ────────────────────────────────────────────────
# # Optimized loading + year filtering
# # ────────────────────────────────────────────────
# @st.cache_data(ttl="24h", show_spinner="Loading selected data (may take a moment for large leagues)…")
# import os
# import hashlib
# import requests
# from io import BytesIO
# import pandas as pd
# import numpy as np
# import streamlit as st

# CACHE_DIR = "./.cache_data"
# os.makedirs(CACHE_DIR, exist_ok=True)

# def _hash_args(selected_tournaments, selected_years):
#     key = "|".join(sorted(selected_tournaments)) + "|" + f"{min(selected_years)}-{max(selected_years)}"
#     return hashlib.md5(key.encode()).hexdigest()

# @st.cache_data(ttl=24*3600, show_spinner="Loading selected data (may take a moment)...")
# def load_filtered_data_fast(selected_tournaments, selected_years, usecols=None, csv_chunksize=200_000):
#     """
#     Fast loader:
#       - chunked reads for CSV
#       - caches per selection to ./ .cache_data/<hash>.parquet
#       - tries to use parquet preconverted for local xlsx files (recommended)
#     Args:
#       selected_tournaments: list of keys in TOURNAMENTS
#       selected_years: list of ints
#       usecols: list of column names you actually need (recommended)
#       csv_chunksize: chunk size for read_csv
#     Returns:
#       pandas.DataFrame (possibly empty)
#     """
#     if not selected_tournaments:
#         return pd.DataFrame()

#     # local cache per selection
#     cache_hash = _hash_args(selected_tournaments, selected_years)
#     cache_path = os.path.join(CACHE_DIR, f"merged_{cache_hash}.parquet")
#     if os.path.exists(cache_path):
#         try:
#             df_cached = pd.read_parquet(cache_path)
#             return df_cached
#         except Exception:
#             # fallback to re-create cache
#             pass

#     dfs = []
#     first_columns = None

#     for tournament in selected_tournaments:
#         source = TOURNAMENTS.get(tournament)
#         if not source:
#             continue

#         try:
#             # ---------- LOCAL FILES ----------
#             if not str(source).lower().startswith("http"):
#                 # If local xlsx exists, prefer a same-folder parquet (much faster). If not, we read the xlsx/ csv.
#                 if source.endswith(".xlsx"):
#                     parquet_equiv = os.path.splitext(source)[0] + ".parquet"
#                     if os.path.exists(parquet_equiv):
#                         # fast read
#                         df_temp = pd.read_parquet(parquet_equiv, columns=usecols)
#                     else:
#                         # read excel (slow) but we will convert & save to parquet for next time
#                         df_temp = pd.read_excel(source, usecols=usecols)
#                         try:
#                             df_temp.to_parquet(parquet_equiv, index=False)
#                         except Exception:
#                             pass
#                 else:
#                     # CSV: stream in chunks and filter by year on the fly
#                     if source.endswith(".csv"):
#                         reader = pd.read_csv(source, usecols=usecols, chunksize=csv_chunksize, low_memory=True)
#                         parts = []
#                         for chunk in reader:
#                             year_col = next((c for c in chunk.columns if 'year' in c.lower() or 'season' in c.lower()), None)
#                             if year_col:
#                                 # try to extract numeric year safely
#                                 try:
#                                     chunk_years = pd.to_numeric(chunk[year_col].astype(str).str[:4], errors='coerce').fillna(0).astype(int)
#                                     chunk = chunk[chunk_years.isin(selected_years)]
#                                 except Exception:
#                                     pass
#                             if not chunk.empty:
#                                 parts.append(chunk)
#                         df_temp = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)
#                     else:
#                         # Unknown local ext: try pandas autodetect
#                         df_temp = pd.read_csv(source, usecols=usecols) if source.endswith(".csv") else pd.read_excel(source, usecols=usecols)

#             # ---------- REMOTE (dropbox/http) ----------
#             else:
#                 # Use requests to fetch bytes (we keep it simple: read into BytesIO then chunk)
#                 resp = requests.get(source, timeout=180)
#                 resp.raise_for_status()
#                 content = BytesIO(resp.content)

#                 if str(source).lower().endswith(".csv") or '.csv?' in source.lower():
#                     reader = pd.read_csv(content, usecols=usecols, chunksize=csv_chunksize, low_memory=True)
#                     parts = []
#                     for chunk in reader:
#                         year_col = next((c for c in chunk.columns if 'year' in c.lower() or 'season' in c.lower()), None)
#                         if year_col:
#                             try:
#                                 chunk_years = pd.to_numeric(chunk[year_col].astype(str).str[:4], errors='coerce').fillna(0).astype(int)
#                                 chunk = chunk[chunk_years.isin(selected_years)]
#                             except Exception:
#                                 pass
#                         if not chunk.empty:
#                             parts.append(chunk)
#                     df_temp = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)
#                 else:
#                     # remote excel -> read into pandas (slow). If used often, convert to parquet offline.
#                     df_temp = pd.read_excel(content, usecols=usecols)

#             if df_temp is None or len(df_temp) == 0:
#                 continue

#             # Filter by year if possible (extra safe)
#             year_col = next((c for c in df_temp.columns if 'year' in c.lower() or 'season' in c.lower()), None)
#             if year_col:
#                 try:
#                     df_temp_years = pd.to_numeric(df_temp[year_col].astype(str).str[:4], errors='coerce').fillna(0).astype(int)
#                     df_temp = df_temp[df_temp_years.isin(selected_years)]
#                 except Exception:
#                     pass

#             # Tag source
#             df_temp['tournament'] = tournament

#             # Standardize columns order on first file read
#             if first_columns is None:
#                 first_columns = df_temp.columns.tolist()
#             else:
#                 # reindex to first_columns where possible (keeps consistent frames)
#                 common = [c for c in first_columns if c in df_temp.columns]
#                 others = [c for c in df_temp.columns if c not in common]
#                 df_temp = df_temp[common + others]

#             dfs.append(df_temp)

#         except Exception as exc:
#             st.warning(f"Failed to load {tournament}: {str(exc)[:200]} (skipping)")

#     if not dfs:
#         return pd.DataFrame()

#     merged_df = pd.concat(dfs, ignore_index=True)

#     # Save merged small cache (parquet) for instant reuse
#     try:
#         merged_df.to_parquet(cache_path, index=False)
#     except Exception:
#         pass

#     return merged_df


# # Load only when selections are made
# df = load_filtered_data(selected_tournaments, selected_years)

# # Feedback
# if not df.empty:
#     st.success(f"Loaded **{len(df):,} rows** from {len(selected_tournaments)} tournament(s) and {len(selected_years)} year(s)")
#     st.sidebar.write("Data loaded successfully!")
# else:
#     st.warning("No data loaded yet. Select at least one tournament and year range.")
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
# Mapping similar bowling styles together
bowl_style_map = {
    'LB': 'Leg break',
    'LBG': 'Leg break',
    'LF': 'Left-arm Fast',
    'LFM': 'Left-arm Fast',
    'LM': 'Left-arm Medium-Fast',
    'LMF': 'Left-arm Medium-Fast',
    'LWS': 'Left-arm Wrist Spin',
    'OB': 'Off break',
    'OB/LB': 'Offbreak/Legbreak',
    'RF': 'Right-arm Fast',
    'RFM': 'Right-arm Fast',
    'RM': 'Right-arm Medium-Fast',
    'RM/OB': 'Right-arm Medium',
    'RMF': 'Right-arm Medium-Fast',
    'SLA': 'Slow Left-arm Orthodox'
}

# Apply mapping to your dataframe
df['bowl_style_grouped'] = df['bowl_style'].map(bowl_style_map)
df['bowl_style_org']=df['bowl_style']
df['bowl_style']=df['bowl_style_grouped']

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# Optional Streamlit support
try:
    import streamlit as st
except Exception:
    st = None

# --- Sector name mapping (RHB canonical) ---
SECTOR_NAMES_RHB = {
    1: "Fine Leg",
    2: "Square Leg",
    3: "Mid Wicket",
    4: "Mid On",
    5: "Mid Off",
    6: "Covers",
    7: "Point",
    8: "Third Man"
}

# --- BASE_ANGLES for the canonical RHB layout (degrees) ---
BASE_ANGLES = {
    8: 112.5,  # Third Man
    7: 157.5,  # Point
    6: 202.5,  # Covers
    5: 247.5,  # Mid Off
    4: 292.5,  # Mid On
    3: 337.5,  # Mid Wicket
    2: 22.5,   # Square Leg
    1: 67.5    # Fine Leg
}

# --- EXPLICIT LHB mapping: reverse pairs (1<->8, 2<->7, 3<->6, 4<->5) ---
# This is exactly the swap you asked for (ThirdMan<->FineLeg, SquareLeg<->Point, MidWicket<->Covers, MidOn<->MidOff)
BASE_ANGLES_LHB = { z: float(BASE_ANGLES.get(9 - z, 0.0)) for z in range(1,9) }

def get_sector_angle_requested(zone, batting_style):
    """Return sector center in radians.

    For RHB: use BASE_ANGLES[zone].
    For LHB: use BASE_ANGLES_LHB[zone] (explicit reversed mapping).
    """
    z = int(zone)
    is_lhb = isinstance(batting_style, str) and batting_style.strip().upper().startswith('L')
    if not is_lhb:
        angle_deg = float(BASE_ANGLES.get(z, 0.0))
    else:
        angle_deg = float(BASE_ANGLES_LHB.get(z, 0.0))
    return math.radians(angle_deg)

def draw_cricket_field_with_run_totals_requested(final_df_local, batsman_name,
                                                wagon_zone_col='wagonZone',
                                                run_col='score',
                                                bat_hand_col='bat_hand',
                                                normalize_to_rhb=True):
    """
    Draw wagon wheel / scoring zones with explicit LHB angle swap implemented via BASE_ANGLES_LHB.
    When normalize_to_rhb=False and batter is LHB, shows true mirrored perspective.
    """
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

    # raw aggregates keyed by the raw sector id (1..8)
    runs_by_zone = tmp.groupby('wagon_zone_int')[run_col].sum().to_dict()
    fours_by_zone = tmp.groupby('wagon_zone_int')[run_col].apply(lambda s: int((s==4).sum())).to_dict()
    sixes_by_zone = tmp.groupby('wagon_zone_int')[run_col].apply(lambda s: int((s==6).sum())).to_dict()
    total_runs_in_wagon = sum(int(v) for v in runs_by_zone.values())

    # Title
    title_text = f"{batsman_name}'s Scoring Zones"
    plt.title(title_text, pad=20, color='white', size=14, fontweight='bold')

    # Determine batter handedness (first non-null sample in the filtered rows)
    batting_style_val = None
    if bat_hand_col in tmp.columns and not tmp[bat_hand_col].dropna().empty:
        batting_style_val = tmp[bat_hand_col].dropna().iloc[0]
    is_lhb = isinstance(batting_style_val, str) and batting_style_val.strip().upper().startswith('L')

    # Place % runs and runs in each sector using sector centers (angles chosen from explicit maps)
    for zone in range(1, 9):
        angle_mid = get_sector_angle_requested(zone, batting_style_val)
        x = 0.60 * math.cos(angle_mid)
        y = 0.60 * math.sin(angle_mid)

        data_zone = zone

        runs = int(runs_by_zone.get(data_zone, 0))
        pct = (runs / total_runs_in_wagon * 100) if total_runs_in_wagon > 0 else 0.0
        pct_str = f"{pct:.2f}%"

        # main labels
        ax.text(x, y+0.03, pct_str, ha='center', va='center', color='white', fontweight='bold', fontsize=18)
        ax.text(x, y-0.03, f"{runs} runs", ha='center', va='center', color='white', fontsize=10)

        # fours & sixes below
        fours = int(fours_by_zone.get(data_zone, 0))
        sixes = int(sixes_by_zone.get(data_zone, 0))
        ax.text(x, y-0.12, f"4s: {fours}  6s: {sixes}", ha='center', va='center', color='white', fontsize=9)

        # HARD-CODED FIX: Use SECTOR_NAMES_RHB directly with zone number
        # Since the wrapper already swapped the data zones, we just use the zone as-is
        sector_name_to_show = SECTOR_NAMES_RHB.get(zone, f"Sector {zone}")
        
        sx = 0.80 * math.cos(angle_mid)
        sy = 0.80 * math.sin(angle_mid)
        ax.text(sx, sy, sector_name_to_show, ha='center', va='center', color='white', fontsize=8)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    plt.tight_layout(pad=0)

    return fig
                                                  
def draw_wagon_if_available(df_wagon, batter_name, normalize_to_rhb=True):
    """
    Wrapper that calls draw_cricket_field_with_run_totals_requested consistently.
    
    - normalize_to_rhb: True => request RHB-normalised output (legacy behaviour).
                       False => request true handedness visualization (LHB will appear mirrored).
    
    This wrapper tries to call the function with the new parameter if available (backwards compatible).
    """
    import matplotlib.pyplot as plt
    import streamlit as st
    import inspect
    from matplotlib.figure import Figure as MplFigure
    
    # Defensive check
    if not isinstance(df_wagon, pd.DataFrame) or df_wagon.empty:
        st.warning("No wagon data available to draw.")
        return
    
    # Decide handedness (for UI messages / debugging)
    batting_style_val = None
    if 'bat_hand' in df_wagon.columns:
        try:
            batting_style_val = df_wagon['bat_hand'].dropna().iloc[0]
        except Exception:
            batting_style_val = None
    
    is_lhb = isinstance(batting_style_val, str) and batting_style_val.strip().upper().startswith('L')
    
    # Check function signature
    draw_fn = globals().get('draw_cricket_field_with_run_totals_requested', None)
    if draw_fn is None or not callable(draw_fn):
        st.warning("Wagon chart function not found; please ensure `draw_cricket_field_with_run_totals_requested` is defined earlier.")
        return
    
    try:
        sig = inspect.signature(draw_fn)
        if 'normalize_to_rhb' in sig.parameters:
            # call with the explicit flag (preferred)
            fig = draw_fn(df_wagon, batter_name, normalize_to_rhb=normalize_to_rhb)
        else:
            # older signature: call without flag (maintain legacy behaviour)
            fig = draw_fn(df_wagon, batter_name)
        
        # If the function returned a Matplotlib fig — display it
        if isinstance(fig, MplFigure):
            safe_fn = globals().get('safe_st_pyplot', None)
            if callable(safe_fn):
                try:
                    safe_fn(fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
                except Exception:
                    st.pyplot(fig)
            else:
                st.pyplot(fig)
            return
        
        # If function returned None, it may have drawn to current fig; capture that
        if fig is None:
            mpl_fig = plt.gcf()
            # If figure has axes and content, display it
            if isinstance(mpl_fig, MplFigure) and len(mpl_fig.axes) > 0:
                safe_fn = globals().get('safe_st_pyplot', None)
                if callable(safe_fn):
                    try:
                        safe_fn(mpl_fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
                    except Exception:
                        st.pyplot(mpl_fig)
                else:
                    st.pyplot(mpl_fig)
                return
        
        # If function returned a Plotly figure (rare), display it
        if isinstance(fig, go.Figure):
            try:
                fig.update_yaxes(scaleanchor="x", scaleratio=1)
            except Exception:
                pass
            st.plotly_chart(fig, use_container_width=True)
            return
        
        # Unknown return — just state it
        st.warning("Wagon draw function executed but returned an unexpected type; nothing displayed.")
        
    except Exception as e:
        st.error(f"Wagon drawing function raised: {e}")
# def draw_wagon_if_available(df_wagon, batter_name, normalize_to_rhb=True):
#     """
#     Wrapper that calls draw_cricket_field_with_run_totals_requested consistently.
    
#     - normalize_to_rhb: True => request RHB-normalised output (legacy behaviour).
#                        False => request true handedness visualization (LHB will appear mirrored).
    
#     This wrapper tries to call the function with the new parameter if available (backwards compatible).
#     """
#     import matplotlib.pyplot as plt
#     import streamlit as st
#     import inspect
#     from matplotlib.figure import Figure as MplFigure
    
#     # Defensive check
#     if not isinstance(df_wagon, pd.DataFrame) or df_wagon.empty:
#         st.warning("No wagon data available to draw.")
#         return
    
#     # Decide handedness (for UI messages / debugging)
#     batting_style_val = None
#     if 'bat_hand' in df_wagon.columns:
#         try:
#             batting_style_val = df_wagon['bat_hand'].dropna().iloc[0]
#         except Exception:
#             batting_style_val = None
    
#     is_lhb = isinstance(batting_style_val, str) and batting_style_val.strip().upper().startswith('L')
    
#     # Check function signature
#     draw_fn = globals().get('draw_cricket_field_with_run_totals_requested', None)
#     if draw_fn is None or not callable(draw_fn):
#         st.warning("Wagon chart function not found; please ensure `draw_cricket_field_with_run_totals_requested` is defined earlier.")
#         return
    
#     try:
#         sig = inspect.signature(draw_fn)
#         if 'normalize_to_rhb' in sig.parameters:
#             # call with the explicit flag (preferred)
#             fig = draw_fn(df_wagon, batter_name, normalize_to_rhb=normalize_to_rhb)
#         else:
#             # older signature: call without flag (maintain legacy behaviour)
#             fig = draw_fn(df_wagon, batter_name)
        
#         # If the function returned a Matplotlib fig — display it
#         if isinstance(fig, MplFigure):
#             safe_fn = globals().get('safe_st_pyplot', None)
#             if callable(safe_fn):
#                 try:
#                     safe_fn(fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
#                 except Exception:
#                     st.pyplot(fig)
#             else:
#                 st.pyplot(fig)
#             return
        
#         # If function returned None, it may have drawn to current fig; capture that
#         if fig is None:
#             mpl_fig = plt.gcf()
#             # If figure has axes and content, display it
#             if isinstance(mpl_fig, MplFigure) and len(mpl_fig.axes) > 0:
#                 safe_fn = globals().get('safe_st_pyplot', None)
#                 if callable(safe_fn):
#                     try:
#                         safe_fn(mpl_fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
#                     except Exception:
#                         st.pyplot(mpl_fig)
#                 else:
#                     st.pyplot(mpl_fig)
#                 return
        
#         # If function returned a Plotly figure (rare), display it
#         if isinstance(fig, go.Figure):
#             try:
#                 fig.update_yaxes(scaleanchor="x", scaleratio=1)
#             except Exception:
#                 pass
#             st.plotly_chart(fig, use_container_width=True)
#             return
        
#         # Unknown return — just state it
#         st.warning("Wagon draw function executed but returned an unexpected type; nothing displayed.")
        
#     except Exception as e:
#         st.error(f"Wagon drawing function raised: {e}")
# def draw_cricket_field_with_run_totals_requested(final_df_local, batsman_name,
#                                                 wagon_zone_col='wagonZone',
#                                                 run_col='score',
#                                                 bat_hand_col='bat_hand'):
#     """
#     Draw wagon wheel / scoring zones with explicit LHB angle swap implemented via BASE_ANGLES_LHB.
#     No other logic changed.
#     """
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.set_aspect('equal')
#     ax.axis('off')

#     # Field base
#     ax.add_patch(Circle((0, 0), 1, fill=True, color='#228B22', alpha=1))
#     ax.add_patch(Circle((0, 0), 1, fill=False, color='black', linewidth=3))
#     ax.add_patch(Circle((0, 0), 0.5, fill=True, color='#66bb6a'))
#     ax.add_patch(Circle((0, 0), 0.5, fill=False, color='white', linewidth=1))

#     # pitch rectangle approx
#     pitch_rect = Rectangle((-0.04, -0.08), 0.08, 0.16, color='tan', alpha=1, zorder=8)
#     ax.add_patch(pitch_rect)

#     # radial sector lines (8)
#     angles = np.linspace(0, 2*np.pi, 9)[:-1]
#     for angle in angles:
#         x = math.cos(angle)
#         y = math.sin(angle)
#         ax.plot([0, x], [0, y], color='white', alpha=0.25, linewidth=1)

#     # prepare data (runs, fours, sixes by sector)
#     tmp = final_df_local.copy()
#     if wagon_zone_col in tmp.columns:
#         tmp['wagon_zone_int'] = pd.to_numeric(tmp[wagon_zone_col], errors='coerce').astype('Int64')
#     else:
#         tmp['wagon_zone_int'] = pd.Series(dtype='Int64')

#     # raw aggregates keyed by the raw sector id (1..8)
#     runs_by_zone = tmp.groupby('wagon_zone_int')[run_col].sum().to_dict()
#     fours_by_zone = tmp.groupby('wagon_zone_int')[run_col].apply(lambda s: int((s==4).sum())).to_dict()
#     sixes_by_zone = tmp.groupby('wagon_zone_int')[run_col].apply(lambda s: int((s==6).sum())).to_dict()
#     total_runs_in_wagon = sum(int(v) for v in runs_by_zone.values())

#     # Title
#     title_text = f"{batsman_name}'s Scoring Zones"
#     plt.title(title_text, pad=20, color='white', size=14, fontweight='bold')

#     # Determine batter handedness (first non-null sample in the filtered rows)
#     batting_style_val = None
#     if bat_hand_col in tmp.columns and not tmp[bat_hand_col].dropna().empty:
#         batting_style_val = tmp[bat_hand_col].dropna().iloc[0]
#     is_lhb = isinstance(batting_style_val, str) and batting_style_val.strip().upper().startswith('L')

#     # Place % runs and runs in each sector using sector centers (angles chosen from explicit maps)
#     for zone in range(1, 9):
#         angle_mid = get_sector_angle_requested(zone, batting_style_val)
#         x = 0.60 * math.cos(angle_mid)
#         y = 0.60 * math.sin(angle_mid)

#         # data_zone: use raw sector index (1..8) directly; we only changed the display angle mapping
#         # BUT to keep labels and data aligned visually for LHB we need to pick the data zone that corresponds
#         # to the displayed sector name. With BASE_ANGLES_LHB we've already swapped sector angles, so the display
#         # location uses the mirrored mapping. To ensure the numeric data (runs/fours/sixes) goes into the
#         # correct display sector we should still pull from the *raw* zone that the batter's wagon data used.
#         # The approach below keeps behaviour identical to prior logic: show data for the raw zone index.
#         data_zone = zone

#         runs = int(runs_by_zone.get(data_zone, 0))
#         pct = (runs / total_runs_in_wagon * 100) if total_runs_in_wagon > 0 else 0.0
#         pct_str = f"{pct:.2f}%"

#         # main labels
#         ax.text(x, y+0.03, pct_str, ha='center', va='center', color='white', fontweight='bold', fontsize=18)
#         ax.text(x, y-0.03, f"{runs} runs", ha='center', va='center', color='white', fontsize=10)

#         # fours & sixes below
#         fours = int(fours_by_zone.get(data_zone, 0))
#         sixes = int(sixes_by_zone.get(data_zone, 0))
#         ax.text(x, y-0.12, f"4s: {fours}  6s: {sixes}", ha='center', va='center', color='white', fontsize=9)

#         # sector name slightly farther out:
#         # For LHB we display the mirrored sector name so label and data match visually.
#         display_sector_idx = zone if not is_lhb else (9 - zone)
#         sector_name_to_show = SECTOR_NAMES_RHB.get(display_sector_idx, f"Sector {display_sector_idx}")
#         sx = 0.80 * math.cos(angle_mid)
#         sy = 0.80 * math.sin(angle_mid)
#         ax.text(sx, sy, sector_name_to_show, ha='center', va='center', color='white', fontsize=8)

#     ax.set_xlim(-1.2, 1.2)
#     ax.set_ylim(-1.2, 1.2)
#     plt.tight_layout(pad=0)
#     # if is_lhb:
#     #     ax.invert_xaxis()

#     return fig

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import streamlit as st
import plotly.graph_objects as go
from matplotlib.figure import Figure as MplFigure
import inspect

# -------------------------
# 1) Updated draw_cricket_field_with_run_totals_requested
# -------------------------
# def draw_cricket_field_with_run_totals_requested(
#     final_df_local,
#     batsman_name,
#     wagon_zone_col='wagonZone',
#     run_col='score',
#     bat_hand_col='bat_hand',
#     normalize_to_rhb=True,   # NEW flag: True => show everything in RHB frame (legacy), False => show true handedness
# ):
#     """
#     Draw wagon wheel / scoring zones.
#     - normalize_to_rhb=True: place sectors in RHB reference frame (analytics default).
#     - normalize_to_rhb=False: place sectors in batter's natural handedness (LHB shows left-right mirror visually).
#     Returns: matplotlib.figure.Figure
#     """
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.set_aspect('equal')
#     ax.axis('off')

#     # Field base
#     ax.add_patch(Circle((0, 0), 1, fill=True, color='#228B22', alpha=1))
#     ax.add_patch(Circle((0, 0), 1, fill=False, color='black', linewidth=3))
#     ax.add_patch(Circle((0, 0), 0.5, fill=True, color='#66bb6a'))
#     ax.add_patch(Circle((0, 0), 0.5, fill=False, color='white', linewidth=1))

#     # pitch rectangle approx
#     pitch_rect = Rectangle((-0.04, -0.08), 0.08, 0.16, color='tan', alpha=1, zorder=8)
#     ax.add_patch(pitch_rect)

#     # radial sector lines (8)
#     angles = np.linspace(0, 2*np.pi, 9)[:-1]
#     for angle in angles:
#         x = math.cos(angle)
#         y = math.sin(angle)
#         ax.plot([0, x], [0, y], color='white', alpha=0.25, linewidth=1)

#     # prepare data (runs, fours, sixes by sector)
#     tmp = final_df_local.copy()
#     if wagon_zone_col in tmp.columns:
#         tmp['wagon_zone_int'] = pd.to_numeric(tmp[wagon_zone_col], errors='coerce').astype('Int64')
#     else:
#         tmp['wagon_zone_int'] = pd.Series(dtype='Int64')

#     # raw aggregates keyed by the raw sector id (1..8)
#     # ensure run_col present, coerce to numeric default 0
#     if run_col in tmp.columns:
#         tmp[run_col] = pd.to_numeric(tmp[run_col], errors='coerce').fillna(0)
#     else:
#         tmp[run_col] = 0

#     runs_by_zone = tmp.groupby('wagon_zone_int')[run_col].sum().to_dict()
#     # count fours / sixes robustly (if run_col is per-ball run)
#     fours_by_zone = tmp[tmp[run_col] == 4].groupby('wagon_zone_int')[run_col].count().to_dict()
#     sixes_by_zone = tmp[tmp[run_col] == 6].groupby('wagon_zone_int')[run_col].count().to_dict()
#     total_runs_in_wagon = sum(float(v) for v in runs_by_zone.values())

#     # Title
#     title_text = f"{batsman_name}'s Scoring Zones"
#     plt.title(title_text, pad=20, color='white', size=14, fontweight='bold')

#     # Determine batter handedness (first non-null sample)
#     batting_style_val = None
#     if bat_hand_col in tmp.columns and not tmp[bat_hand_col].dropna().empty:
#         batting_style_val = tmp[bat_hand_col].dropna().iloc[0]
#     is_lhb = isinstance(batting_style_val, str) and batting_style_val.strip().upper().startswith('L')

#     # Choose display_hand for angle computation:
#     # If normalize_to_rhb is True, force 'R' (everyone shown in RHB frame).
#     # If False, use batter's handedness so LHB will get L angles.
#     display_hand = 'R' if normalize_to_rhb else (batting_style_val if isinstance(batting_style_val, str) else 'R')

#     # Place % runs and runs in each sector using sector centers (angles chosen from get_sector_angle_requested)
#     for zone in range(1, 9):
#         angle_mid = get_sector_angle_requested(zone, display_hand)   # IMPORTANT: pass display_hand explicitly
#         x = 0.60 * math.cos(angle_mid)
#         y = 0.60 * math.sin(angle_mid)

#         data_zone = zone  # use raw zone index for numeric data (keeps aggregation logic stable)

#         runs = int(runs_by_zone.get(data_zone, 0))
#         pct = (runs / total_runs_in_wagon * 100) if total_runs_in_wagon > 0 else 0.0
#         pct_str = f"{pct:.2f}%"

#         # main labels
#         ax.text(x, y+0.03, pct_str, ha='center', va='center', color='white', fontweight='bold', fontsize=18)
#         ax.text(x, y-0.03, f"{runs} runs", ha='center', va='center', color='white', fontsize=10)

#         # fours & sixes below
#         fours = int(fours_by_zone.get(data_zone, 0))
#         sixes = int(sixes_by_zone.get(data_zone, 0))
#         ax.text(x, y-0.12, f"4s: {fours}  6s: {sixes}", ha='center', va='center', color='white', fontsize=9)

#         # sector name slightly farther out:
#         # We display sector name for the display_hand (angles already place text in correct location).
#         sector_name_to_show = SECTOR_NAMES_RHB.get(zone, f"Sector {zone}")  # keep canonical names
#         sx = 0.80 * math.cos(angle_mid)
#         sy = 0.80 * math.sin(angle_mid)
#         ax.text(sx, sy, sector_name_to_show, ha='center', va='center', color='white', fontsize=8)

#     ax.set_xlim(-1.2, 1.2)
#     ax.set_ylim(-1.2, 1.2)
#     plt.tight_layout(pad=0)

#     # IMPORTANT: do NOT invert axes here. Axis inversion is a different strategy.
#     # If normalize_to_rhb==False and you want an axis-inversion-based mirror, you can invert externally.
#     # But for consistency we avoid double-handling. We return the figure as drawn for display_hand used.
#     return fig
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
                if d.at[i, 'out_flag'] != 1: # Check if out=True
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
    d = d.dropna(subset=['batsman', 'match_id'])
    # last snapshot per batsman per match
    last_bat_snapshot = (
        d.groupby(['batsman', 'match_id'], sort=False).agg(match_runs=('cur_bat_runs', 'last'), match_balls=('cur_bat_bf', 'last')).reset_index()
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
    ("Player Profile", "Matchup Analysis", "Strength vs Weakness", "Match by Match Analysis","Integrated Contextual Ratings")
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
    bdf=df
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
    
    # ---------- Career Batting ----------
    if option == "Batting":
        player_stats = as_dataframe(idf[idf['batsman'] == player_name])
        if player_stats is None or player_stats.empty:
            st.warning(f"No data available for {player_name}.")
            st.stop()
    
        # cleanup
        player_stats = player_stats.drop(columns=['final_year'], errors='ignore')
        player_stats.columns = [str(col).upper().replace('_', ' ') for col in player_stats.columns]
        player_stats = round_up_floats(player_stats)
    
        int_cols = ['RUNS', 'HUNDREDS', 'FIFTIES', 'THIRTIES', 'HIGHEST SCORE']
        for c in int_cols:
            if c in player_stats.columns:
                player_stats[c] = pd.to_numeric(player_stats[c], errors='coerce').fillna(0).astype(int)
    
        st.markdown("### Batting Statistics")
    
        # helper to find column
        def find_col(df, candidates):
            for cand in candidates:
                if cand in df.columns:
                    return cand
            return None
    
        top_metric_mapping = {
            "Runs": ["RUNS"],
            "Innings": ["INNINGS", "MATCHES"],
            "Average": ["AVG", "AVERAGE"],
            "Strike Rate": ["SR", "STRIKE RATE"],
            "Highest Score": ["HIGHEST SCORE", "HS"],
            "50s": ["FIFTIES", "50S"],
            "100s": ["HUNDREDS", "100S"],
        }
    
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
    
        # # ========== RAA / DAA computation ==========
        # if 'bdf' not in globals():
        #     st.warning("Global dataframe `bdf` not found; skipping RAA/DAA.")
        #     avg_RAA = avg_DAA = np.nan
        # else:
        #     bdf = bdf.copy()
        #     batter_col = 'bat' if 'bat' in bdf.columns else ('batsman' if 'batsman' in bdf.columns else None)
        #     if batter_col is None:
        #         st.warning("No batter column found in bdf.")
        #         avg_RAA = avg_DAA = np.nan
        #     else:
        #         # mark top7 batters per innings
        #         if 'p_bat' in bdf.columns:
        #             bdf['top7_flag'] = (pd.to_numeric(bdf['p_bat'], errors='coerce').fillna(99) <= 7).astype(int)
        #         else:
        #             # derive via first appearance
        #             bdf['_order'] = bdf.groupby(['p_match','inns']).cumcount()
        #             first_app = bdf.groupby(['p_match','inns',batter_col], as_index=False)['_order'].min()
        #             top7 = first_app.groupby(['p_match','inns']).apply(lambda x: x.nsmallest(7, '_order')).reset_index(drop=True)
        #             top7['top7_flag'] = 1
        #             bdf = bdf.merge(top7[['p_match','inns',batter_col,'top7_flag']], how='left', on=['p_match','inns',batter_col])
        #             bdf['top7_flag'] = bdf['top7_flag'].fillna(0).astype(int)
    
        #         # compute for each top7 batter
        #         top7 = bdf[bdf['top7_flag'] == 1]
        #         if top7.empty:
        #             st.warning("No top-7 rows found in bdf; cannot compute average RAA/DAA.")
        #             avg_RAA = avg_DAA = np.nan
        #         else:
        #             # SR and Balls per dismissal (BPD)
        #             sr_all = (
        #                 top7.groupby(batter_col)
        #                 .agg(runs=('batruns','sum'), balls=('ball','count'),
        #                      dismissals=('is_wicket','sum'))
        #                 .reset_index()
        #             )
        #             sr_all['SR'] = sr_all['runs'] / sr_all['balls'] * 100
        #             sr_all['BPD'] = sr_all.apply(
        #                 lambda x: x['balls']/x['dismissals'] if x['dismissals']>0 else np.nan, axis=1
        #             )
    
        #             # compute averages (excluding player)
        #             player_data = sr_all[sr_all[batter_col] == player_name]
        #             rest = sr_all[sr_all[batter_col] != player_name]
        #             avg_SR_others = rest['SR'].mean() if not rest.empty else np.nan
        #             avg_BPD_others = rest['BPD'].mean() if not rest.empty else np.nan
    
        #             if not player_data.empty:
        #                 player_SR = player_data['SR'].values[0]
        #                 player_BPD = player_data['BPD'].values[0]
        #                 avg_RAA = round(player_SR - avg_SR_others, 2) if not np.isnan(avg_SR_others) else np.nan
        #                 avg_DAA = round(player_BPD - avg_BPD_others, 2) if not np.isnan(avg_BPD_others) else np.nan
        #             else:
        #                 avg_RAA = avg_DAA = np.nan
    
        # found_top_cols["RAA"] = avg_RAA
        # found_top_cols["DAA"] = avg_DAA
    
        # display metrics
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
                    # 🔥 smart formatting
                    if "strike" in label.lower():
                        disp = f"{val:.1f}"        # SR → 1 decimal
                    elif "average" in label.lower():
                        disp = f"{val:.2f}"        # Avg → 2 decimals
                    else:
                        disp = f"{val:.2f}"
        
                else:
                    disp = str(val)
        
                col.metric(label, disp)
        else:
            st.write("Top metrics not available.")


    
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
        
            def fmt_val_with_col(metric_name, value):
                if pd.isna(value):
                    return ""
            
                metric_lower = str(metric_name).lower()
            
                # Force INT for Balls / Runs / Innings
                if any(k in metric_lower for k in ["balls", "runs", "innings"]):
                    try:
                        return str(int(round(float(value))))
                    except Exception:
                        return str(value)
            
                # Floats → clean formatting
                try:
                    val = float(value)
                    if abs(val - round(val)) < 1e-6:
                        return str(int(round(val)))      # 30.000000 → "30"
                    return f"{val:.2f}".rstrip("0").rstrip(".")  # 13.190000 → "13.19"
                except Exception:
                    return str(value)


        
            rest_df["Value"] = [
                fmt_val_with_col(m, v) for m, v in zip(rest_df["Metric"], rest_df["Value"])
            ]
            rest_df = rest_df[rest_df["Metric"].str.upper() != "BATTING TEAM"]


        
            detailed_header_color = "#fff0e6"
            
            detailed_table_styles = [
                # Header row
                {
                    "selector": "thead th",
                    "props": [
                        ("background-color", detailed_header_color),
                        ("color", "#000"),
                        ("font-weight", "600"),
                        ("text-align", "center"),
                    ],
                },
                # Body cells
                {
                    "selector": "tbody td",
                    "props": [
                        ("text-align", "center"),
                        ("vertical-align", "middle"),
                    ],
                },
                # Zebra striping
                {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
                {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#fff9f4")]},
            ]

            styled_df = (
                rest_df
                .style
                .set_properties(**{
                    "text-align": "center",
                    "vertical-align": "middle"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#fff0e6"),
                            ("color", "#000"),
                            ("font-weight", "600"),
                            ("text-align", "center"),
                        ],
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("text-align", "center"),
                        ],
                    },
                ])
                .hide(axis="index")
            )

            st.markdown("#### Detailed Stats")
            st.dataframe(styled_df, use_container_width=True)

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

            # -------------------------
            # Top metrics as metric cards
            # -------------------------
            visible_metrics = [
                (k, v) for k, v in found_top_cols.items()
                if v is not None and not (isinstance(v, float) and np.isnan(v))
            ]
            
            if visible_metrics:
            
                MAX_PER_ROW = 4   # 🔥 key fix (3–4 is ideal)
            
                for row_metrics in chunk_list(visible_metrics, MAX_PER_ROW):
                    cols = st.columns(len(row_metrics))
            
                    for (label, val), col in zip(row_metrics, cols):
                        label_l = label.lower()
            
                        # ---- BBI: always string
                        if "bbi" in label_l:
                            disp = str(val)
            
                        elif isinstance(val, (int, np.integer)):
                            disp = str(int(val))
            
                        elif isinstance(val, (float, np.floating)) and not np.isnan(val):
                            if "strike" in label_l:
                                disp = f"{val:.1f}"
                            elif "average" in label_l:
                                disp = f"{val:.2f}".rstrip("0").rstrip(".")
                            else:
                                disp = f"{val:.2f}".rstrip("0").rstrip(".")
            
                        else:
                            disp = str(val)
            
                        col.metric(label, disp)
            
            else:
                st.write("Top bowling metrics not available.")


            
            # -------------------------
            # Detailed stats (vertical key:value)
            # keep RUNS, remove BOWLING TEAM
            # -------------------------
            top_cols_used = [find_col(disp_stats, cand) for cand in top_metric_mapping.values()]
            top_cols_used = [c for c in top_cols_used if c is not None]
            top_cols_used_excluding_runs = [
                c for c in top_cols_used
                if c is not None and str(c).upper() != "RUNS"
            ]
            
            try:
                rest_series = disp_stats.iloc[0].drop(
                    labels=top_cols_used_excluding_runs, errors="ignore"
                )
            except Exception:
                rest_series = pd.Series(dtype=object)
            
            if not rest_series.empty:
                rest_df = rest_series.reset_index()
                rest_df.columns = ["Metric", "Value"]
            
                # ❌ remove BOWLING TEAM row
                rest_df = rest_df[
                    ~rest_df["Metric"].str.lower().str.contains("bowling team")
                ]
            
                # -------------------------
                # Formatting (STRING SAFE – STREAMLIT PROOF)
                # -------------------------
                def fmt_val_with_col(metric_name, value):
                    if pd.isna(value):
                        return ""
            
                    metric_lower = str(metric_name).lower()
            
                    # force int for balls / runs / innings
                    if any(k in metric_lower for k in ["balls", "runs", "innings"]):
                        try:
                            return str(int(round(float(value))))
                        except Exception:
                            return str(value)
            
                    # numeric formatting
                    try:
                        val = float(value)
                        if abs(val - round(val)) < 1e-6:
                            return str(int(round(val)))
                        return f"{val:.2f}".rstrip("0").rstrip(".")
                    except Exception:
                        return str(value)
            
                rest_df["Value"] = [
                    fmt_val_with_col(m, v)
                    for m, v in zip(rest_df["Metric"], rest_df["Value"])
                ]
                # -------------------------------------------------
                # Remove unwanted metadata rows
                # -------------------------------------------------
                remove_keywords = [
                    "unknown",
                    "mega over",
                    "debut year",
                    "final year",
                    "match id"
                ]
                
                rest_df = rest_df[
                    ~rest_df["Metric"].str.lower().apply(
                        lambda x: any(k in x for k in remove_keywords)
                    )
                ]

                # -------------------------------------------------
                # Rename specific metric labels
                # -------------------------------------------------
                rest_df["Metric"] = (
                    rest_df["Metric"]
                    .str.replace("7 PLUS RUN OVERS", ">6 RUN OVERS", case=False, regex=False)
                    .str.replace("6 MINUS RUN OVERS", "<7 RUN OVERS", case=False, regex=False)
                )


            
                # -------------------------
                # Styling + Center alignment
                # -------------------------
                detailed_header_color = "#fff0e6"
            
                styled_df = (
                    rest_df
                    .style
                    .set_properties(**{
                        "text-align": "center",
                        "vertical-align": "middle"
                    })
                    .set_table_styles([
                        {
                            "selector": "th",
                            "props": [
                                ("background-color", detailed_header_color),
                                ("color", "#000"),
                                ("font-weight", "600"),
                                ("text-align", "center"),
                            ],
                        },
                        {
                            "selector": "td",
                            "props": [("text-align", "center")],
                        },
                        {
                            "selector": "tbody tr:nth-child(odd)",
                            "props": [("background-color", "#ffffff")],
                        },
                        {
                            "selector": "tbody tr:nth-child(even)",
                            "props": [("background-color", "#fff9f4")],
                        },
                    ])
                    .hide(axis="index")
                )
            
                st.markdown("#### Detailed Stats")
                st.dataframe(styled_df, use_container_width=True)
            
            else:
                st.write("No detailed bowling metrics available.")
  
            # st.markdown("### Bowling Statistics")
    
            # # show top metrics as metric cards
            # visible_metrics = [
            #     (k, v) for k, v in found_top_cols.items()
            #     if v is not None and not (isinstance(v, float) and np.isnan(v))
            # ]
            # if visible_metrics:
            #     cols = st.columns(len(visible_metrics))
            #     for (label, val), col in zip(visible_metrics, cols):
            #         if isinstance(val, (int, np.integer)):
            #             disp = f"{int(val)}"
            #         elif isinstance(val, (float, np.floating)) and not np.isnan(val):
            #             disp = f"{val:.2f}"
            #         else:
            #             disp = str(val)
            #         col.metric(label, disp)
            # else:
            #     st.write("Top bowling metrics not available.")
    
            # # -------------------------
            # # Detailed stats (vertical key:value). keep RUNS displayed
            # # -------------------------
            # top_cols_used = [find_col(disp_stats, cand) for cand in top_metric_mapping.values()]
            # top_cols_used = [c for c in top_cols_used if c is not None]
            # top_cols_used_excluding_runs = [c for c in top_cols_used if c is not None and str(c).upper() != 'RUNS']
    
            # try:
            #     rest_series = disp_stats.iloc[0].drop(labels=top_cols_used_excluding_runs, errors='ignore')
            # except Exception:
            #     rest_series = pd.Series(dtype=object)
    
            # if not rest_series.empty:
            #     rest_df = rest_series.reset_index()
            #     rest_df.columns = ["Metric", "Value"]
    
            #     def fmt_val(x):
            #         if pd.isna(x):
            #             return ""
            #         if isinstance(x, (int, np.integer)):
            #             return int(x)
            #         if isinstance(x, (float, np.floating)):
            #             return round(x, 2)
            #         return x
    
            #     rest_df["Value"] = rest_df["Value"].apply(fmt_val)
    
            #     # light skin / peach header color
            #     detailed_header_color = "#fff0e6"
            #     detailed_table_styles = [
            #         {"selector": "thead th", "props": [("background-color", detailed_header_color), ("color", "#000"), ("font-weight", "600")]},
            #         {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
            #         {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#fff9f4")]},
            #     ]
    
            #     st.markdown("#### Detailed stats")
            #     st.dataframe(rest_df.style.set_table_styles(detailed_table_styles), use_container_width=True)
            # else:
            #     st.write("No detailed bowling metrics available.")
    
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
        'YORKER': 4,
        'FULLTOSS': 5
    }
    # ---------- robust map-lookup helpers ----------
    def _norm_key(s):
        if s is None:
            return ''
        return str(s).strip().upper().replace(' ', '_').replace('-', '_')
   
    def get_map_index(map_obj, raw_val):
        if raw_val is None:
            return None
        sval = str(raw_val).strip()
        if sval == '' or sval.lower() in ('nan', 'none'):
            return None
   
        if sval in map_obj:
            return int(map_obj[sval])
        s_norm = _norm_key(sval)
        for k in map_obj:
            try:
                if isinstance(k, str) and _norm_key(k) == s_norm:
                    return int(map_obj[k])
            except Exception:
                continue
        for k in map_obj:
            try:
                if isinstance(k, str) and (k.lower() in sval.lower() or sval.lower() in k.lower()):
                    return int(map_obj[k])
            except Exception:
                continue
        return None


    def assign_phase(over):
        if pd.isna(over):
            return 'Unknown'
        try:
            over_int = int(float(over))  # Handle float/string
            if 1 <= over_int <= 6:
                return 'Powerplay'
            elif 7 <= over_int <= 11:
                return 'Middle 1'
            elif 12 <= over_int <= 16:
                return 'Middle 2'
            elif 17 <= over_int <= 20:
                return 'Death'
            else:
                return 'Unknown'
        except:
            return 'Unknown'  
    # ---------- grids builder ----------
    def build_pitch_grids(df_in, line_col_name='line', length_col_name='length', runs_col_candidates=('batruns', 'score'),
                          control_col='control', dismissal_col='dismissal'):
        if 'length_map' in globals() and isinstance(length_map, dict) and len(length_map) > 0:
            try:
                max_idx = max(int(v) for v in length_map.values())
                n_rows = max(5, max_idx + 1)
            except Exception:
                n_rows = 5
        else:
            n_rows = 5
            st.warning("length_map not found; defaulting to 5 rows.")
   
        length_vals = df_in.get(length_col_name, pd.Series()).dropna().astype(str).str.lower().unique()
        if any('full toss' in val for val in length_vals):
            n_rows = max(n_rows, 6)
   
        n_cols = 5
   
        count = np.zeros((n_rows, n_cols), dtype=int)
        bounds = np.zeros((n_rows, n_cols), dtype=int)
        dots = np.zeros((n_rows, n_cols), dtype=int)
        runs = np.zeros((n_rows, n_cols), dtype=float)
        wkt = np.zeros((n_rows, n_cols), dtype=int)
        ctrl_not = np.zeros((n_rows, n_cols), dtype=int)
   
        runs_col = None
        for c in runs_col_candidates:
            if c in df_in.columns:
                runs_col = c
                break
        if runs_col is None:
            runs_col = None # will use 0
   
        wkt_tokens = {'caught', 'bowled', 'stumped', 'lbw','leg before wicket','hit wicket'}
        dismissal_series = df_in[dismissal_col].fillna('').astype(str).str.lower()
        for _, row in df_in.iterrows():
            li = get_map_index(line_map, row.get(line_col_name, None)) if 'line_map' in globals() else None
            le = get_map_index(length_map, row.get(length_col_name, None)) if 'length_map' in globals() else None
            if li is None or le is None:
                continue
            if not (0 <= le < n_rows and 0 <= li < n_cols):
                continue
            count[le, li] += 1
            rv = 0
            if runs_col:
                try:
                    rv = int(row.get(runs_col, 0) or 0)
                except:
                    rv = 0
            runs[le, li] += rv
            if rv >= 4:
                bounds[le, li] += 1
            if rv == 0:
                dots[le, li] += 1
            dval = str(row.get(dismissal_col, '') or '').lower()
            if any(tok in dval for tok in wkt_tokens):
                wkt[le, li] += 1
            cval = row.get(control_col, None)
            if cval is not None:
                if isinstance(cval, str) and 'not' in cval.lower():
                    ctrl_not[le, li] += 1
                elif isinstance(cval, (int, float)) and float(cval) == 0:
                    ctrl_not[le, li] += 1
   
        sr = np.full(count.shape, np.nan)
        ctrl_pct = np.full(count.shape, np.nan)
        for i in range(n_rows):
            for j in range(n_cols):
                if count[i, j] > 0:
                    sr[i, j] = runs[i, j] / count[i, j] * 100.0
                    ctrl_pct[i, j] = (ctrl_not[i, j] / count[i, j]) * 100.0
   
        return {
            'count': count, 'bounds': bounds, 'dots': dots,
            'runs': runs, 'sr': sr, 'ctrl_pct': ctrl_pct, 'wkt': wkt, 'n_rows': n_rows, 'n_cols': n_cols
        }


    def display_pitchmaps_from_df(df_src, title_prefix):
          if df_src is None or df_src.empty:
              st.info(f"No deliveries to show for {title_prefix}")
              return
    
          grids = build_pitch_grids(df_src)
    
          bh_col_name = globals().get('bat_hand_col', 'bat_hand')
          is_lhb = False
          if bh_col_name in df_src.columns:
              hands = df_src[bh_col_name].dropna().astype(str).str.strip().unique()
              if any(h.upper().startswith('L') for h in hands):
                  is_lhb = True
    
          def maybe_flip(arr):
              return np.fliplr(arr) if is_lhb else arr.copy()
    
          count = maybe_flip(grids['count'])
          bounds = maybe_flip(grids['bounds'])
          dots = maybe_flip(grids['dots'])
          sr = maybe_flip(grids['sr'])
          ctrl = maybe_flip(grids['ctrl_pct'])
          wkt = maybe_flip(grids['wkt'])
          runs = maybe_flip(grids['runs'])
    
          total = count.sum() if count.sum() > 0 else 1.0
          perc = count.astype(float) / total * 100.0
    
          # NEW: Calculate percentages
          total_bounds = bounds.sum() if bounds.sum() > 0 else 1.0
          bound_pct = bounds.astype(float) / total_bounds * 100.0
    
          total_dots = dots.sum() if dots.sum() > 0 else 1.0
          dot_pct = dots.astype(float) / total_dots * 100.0
    
          total_runs = runs.sum() if runs.sum() > 0 else 1.0
          runs_pct = runs.astype(float) / total_runs * 100.0
    
          xticks_base = ['Wide Out Off', 'Outside Off', 'On Stumps', 'Down Leg', 'Wide Down Leg']
          xticks = xticks_base[::-1] if is_lhb else xticks_base
    
          n_rows = grids['n_rows']
          if n_rows >= 6:
              yticklabels = ['Short', 'Back of Length', 'Good', 'Full', 'Yorker', 'Full Toss'][:n_rows]
          else:
              yticklabels = ['Short', 'Back of Length', 'Good', 'Full', 'Yorker'][:n_rows]
    
          fig, axes = plt.subplots(3, 2, figsize=(14, 18))
          plt.suptitle(f"{title_prefix}", fontsize=16, weight='bold')
    
          plot_list = [
              (perc, '% of balls (heat)', 'Blues'),
              (bound_pct, 'Boundary %', 'OrRd'),
              (dot_pct, 'Dot %', 'Blues'),
              (sr, 'SR (runs/100 balls)', 'Reds'),
              (ctrl, 'False Shot % (not in control)', 'PuBu'),
              (runs_pct, 'Runs Scored %', 'Reds')
          ]
    
          for ax_idx, (ax, (arr, ttl, cmap)) in enumerate(zip(axes.flat, plot_list)):
              safe_arr = np.nan_to_num(arr.astype(float), nan=0.0)
              flat = safe_arr.flatten()
              if np.all(flat == 0):
                  vmin, vmax = 0, 1
              else:
                  vmin = float(np.nanmin(flat))
                  vmax = float(np.nanpercentile(flat, 95))
                  if vmax <= vmin:
                      vmax = vmin + 1.0
    
              im = ax.imshow(safe_arr, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
              ax.set_title(ttl)
              ax.set_xticks(range(grids['n_cols'])); ax.set_yticks(range(grids['n_rows']))
              ax.set_xticklabels(xticks, rotation=45, ha='right')
              ax.set_yticklabels(yticklabels)
    
              ax.set_xticks(np.arange(-0.5, grids['n_cols'], 1), minor=True)
              ax.set_yticks(np.arange(-0.5, grids['n_rows'], 1), minor=True)
              ax.grid(which='minor', color='black', linewidth=0.6, alpha=0.95)
              ax.tick_params(which='minor', bottom=False, left=False)
    
              if ax_idx == 0:
                  for i in range(grids['n_rows']):
                      for j in range(grids['n_cols']):
                          w_count = int(wkt[i, j])
                          if w_count > 0:
                              w_text = f"{w_count} W" if w_count > 1 else 'W'
                              ax.text(j, i, w_text, ha='center', va='center', fontsize=14, color='gold', weight='bold',
                                      bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
    
              fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    
          plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
          safe_fn = globals().get('safe_st_pyplot', None)
          try:
              if callable(safe_fn):
                  safe_fn(fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
              else:
                  st.pyplot(fig)
          except Exception:
              st.pyplot(fig)
          finally:
              plt.close(fig) 
      # def display_pitchmaps_from_df(df_src, title_prefix):
      #     if df_src is None or df_src.empty:
      #         st.info(f"No deliveries to show for {title_prefix}")
      #         return
     
      #     grids = build_pitch_grids(df_src)
     
      #     bh_col_name = globals().get('bat_hand_col', 'bat_hand')
      #     is_lhb = False
      #     if bh_col_name in df_src.columns:
      #         hands = df_src[bh_col_name].dropna().astype(str).str.strip().unique()
      #         if any(h.upper().startswith('L') for h in hands):
      #             is_lhb = True
     
      #     def maybe_flip(arr):
      #         return np.fliplr(arr) if is_lhb else arr.copy()
     
      #     count = maybe_flip(grids['count'])
      #     bounds = maybe_flip(grids['bounds'])
      #     dots = maybe_flip(grids['dots'])
      #     sr = maybe_flip(grids['sr'])
      #     ctrl = maybe_flip(grids['ctrl_pct'])
      #     wkt = maybe_flip(grids['wkt'])
      #     runs = maybe_flip(grids['runs'])
     
      #     total = count.sum() if count.sum() > 0 else 1.0
      #     perc = count.astype(float) / total * 100.0
     
      #     xticks_base = ['Wide Out Off', 'Outside Off', 'On Stumps', 'Down Leg', 'Wide Down Leg']
      #     xticks = xticks_base[::-1] if is_lhb else xticks_base
     
      #     n_rows = grids['n_rows']
      #     if n_rows >= 6:
      #         yticklabels = ['Short', 'Back of Length', 'Good', 'Full', 'Yorker', 'Full Toss'][:n_rows]
      #     else:
      #         yticklabels = ['Short', 'Back of Length', 'Good', 'Full', 'Yorker'][:n_rows]
     
      #     fig, axes = plt.subplots(3, 2, figsize=(14, 18))
      #     plt.suptitle(f"{title_prefix}", fontsize=16, weight='bold')
     
      #     plot_list = [
      #         (perc, '% of balls (heat)', 'Blues'),
      #         (bounds, 'Boundaries (count)', 'OrRd'),
      #         (dots, 'Dot balls (count)', 'Blues'),
      #         (sr, 'SR (runs/100 balls)', 'Reds'),
      #         (ctrl, 'False Shot % (not in control)', 'PuBu'),
      #         (runs, 'Runs (sum)', 'Reds')
      #     ]
     
      #     for ax_idx, (ax, (arr, ttl, cmap)) in enumerate(zip(axes.flat, plot_list)):
      #         safe_arr = np.nan_to_num(arr.astype(float), nan=0.0)
      #         flat = safe_arr.flatten()
      #         if np.all(flat == 0):
      #             vmin, vmax = 0, 1
      #         else:
      #             vmin = float(np.nanmin(flat))
      #             vmax = float(np.nanpercentile(flat, 95))
      #             if vmax <= vmin:
      #                 vmax = vmin + 1.0
     
      #         im = ax.imshow(safe_arr, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
      #         ax.set_title(ttl)
      #         ax.set_xticks(range(grids['n_cols'])); ax.set_yticks(range(grids['n_rows']))
      #         ax.set_xticklabels(xticks, rotation=45, ha='right')
      #         ax.set_yticklabels(yticklabels)
     
      #         ax.set_xticks(np.arange(-0.5, grids['n_cols'], 1), minor=True)
      #         ax.set_yticks(np.arange(-0.5, grids['n_rows'], 1), minor=True)
      #         ax.grid(which='minor', color='black', linewidth=0.6, alpha=0.95)
      #         ax.tick_params(which='minor', bottom=False, left=False)
     
      #         if ax_idx == 0:
      #             for i in range(grids['n_rows']):
      #                 for j in range(grids['n_cols']):
      #                     w_count = int(wkt[i, j])
      #                     if w_count > 0:
      #                         w_text = f"{w_count} W" if w_count > 1 else 'W'
      #                         ax.text(j, i, w_text, ha='center', va='center', fontsize=14, color='gold', weight='bold',
      #                                 bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
     
      #         fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
     
      #     plt.tight_layout(rect=[0, 0.03, 1, 0.97])
     
      #     safe_fn = globals().get('safe_st_pyplot', None)
      #     try:
      #         if callable(safe_fn):
      #             safe_fn(fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
      #         else:
      #             st.pyplot(fig)
      #     except Exception:
      #         st.pyplot(fig)
      #     finally:
      #         plt.close(fig)
   
    # ---------- attempt to draw wagon chart using your existing function ----------
    def draw_wagon_if_available(df_wagon, batter_name):
        if 'draw_cricket_field_with_run_totals_requested' in globals() and callable(globals()['draw_cricket_field_with_run_totals_requested']):
            try:
                fig_w = draw_cricket_field_with_run_totals_requested(df_wagon, batter_name)
                safe_fn = globals().get('safe_st_pyplot', None)
                if callable(safe_fn):
                    safe_fn(fig_w, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
                else:
                    st.pyplot(fig_w)
            except Exception as e:
                st.error(f"Wagon drawing function exists but raised: {e}")
        else:
            st.warning("Wagon chart function not found; please ensure `draw_cricket_field_with_run_totals_requested` is defined earlier.")
   
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
        st.image(img_resized, use_container_width=False, width=new_w)
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
    def safe_get_col(df_local, candidates, default=None):
        """
        Safely return the first matching column from candidates.
        If none are found, return default.
        """
        for c in candidates:
            if c in df_local.columns:
                return c
        return default
    bdf = as_dataframe(df)
    # Detect column names in your data
    batter_col = safe_get_col(bdf, ['bat', 'batsman'], default=None)
    bowler_col = safe_get_col(bdf, ['bowl', 'bowler'], default=None)
    match_col = safe_get_col(bdf, ['p_match', 'match_id'], default=None)
    year_col = safe_get_col(bdf, ['season', 'year'], default=None)
    inning_col = safe_get_col(bdf, ['inns', 'inning'], default=None)
    venue_col = safe_get_col(bdf, ['ground', 'venue', 'stadium', 'ground_name'], default=None)
    dismissal_col = safe_get_col(bdf, ['dismissal', 'Dismissal'], default=None)
    if batter_col is None or bowler_col is None:
        st.error("Dataframe must contain batter and bowler columns (e.g. 'bat' and 'bowl').")
        st.stop()
    if dismissal_col is None :
        st.error("Dataframe must contain dismissal column.")
        st.stop()
    # st.write(bdf.dismissal.unique())
    # Build unique player lists (filter out nulls and '0')
    unique_batters = sorted([x for x in pd.unique(bdf[batter_col].dropna()) if str(x).strip() not in ("", "0")])
    unique_bowlers = sorted([x for x in pd.unique(bdf[bowler_col].dropna()) if str(x).strip() not in ("", "0")])
    if not unique_batters or not unique_bowlers:
        st.warning("No batters or bowlers found in the dataset.")
        st.stop()
    # Player selectors
    batter_name = st.selectbox("Select a Batter", unique_batters, index=0)
    bowler_name = st.selectbox("Select a Bowler", unique_bowlers, index=0)
    # Grouping option
    grouping_option = st.selectbox("Group By", ["Year", "Match", "Venue", "Inning"])
    # PHASE SELECTBOX (for filtering the matchup)
    phase_opts = ['Overall', 'Powerplay', 'Middle 1', 'Middle 2', 'Death']
    chosen_phase = st.selectbox("Phase", options=phase_opts, index=0) # Default Overall
    # Raw matchup rows for download/sanity
    matchup_df = bdf[(bdf[batter_col] == batter_name) & (bdf[bowler_col] == bowler_name)].copy()
    # st.write(matchup_df.dismissal.unique())
    if chosen_phase != 'Overall':
        # Ensure PHASE column exists (derive from 'over' if missing)
        if 'PHASE' not in matchup_df.columns:
            if 'over' in matchup_df.columns:
                matchup_df['PHASE'] = matchup_df['over'].apply(assign_phase)
            else:
                matchup_df['PHASE'] = 'Unknown'  # Fallback if no 'over'
        
        # Now filter safely
        matchup_df = matchup_df[matchup_df['PHASE'] == chosen_phase].copy()
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
        csv = matchup_df.to_csv(index=False)
        # st.write(matchup_df.columns)
        st.download_button(
            label="Download raw matchup rows (CSV)",
            data=csv,
            file_name=f"{batter_name}_vs_{bowler_name}_matchup.csv",
            mime="text/csv"
        )
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
                if any(keyword in col_lower for keyword in ['innings', 'inning', 'runs', 'balls', 'wickets', 'wkts', 'matches', 'fours', 'sixes', 'dots', 'matches']):
                    df[col] = df[col].fillna(0).astype(int)
          
            # Round all other numeric columns to 2 decimals
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            for nc in numeric_cols:
                # Skip if already integer type
                if df[nc].dtype == int:
                    continue
                df[nc] = df[nc].round(2)
          
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
                return None
          
            # Concatenate all formatted dataframes
            out = pd.concat(df_list, ignore_index=True)
            # Remove batter and bowler columns if they exist
            cols_to_drop = []
            for col in out.columns:
                col_lower = str(col).lower()
                if any(x in col_lower for x in ['bat', 'bowl', 'batsman', 'bowler']):
                    cols_to_drop.append(col)
          
            out = out.drop(columns=cols_to_drop, errors='ignore')
          
            # Replace None/NaN with '-'
            out = out.fillna('-')
          
            # Capitalize first letter of each column name
            out.columns = [str(col).strip().capitalize() for col in out.columns]
          
            # Ensure primary column name is also capitalized
            primary_col_name_norm = str(primary_col_name).strip().capitalize()
          
            st.markdown(f"### {title}")
            st.dataframe(out, use_container_width=True)
            return out
        # Cumulator function (assuming based on typical stats; adjust as needed)
        def cumulator(temp_df):
            if temp_df.empty:
                return pd.DataFrame()
            runs_col = safe_get_col(temp_df, ['batruns', 'batsman_runs', 'score', 'runs'])
            if runs_col is None:
                return pd.DataFrame()
            balls = len(temp_df)
            runs = int(temp_df[runs_col].sum())
            fours = int((temp_df[runs_col] == 4).sum())
            sixes = int((temp_df[runs_col] == 6).sum())
            # Define the dismissal types that count as wickets
            wkt_types = {"caught", "bowled", "stumped", "lbw"}
            # st.write(temp_df.columns)
            # st.write(temp_df.dismissal.unique())
            # Calculate wickets
            wkts = temp_df['dismissal'].isin(wkt_types).sum() if 'dismissal' in temp_df.columns else 0
            
            # print(f"Total wickets: {wkts}")
            avg = runs / wkts if wkts > 0 else np.inf
            sr = (runs / balls * 100) if balls > 0 else 0.0
            summary = pd.DataFrame({
                'balls': [balls],
                'runs': [runs],
                'wickets': [wkts],
                'avg': [avg],
                'sr': [sr],
                'fours': [fours],
                'sixes': [sixes]
            })
            return summary
        # -------------------
        # Year grouping
        # -------------------
        if grouping_option == "Year":
            if year_col is None:
                st.info("Year/season column not detected in dataset.")
            else:
                tdf = matchup_df.copy()
                # st.write(tdf.dismissal.unique())
                seasons = sorted(tdf[year_col].dropna().unique().tolist())
                all_seasons = []
                for s in seasons:
                    temp = tdf[tdf[year_col] == s].copy()
                    if temp.empty:
                        continue
                    temp_summary = cumulator(temp)
                    if temp_summary.empty:
                        continue
                    # FORMAT BEFORE ADDING TO LIST
                    temp_summary = format_summary_df(temp_summary)
                    temp_summary.insert(0, 'year', s)
                    all_seasons.append(temp_summary)
                out = finalize_and_show(all_seasons, 'year', "Yearwise Performance", header_color="#efe6ff")
                if out is not None:
                    # Add Batter and Bowler columns (in uppercase)
                    out['Batsman'] = batter_name
                    out['Bowler'] = bowler_name
                  
                    # Move them to the front if you want them as leading columns
                    cols = ['Batsman', 'Bowler'] + [c for c in out.columns if c not in ['Batsman', 'Bowler']]
                    out = out[cols]
                    csv = out.to_csv(index=False)
                    st.download_button(
                        label="Download yearwise summary (CSV)",
                        data=csv,
                        file_name=f"{batter_name}_vs_{bowler_name}_yearwise.csv",
                        mime="text/csv"
                    )
                # Visualize selected year
                if all_seasons:
                    seasons_list = ['Overall'] + [str(s) for s in seasons]
                    selected_year = st.selectbox("Select Year to Visualize Wagon and Pitchmaps", seasons_list, index=0)
                    if selected_year == 'Overall':
                        selected_df = tdf.copy()
                    else:
                        selected_df = tdf[tdf[year_col] == int(selected_year)].copy()
                    if not selected_df.empty:
                        st.markdown(f"### Wagon and Pitchmaps for {selected_year}")
                        # Call wagon and pitchmaps (with safety checks)
                        try:
                            draw_wagon_if_available(selected_df, f"{batter_name} vs {bowler_name} - {selected_year}")
                            display_pitchmaps_from_df(selected_df, f"{batter_name} vs {bowler_name} - {selected_year}")
                        except NameError as ne:
                            st.warning(f"Visualization functions not available: {ne}. Define draw_wagon_if_available and display_pitchmaps_from_df earlier in the script.")
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
                    if temp_summary.empty:
                        continue
                    # FORMAT BEFORE ADDING TO LIST
                    temp_summary = format_summary_df(temp_summary)
                    temp_summary.insert(0, 'match_id', m)
                    all_matches.append(temp_summary)
                out = finalize_and_show(all_matches, 'match_id', "Matchwise Performance", header_color="#efe6ff")
                if out is not None:
                    # Add Batter and Bowler columns (in uppercase)
                    out['Batsman'] = batter_name
                    out['Bowler'] = bowler_name
                  
                    # Move them to the front if you want them as leading columns
                    cols = ['Batsman', 'Bowler'] + [c for c in out.columns if c not in ['Batsman', 'Bowler']]
                    out = out[cols]
                    csv = out.to_csv(index=False)
                    st.download_button(
                        label="Download matchwise summary (CSV)",
                        data=csv,
                        file_name=f"{batter_name}_vs_{bowler_name}_matchwise.csv",
                        mime="text/csv"
                    )
                # Visualize selected match
                if all_matches:
                    match_list = [str(m) for m in match_ids]
                    selected_match = st.selectbox("Select Match to Visualize Wagon and Pitchmaps", match_list, index=0)
                    selected_df = tdf[tdf[match_col] == int(selected_match)].copy()
                    if not selected_df.empty:
                        st.markdown(f"### Wagon and Pitchmaps for Match {selected_match}")
                        try:
                            draw_wagon_if_available(selected_df, f"{batter_name} vs {bowler_name} - Match {selected_match}")
                            display_pitchmaps_from_df(selected_df, f"{batter_name} vs {bowler_name} - Match {selected_match}")
                        except NameError as ne:
                            st.warning(f"Visualization functions not available: {ne}. Define draw_wagon_if_available and display_pitchmaps_from_df earlier in the script.")
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
                    if temp_summary.empty:
                        continue
                    # FORMAT BEFORE ADDING TO LIST
                    temp_summary = format_summary_df(temp_summary)
                    temp_summary.insert(0, 'venue', v)
                    all_venues.append(temp_summary)
                out = finalize_and_show(all_venues, 'venue', "Venuewise Performance", header_color="#efe6ff")
                if out is not None:
                    # Add Batter and Bowler columns (in uppercase)
                    out['Batsman'] = batter_name
                    out['Bowler'] = bowler_name
                  
                    # Move them to the front if you want them as leading columns
                    cols = ['Batsman', 'Bowler'] + [c for c in out.columns if c not in ['Batsman', 'Bowler']]
                    out = out[cols]
                    csv = out.to_csv(index=False)
                    st.download_button(
                        label="Download venuewise summary (CSV)",
                        data=csv,
                        file_name=f"{batter_name}_vs_{bowler_name}_venuewise.csv",
                        mime="text/csv"
                    )
                # Visualize selected venue
                if all_venues:
                    venue_list = [str(v) for v in venues]
                    selected_venue = st.selectbox("Select Venue to Visualize Wagon and Pitchmaps", venue_list, index=0)
                    selected_df = tdf[tdf[venue_col] == selected_venue].copy()
                    if not selected_df.empty:
                        st.markdown(f"### Wagon and Pitchmaps for Venue {selected_venue}")
                        try:
                            draw_wagon_if_available(selected_df, f"{batter_name} vs {bowler_name} - Venue {selected_venue}")
                            display_pitchmaps_from_df(selected_df, f"{batter_name} vs {bowler_name} - Venue {selected_venue}")
                        except NameError as ne:
                            st.warning(f"Visualization functions not available: {ne}. Define draw_wagon_if_available and display_pitchmaps_from_df earlier in the script.")
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
                    if temp_summary.empty:
                        continue
                    # FORMAT BEFORE ADDING TO LIST
                    temp_summary = format_summary_df(temp_summary)
                    temp_summary.insert(0, 'inning', inn)
                    all_inns.append(temp_summary)
                out = finalize_and_show(all_inns, 'inning', "Inningwise Performance", header_color="#efe6ff")
                if out is not None:
                    # Add Batter and Bowler columns (in uppercase)
                    out['Batsman'] = batter_name
                    out['Bowler'] = bowler_name
                  
                    # Move them to the front if you want them as leading columns
                    cols = ['Batsman', 'Bowler'] + [c for c in out.columns if c not in ['Batsman', 'Bowler']]
                    out = out[cols]
                    csv = out.to_csv(index=False)
                    st.download_button(
                        label="Download inningwise summary (CSV)",
                        data=csv,
                        file_name=f"{batter_name}_vs_{bowler_name}_inningwise.csv",
                        mime="text/csv"
                    )
                # Visualize selected inning
                if all_inns:
                    inning_list = [str(inn) for inn in innings]
                    selected_inning = st.selectbox("Select Inning to Visualize Wagon and Pitchmaps", inning_list, index=0)
                    selected_df = tdf[tdf[inning_col] == int(selected_inning)].copy()
                    if not selected_df.empty:
                        st.markdown(f"### Wagon and Pitchmaps for Inning {selected_inning}")
                        try:
                            draw_wagon_if_available(selected_df, f"{batter_name} vs {bowler_name} - Inning {selected_inning}")
                            display_pitchmaps_from_df(selected_df, f"{batter_name} vs {bowler_name} - Inning {selected_inning}")
                        except NameError as ne:
                            st.warning(f"Visualization functions not available: {ne}. Define draw_wagon_if_available and display_pitchmaps_from_df earlier in the script.")
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
        Uses use_container_width param (no deprecated use_container_width).
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
        'YORKER': 4,
        'FULLTOSS': 5
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
            # sector_names = {
            #     1: "Fine Leg",
            #     2: "Square Leg",
            #     3: "Mid Wicket",
            #     4: "Mid On",
            #     5: "Mid Off",
            #     6: "Covers",
            #     7: "Point",
            #     8: "Third Man"
            # }
    
            # # Base angles for RHB per your clock instruction: Third Man centered at 11:15 (112.5°)
            # # and then each sector moves 45° toward the left (counter-clockwise)
            # base_angles = {
            #     8: 112.5,  # Third Man
            #     7: 157.5,  # Point
            #     6: 202.5,  # Covers
            #     5: 247.5,  # Mid Off
            #     4: 292.5,  # Mid On
            #     3: 337.5,  # Mid Wicket
            #     2: 22.5,   # Square Leg
            #     1: 67.5    # Fine Leg
            # }
    
            # def get_sector_angle_requested(zone, batting_style):
            #     """Return sector center in radians.
    
            #     IMPORTANT: For left-handers we must *mirror* the chart across vertical axis so that
            #     Third Man <-> Fine Leg, Point <-> Square Leg, Covers <-> Mid Wicket, Mid Off <-> Mid On.
            #     This is achieved by reflecting the angle across the vertical axis: angle -> (180 - angle) % 360.
            #     """
            #     angle = float(base_angles.get(int(zone), 0.0))
            #     if str(batting_style).strip().upper().startswith('L'):
            #         angle = (180.0 - angle) % 360.0
            #     return math.radians(angle)
    
            # def draw_cricket_field_with_run_totals_requested(final_df_local, batsman_name):
            #     fig, ax = plt.subplots(figsize=(10, 10))
            #     ax.set_aspect('equal')
            #     ax.axis('off')
    
            #     # Field base
            #     ax.add_patch(Circle((0, 0), 1, fill=True, color='#228B22', alpha=1))
            #     ax.add_patch(Circle((0, 0), 1, fill=False, color='black', linewidth=3))
            #     ax.add_patch(Circle((0, 0), 0.5, fill=True, color='#66bb6a'))
            #     ax.add_patch(Circle((0, 0), 0.5, fill=False, color='white', linewidth=1))
    
            #     # pitch rectangle approx
            #     pitch_rect = Rectangle((-0.04, -0.08), 0.08, 0.16, color='tan', alpha=1, zorder=8)
            #     ax.add_patch(pitch_rect)
    
            #     # radial sector lines (8)
            #     angles = np.linspace(0, 2*np.pi, 9)[:-1]
            #     for angle in angles:
            #         x = math.cos(angle)
            #         y = math.sin(angle)
            #         ax.plot([0, x], [0, y], color='white', alpha=0.25, linewidth=1)
    
            #     # prepare data (runs, fours, sixes by sector)
            #     tmp = final_df_local.copy()
            #     if wagon_zone_col in tmp.columns:
            #         tmp['wagon_zone_int'] = pd.to_numeric(tmp[wagon_zone_col], errors='coerce').astype('Int64')
            #     else:
            #         tmp['wagon_zone_int'] = pd.Series(dtype='Int64')
    
            #     runs_by_zone = tmp.groupby('wagon_zone_int')[run_col].sum().to_dict()
            #     fours_by_zone = tmp.groupby('wagon_zone_int')[run_col].apply(lambda s: int((s==4).sum())).to_dict()
            #     sixes_by_zone = tmp.groupby('wagon_zone_int')[run_col].apply(lambda s: int((s==6).sum())).to_dict()
            #     total_runs_in_wagon = sum(int(v) for v in runs_by_zone.values())
    
            #     # Title
            #     title_text = f"{batsman_name}'s Scoring Zones"
            #     plt.title(title_text, pad=20, color='white', size=14, fontweight='bold')
    
            #     # Place % runs and runs in each sector using sector centers
            #     for zone in range(1, 9):
            #         batting_style_val = None
            #         if bat_hand_col in tmp.columns and not tmp[bat_hand_col].dropna().empty:
            #             batting_style_val = tmp[bat_hand_col].dropna().iloc[0]
            #         angle_mid = get_sector_angle_requested(zone, batting_style_val)
            #         x = 0.60 * math.cos(angle_mid)
            #         y = 0.60 * math.sin(angle_mid)
            #         runs = int(runs_by_zone.get(zone, 0))
            #         pct = (runs / total_runs_in_wagon * 100) if total_runs_in_wagon > 0 else 0.0
            #         pct_str = f"{pct:.2f}%"
    
            #         # main labels
            #         ax.text(x, y+0.03, pct_str, ha='center', va='center', color='white', fontweight='bold', fontsize=18)
            #         ax.text(x, y-0.03, f"{runs} runs", ha='center', va='center', color='white', fontsize=10)
    
            #         # fours & sixes below
            #         fours = int(fours_by_zone.get(zone, 0))
            #         sixes = int(sixes_by_zone.get(zone, 0))
            #         ax.text(x, y-0.12, f"4s: {fours}  6s: {sixes}", ha='center', va='center', color='white', fontsize=9)
    
            #         # sector name slightly farther out
            #         sx = 0.80 * math.cos(angle_mid)
            #         sy = 0.80 * math.sin(angle_mid)
            #         ax.text(sx, sy, sector_names.get(zone, f"Sector {zone}"), ha='center', va='center', color='white', fontsize=8)
    
            #     ax.set_xlim(-1.2, 1.2)
            #     ax.set_ylim(-1.2, 1.2)
            #     plt.tight_layout(pad=0)
            #     return fig
    
            # # run the wagon wheel drawing if column exists
            # if wagon_zone_col not in final_df.columns:
            #     st.info("wagonZone column not available for wagon wheel.")
            # else:
            #     ww_df = final_df.copy()
            #     ww_df['wagon_zone_int'] = pd.to_numeric(ww_df[wagon_zone_col], errors='coerce').astype('Int64')
            #     grouped = ww_df.groupby('wagon_zone_int').agg(
            #         runs = (run_col, 'sum'),
            #         balls = (run_col, 'size'),
            #         fours = (run_col, lambda s: int((s==4).sum())),
            #         sixes = (run_col, lambda s: int((s==6).sum()))
            #     ).reset_index().rename(columns={'wagon_zone_int':'sector'})
            #     all_sectors = pd.DataFrame({"sector": list(range(1,9))})
            #     grouped = all_sectors.merge(grouped, on='sector', how='left').fillna(0)
            #     grouped[['runs','balls','fours','sixes']] = grouped[['runs','balls','fours','sixes']].astype(int)
            #     total_runs = grouped['runs'].sum()
            #     grouped['pct_runs'] = grouped['runs'].apply(lambda x: round((x / total_runs * 100) if total_runs>0 else 0.0,2))
    
            #     ww_display = grouped.copy()
            #     # ww_display['Sector Name'] = ww_display['sector'].map({
            #     #     1:"Third Man",2:"Point",3:"Covers",4:"Mid Off",5:"Mid On",6:"Mid Wicket",7:"Square Leg",8:"Fine Leg"
            #     # })
            #     ww_display['Sector Name'] = ww_display['sector'].map({
            #         1: "Fine Leg",
            #         2: "Square Leg",
            #         3: "Mid Wicket",
            #         4: "Mid On",
            #         5: "Mid Off",
            #         6: "Covers",
            #         7: "Point",
            #         8: "Third Man"
            #     })
            #     ww_display = ww_display[['sector','Sector Name','runs','pct_runs','fours','sixes','balls']].rename(columns={
            #         'sector':'Sector','runs':'Runs','pct_runs':'Pct of Runs','fours':'4s','sixes':'6s','balls':'Balls'
            #     })
            #     ww_display['Pct of Runs'] = ww_display['Pct of Runs'].apply(lambda x: f"{x:.2f}%")
    
            #     # draw figure using requested style and mapping
            #     fig = draw_cricket_field_with_run_totals_requested(final_df, batsman_selected)
            #     safe_st_pyplot(fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
    
            #     # Label below wagon chart indicating RHB/LHB
            #     batting_style_display = None
            #     if bat_hand_col in final_df.columns and not final_df[bat_hand_col].dropna().empty:
            #         batting_style_display = final_df[bat_hand_col].dropna().iloc[0]
            #     side_label = "LHB" if batting_style_display and str(batting_style_display).strip().upper().startswith('L') else "RHB"
            #     st.markdown(f"<div style='text-align:center; margin-top:6px;'><strong>{batsman_selected}'s Wagon Chart ({side_label})</strong></div>", unsafe_allow_html=True)

            # Full wagon-wheel / sector-summary code (self-contained)

            # --- wagon wheel with correct LHB mirroring and sector name reversal ---
            # import math
            # import numpy as np
            # import pandas as pd
            # import matplotlib.pyplot as plt
            # from matplotlib.patches import Circle, Rectangle
            
            # # If using Streamlit front-end (your original code used st.*), import it.
            # # If not running inside Streamlit this import will still work if streamlit is installed.
            # try:
            #     import streamlit as st
            # except Exception:
            #     st = None
            
            # # -------------------------
            # # Sector names for RHB and LHB (reversed)
            # # -------------------------
            # SECTOR_NAMES_RHB = {
            #     1: "Fine Leg",
            #     2: "Square Leg",
            #     3: "Mid Wicket",
            #     4: "Mid On",
            #     5: "Mid Off",
            #     6: "Covers",
            #     7: "Point",
            #     8: "Third Man"
            # }
            
            # SECTOR_NAMES_LHB = {
            #     1: "Third Man",
            #     2: "Point",
            #     3: "Covers",
            #     4: "Mid Off",
            #     5: "Mid On",
            #     6: "Mid Wicket",
            #     7: "Square Leg",
            #     8: "Fine Leg"
            # }
            
            # # Base angles for RHB per your clock instruction (zone centers in degrees)
            # BASE_ANGLES = {
            #     8: 112.5,  # Third Man
            #     7: 157.5,  # Point
            #     6: 202.5,  # Covers
            #     5: 247.5,  # Mid Off
            #     4: 292.5,  # Mid On
            #     3: 337.5,  # Mid Wicket
            #     2: 22.5,   # Square Leg
            #     1: 67.5    # Fine Leg
            # }
            
            # def get_sector_angle_requested(zone, batting_style):
            #     """
            #     Return sector center in radians.
            #     For left-handers mirror across vertical axis: angle -> (180 - angle) % 360
            #     """
            #     angle = float(BASE_ANGLES.get(int(zone), 0.0))
            #     if isinstance(batting_style, str) and batting_style.strip().upper().startswith('L'):
            #         angle = (180.0 - angle) % 360.0
            #     return math.radians(angle)
            
            
            # def draw_cricket_field_with_run_totals_requested(final_df_local,
            #                                                 batsman_name,
            #                                                 wagon_zone_col='wagonZone',
            #                                                 run_col='score',
            #                                                 bat_hand_col='bat_hand',
            #                                                 figsize=(10,10)):
            #     """
            #     Draw wagon wheel / scoring zones for a batsman.
            #     - final_df_local : full dataframe (deliveries)
            #     - batsman_name : string (used only for title)
            #     - wagon_zone_col : column containing zone indices 1..8
            #     - run_col : column containing runs scored off the ball
            #     - bat_hand_col : column containing 'RHB'/'LHB' if available
            #     Returns matplotlib Figure.
            #     """
            
            #     fig, ax = plt.subplots(figsize=figsize)
            #     ax.set_aspect('equal')
            #     ax.axis('off')
            
            #     # Field base
            #     ax.add_patch(Circle((0, 0), 1, fill=True, color='#228B22', alpha=1, zorder=0))
            #     ax.add_patch(Circle((0, 0), 1, fill=False, color='black', linewidth=3, zorder=1))
            #     ax.add_patch(Circle((0, 0), 0.5, fill=True, color='#66bb6a', zorder=2))
            #     ax.add_patch(Circle((0, 0), 0.5, fill=False, color='white', linewidth=1, zorder=3))
            
            #     # pitch rectangle approx
            #     pitch_rect = Rectangle((-0.04, -0.08), 0.08, 0.16, color='tan', alpha=1, zorder=8)
            #     ax.add_patch(pitch_rect)
            
            #     # radial sector lines (8)
            #     angles = np.linspace(0, 2*np.pi, 9)[:-1]
            #     for angle in angles:
            #         x = math.cos(angle)
            #         y = math.sin(angle)
            #         ax.plot([0, x], [0, y], color='white', alpha=0.25, linewidth=1, zorder=4)
            
            #     # prepare data
            #     tmp = final_df_local.copy()
            
            #     if wagon_zone_col in tmp.columns:
            #         tmp['wagon_zone_int'] = pd.to_numeric(tmp[wagon_zone_col], errors='coerce').astype('Int64')
            #     else:
            #         tmp['wagon_zone_int'] = pd.Series(dtype='Int64')
            
            #     # Determine batting hand (prefer first non-null in this player's rows, else fallback)
            #     batting_style_val = None
            #     if bat_hand_col in tmp.columns and not tmp[bat_hand_col].dropna().empty:
            #         batting_style_val = tmp[bat_hand_col].dropna().iloc[0]
            #     # If still None, try check final_df_local top-level (might be global)
            #     if batting_style_val is None and bat_hand_col in final_df_local.columns and not final_df_local[bat_hand_col].dropna().empty:
            #         batting_style_val = final_df_local[bat_hand_col].dropna().iloc[0]
            
            #     # Choose sector name mapping based on hand
            #     if isinstance(batting_style_val, str) and batting_style_val.strip().upper().startswith('L'):
            #         sector_names = SECTOR_NAMES_LHB
            #         side_label = "LHB"
            #     else:
            #         sector_names = SECTOR_NAMES_RHB
            #         side_label = "RHB"
            
            #     # Aggregate runs, fours, sixes by zone (no remapping of zone numbers; we mirror geometry & names)
            #     runs_by_zone = tmp.groupby('wagon_zone_int')[run_col].sum().to_dict()
            #     # Count fours and sixes by raw run_col values
            #     fours_by_zone = tmp.groupby('wagon_zone_int')[run_col].apply(lambda s: int((s==4).sum())).to_dict()
            #     sixes_by_zone = tmp.groupby('wagon_zone_int')[run_col].apply(lambda s: int((s==6).sum())).to_dict()
            #     balls_by_zone = tmp.groupby('wagon_zone_int')[run_col].size().to_dict()
            
            #     total_runs_in_wagon = sum(int(v) for v in runs_by_zone.values())
            
            #     # Title
            #     title_text = f"{batsman_name}'s Scoring Zones"
            #     plt_title_color = 'white'
            #     fig.suptitle(title_text, fontsize=14, fontweight='bold', color=plt_title_color, y=0.95)
            
            #     # Place % runs and runs in each sector using sector centers
            #     for zone in range(1, 9):
            #         angle_mid = get_sector_angle_requested(zone, batting_style_val)
            #         x = 0.60 * math.cos(angle_mid)
            #         y = 0.60 * math.sin(angle_mid)
            
            #         runs = int(runs_by_zone.get(zone, 0))
            #         pct = (runs / total_runs_in_wagon * 100) if total_runs_in_wagon > 0 else 0.0
            #         pct_str = f"{pct:.2f}%"
            
            #         # main labels
            #         ax.text(x, y+0.03, pct_str, ha='center', va='center', color='white', fontweight='bold', fontsize=16, zorder=10)
            #         ax.text(x, y-0.03, f"{runs} runs", ha='center', va='center', color='white', fontsize=10, zorder=10)
            
            #         # fours & sixes below
            #         fours = int(fours_by_zone.get(zone, 0))
            #         sixes = int(sixes_by_zone.get(zone, 0))
            #         ax.text(x, y-0.12, f"4s: {fours}  6s: {sixes}", ha='center', va='center', color='white', fontsize=9, zorder=10)
            
            #         # sector name slightly farther out
            #         sx = 0.80 * math.cos(angle_mid)
            #         sy = 0.80 * math.sin(angle_mid)
            #         ax.text(sx, sy, sector_names.get(zone, f"Sector {zone}"), ha='center', va='center', color='white', fontsize=8, zorder=9)
            
            #     ax.set_xlim(-1.2, 1.2)
            #     ax.set_ylim(-1.2, 1.2)
            #     plt.tight_layout(pad=0)
            
            #     return fig, {
            #         'side_label': side_label,
            #         'runs_by_zone': runs_by_zone,
            #         'fours_by_zone': fours_by_zone,
            #         'sixes_by_zone': sixes_by_zone,
            #         'balls_by_zone': balls_by_zone,
            #         'total_runs': total_runs_in_wagon
            #     }
            
            
            # # -------------------------
            # # run the wagon wheel drawing if column exists (main integration block)
            # # -------------------------
            # # expected outer-scope variables used in your app:
            # # final_df, wagon_zone_col, run_col, bat_hand_col, batsman_selected
            # #
            # # If you're running this block stand-alone, set them here for a quick test:
            # # Example test (uncomment to use):
            # # import random
            # # random.seed(1)
            # # final_df = pd.DataFrame({
            # #     'batter': ['Player X']*200,
            # #     'wagonZone': [random.randint(1,8) for _ in range(200)],
            # #     'score': [random.choice([0,0,1,1,2,4,6]) for _ in range(200)],
            # #     'bat_hand': ['RHB']*200
            # # })
            # # batsman_selected = 'Player X'
            # # wagon_zone_col = 'wagonZone'
            # # run_col = 'score'
            # # bat_hand_col = 'bat_hand'
            
            # # Ensure variables exist (this will raise if you don't set them in your environment)
            # try:
            #     _ = final_df
            #     _ = batsman_selected
            #     _ = wagon_zone_col
            #     _ = run_col
            #     _ = bat_hand_col
            # except NameError:
            #     raise RuntimeError("You must set final_df, batsman_selected, wagon_zone_col, run_col and bat_hand_col in your environment before running this block.")
            
            # if wagon_zone_col not in final_df.columns:
            #     # If in Streamlit show an info. Otherwise print.
            #     if st:
            #         st.info(f"{wagon_zone_col} column not available for wagon wheel.")
            #     else:
            #         print(f"{wagon_zone_col} column not available for wagon wheel.")
            # else:
            #     ww_df = final_df.copy()
            #     ww_df['wagon_zone_int'] = pd.to_numeric(ww_df[wagon_zone_col], errors='coerce').astype('Int64')
            
            #     # aggregated table similar to your earlier code
            #     grouped = ww_df.groupby('wagon_zone_int').agg(
            #         runs = (run_col, 'sum'),
            #         balls = (run_col, 'size'),
            #         fours = (run_col, lambda s: int((s==4).sum())),
            #         sixes = (run_col, lambda s: int((s==6).sum()))
            #     ).reset_index().rename(columns={'wagon_zone_int':'sector'})
            
            #     all_sectors = pd.DataFrame({"sector": list(range(1,9))})
            #     grouped = all_sectors.merge(grouped, on='sector', how='left').fillna(0)
            #     grouped[['runs','balls','fours','sixes']] = grouped[['runs','balls','fours','sixes']].astype(int)
            #     total_runs = grouped['runs'].sum()
            #     grouped['pct_runs'] = grouped['runs'].apply(lambda x: round((x / total_runs * 100) if total_runs>0 else 0.0, 2))
            
            #     ww_display = grouped.copy()
            #     # Sector names for table — use RHB mapping (this is tabular; visual will show LHB mirrored)
            #     ww_display['Sector Name'] = ww_display['sector'].map({
            #         1: "Fine Leg",
            #         2: "Square Leg",
            #         3: "Mid Wicket",
            #         4: "Mid On",
            #         5: "Mid Off",
            #         6: "Covers",
            #         7: "Point",
            #         8: "Third Man"
            #     })
            #     ww_display = ww_display[['sector','Sector Name','runs','pct_runs','fours','sixes','balls']].rename(columns={
            #         'sector':'Sector','runs':'Runs','pct_runs':'Pct of Runs','fours':'4s','sixes':'6s','balls':'Balls'
            #     })
            #     ww_display['Pct of Runs'] = ww_display['Pct of Runs'].apply(lambda x: f"{x:.2f}%")
            
            #     # draw figure using the requested mapping and styles
            #     fig, metadata = draw_cricket_field_with_run_totals_requested(final_df, batsman_selected,
            #                                                                 wagon_zone_col=wagon_zone_col,
            #                                                                 run_col=run_col,
            #                                                                 bat_hand_col=bat_hand_col)
            
            #     # try safe_st_pyplot if available (your original code used it). Fallback to st.pyplot or plt.show
            #     try:
            #         # safe_st_pyplot may be defined in your app runtime; attempt to call it if present
            #         safe_st_pyplot  # check existence
            #     except NameError:
            #         safe_st_pyplot = None
            
            #     plotted = False
            #     if safe_st_pyplot:
            #         try:
            #             safe_st_pyplot(fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
            #             plotted = True
            #         except Exception:
            #             plotted = False
            
            #     if not plotted and st:
            #         try:
            #             st.pyplot(fig)
            #             plotted = True
            #         except Exception:
            #             plotted = False
            
            #     if not plotted:
            #         # notebook fallback
            #         plt.show()
            
            #     # Show table in Streamlit nicely if available, else print
            #     if st:
            #         st.dataframe(ww_display)
            #         # beneath wagon chart, show which hand
            #         batting_style_display = None
            #         if bat_hand_col in final_df.columns and not final_df[bat_hand_col].dropna().empty:
            #             batting_style_display = final_df[bat_hand_col].dropna().iloc[0]
            #         side_label = "LHB" if batting_style_display and str(batting_style_display).strip().upper().startswith('L') else "RHB"
            #         st.markdown(f"<div style='text-align:center; margin-top:6px;'><strong>{batsman_selected}'s Wagon Chart ({side_label})</strong></div>", unsafe_allow_html=True)
            #     else:
            #         # Print summary table to console
            #         print(ww_display.to_string(index=False))
            #         batting_style_display = None
            #         if bat_hand_col in final_df.columns and not final_df[bat_hand_col].dropna().empty:
            #             batting_style_display = final_df[bat_hand_col].dropna().iloc[0]
            #         side_label = "LHB" if batting_style_display and str(batting_style_display).strip().upper().startswith('L') else "RHB"
            #         print(f"\n{batsman_selected}'s Wagon Chart ({side_label})")

            #     batting_style_display = None
            #     if bat_hand_col in final_df.columns and not final_df[bat_hand_col].dropna().empty:
            #         batting_style_display = final_df[bat_hand_col].dropna().iloc[0]
            #     side_label = "LHB" if batting_style_display and str(batting_style_display).strip().upper().startswith('L') else "RHB"
            #     st.markdown(f"<div style='text-align:center; margin-top:6px;'><strong>{batsman_selected}'s Wagon Chart ({side_label})</strong></div>", unsafe_allow_html=True)

                # show table below wheel
                # st.dataframe(ww_display.style.set_table_styles([
                #     {"selector":"thead th", "props":[("background-color","#e6f2ff"),("font-weight","600")]},
                # ]), use_container_width=True)
            # --- Wagon wheel with vertical-axis mirroring for LHB (single change) ---
            # This is your original wagon-wheel block with exactly one enforced change:
            # if batter is LHB, the plotted wagon-wheel is mirrored about the vertical axis
            # passing through the pitch. No other logic or table mappings were altered.
            
            # --- Wagon wheel with LHB data-sector mirroring (fixed) ---
            # Full wagon-wheel / scoring-zones plotting code with the requested LHB angle swap.
            # Change made: For LHB we now use the reversed BASE_ANGLES mapping (zone -> mirrored angle)
            # so that Third Man <-> Fine Leg, Point <-> Square Leg, Covers <-> Mid Wicket, Mid Off <-> Mid On.
            # No other logic changed.
            
            import math
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle, Rectangle
            
            # Optional Streamlit support
            try:
                import streamlit as st
            except Exception:
                st = None
            
            # --- Sector name mapping (RHB canonical) ---
            SECTOR_NAMES_RHB = {
                1: "Fine Leg",
                2: "Square Leg",
                3: "Mid Wicket",
                4: "Mid On",
                5: "Mid Off",
                6: "Covers",
                7: "Point",
                8: "Third Man"
            }
            
            # --- BASE_ANGLES for the canonical RHB layout (degrees) ---
            BASE_ANGLES = {
                8: 112.5,  # Third Man
                7: 157.5,  # Point
                6: 202.5,  # Covers
                5: 247.5,  # Mid Off
                4: 292.5,  # Mid On
                3: 337.5,  # Mid Wicket
                2: 22.5,   # Square Leg
                1: 67.5    # Fine Leg
            }
            
            # --- EXPLICIT LHB mapping: reverse pairs (1<->8, 2<->7, 3<->6, 4<->5) ---
            # This is exactly the swap you asked for (ThirdMan<->FineLeg, SquareLeg<->Point, MidWicket<->Covers, MidOn<->MidOff)
            BASE_ANGLES_LHB = { z: float(BASE_ANGLES.get(9 - z, 0.0)) for z in range(1,9) }
            
            def get_sector_angle_requested(zone, batting_style):
                """Return sector center in radians.
            
                For RHB: use BASE_ANGLES[zone].
                For LHB: use BASE_ANGLES_LHB[zone] (explicit reversed mapping).
                """
                z = int(zone)
                is_lhb = isinstance(batting_style, str) and batting_style.strip().upper().startswith('L')
                if not is_lhb:
                    angle_deg = float(BASE_ANGLES.get(z, 0.0))
                else:
                    angle_deg = float(BASE_ANGLES_LHB.get(z, 0.0))
                return math.radians(angle_deg)
            
            
            def draw_cricket_field_with_run_totals_requested(final_df_local, batsman_name,
                                                            wagon_zone_col='wagonZone',
                                                            run_col='score',
                                                            bat_hand_col='bat_hand'):
                """
                Draw wagon wheel / scoring zones with explicit LHB angle swap implemented via BASE_ANGLES_LHB.
                No other logic changed.
                """
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
            
                # raw aggregates keyed by the raw sector id (1..8)
                runs_by_zone = tmp.groupby('wagon_zone_int')[run_col].sum().to_dict()
                fours_by_zone = tmp.groupby('wagon_zone_int')[run_col].apply(lambda s: int((s==4).sum())).to_dict()
                sixes_by_zone = tmp.groupby('wagon_zone_int')[run_col].apply(lambda s: int((s==6).sum())).to_dict()
                total_runs_in_wagon = sum(int(v) for v in runs_by_zone.values())
            
                # Title
                title_text = f"{batsman_name}'s Scoring Zones"
                plt.title(title_text, pad=20, color='white', size=14, fontweight='bold')
            
                # Determine batter handedness (first non-null sample in the filtered rows)
                batting_style_val = None
                if bat_hand_col in tmp.columns and not tmp[bat_hand_col].dropna().empty:
                    batting_style_val = tmp[bat_hand_col].dropna().iloc[0]
                is_lhb = isinstance(batting_style_val, str) and batting_style_val.strip().upper().startswith('L')
            
                # Place % runs and runs in each sector using sector centers (angles chosen from explicit maps)
                for zone in range(1, 9):
                    angle_mid = get_sector_angle_requested(zone, batting_style_val)
                    x = 0.60 * math.cos(angle_mid)
                    y = 0.60 * math.sin(angle_mid)
            
                    # data_zone: use raw sector index (1..8) directly; we only changed the display angle mapping
                    # BUT to keep labels and data aligned visually for LHB we need to pick the data zone that corresponds
                    # to the displayed sector name. With BASE_ANGLES_LHB we've already swapped sector angles, so the display
                    # location uses the mirrored mapping. To ensure the numeric data (runs/fours/sixes) goes into the
                    # correct display sector we should still pull from the *raw* zone that the batter's wagon data used.
                    # The approach below keeps behaviour identical to prior logic: show data for the raw zone index.
                    data_zone = zone
            
                    runs = int(runs_by_zone.get(data_zone, 0))
                    pct = (runs / total_runs_in_wagon * 100) if total_runs_in_wagon > 0 else 0.0
                    pct_str = f"{pct:.2f}%"
            
                    # main labels
                    ax.text(x, y+0.03, pct_str, ha='center', va='center', color='white', fontweight='bold', fontsize=18)
                    ax.text(x, y-0.03, f"{runs} runs", ha='center', va='center', color='white', fontsize=10)
            
                    # fours & sixes below
                    fours = int(fours_by_zone.get(data_zone, 0))
                    sixes = int(sixes_by_zone.get(data_zone, 0))
                    ax.text(x, y-0.12, f"4s: {fours}  6s: {sixes}", ha='center', va='center', color='white', fontsize=9)
            
                    # sector name slightly farther out:
                    # For LHB we display the mirrored sector name so label and data match visually.
                    display_sector_idx = zone if not is_lhb else (9 - zone)
                    sector_name_to_show = SECTOR_NAMES_RHB.get(display_sector_idx, f"Sector {display_sector_idx}")
                    sx = 0.80 * math.cos(angle_mid)
                    sy = 0.80 * math.sin(angle_mid)
                    ax.text(sx, sy, sector_name_to_show, ha='center', va='center', color='white', fontsize=8)
            
                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.2, 1.2)
                plt.tight_layout(pad=0)
                if is_lhb:
                    ax.invert_xaxis()

                return fig
            
            
            # ----------------------------
            # Execution block (expects final_df, batsman_selected, wagon_zone_col, run_col, bat_hand_col defined)
            # ----------------------------
            
            try:
                _ = final_df
                _ = batsman_selected
                _ = wagon_zone_col
                _ = run_col
                _ = bat_hand_col
            except NameError:
                raise RuntimeError("You must set final_df, batsman_selected, wagon_zone_col, run_col and bat_hand_col before running this block.")
            
            if wagon_zone_col not in final_df.columns:
                if st:
                    st.info("wagonZone column not available for wagon wheel.")
                else:
                    print("wagonZone column not available for wagon wheel.")
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
            
                # If batter is LHB, present the display table with mirrored sector names so table aligns with chart
                batting_style_display = None
                if bat_hand_col in final_df.columns and not final_df[bat_hand_col].dropna().empty:
                    batting_style_display = final_df[bat_hand_col].dropna().iloc[0]
                is_lhb_table = isinstance(batting_style_display, str) and batting_style_display.strip().upper().startswith('L')
            
                if not is_lhb_table:
                    ww_display = grouped.copy()
                    ww_display['Sector Name'] = ww_display['sector'].map(SECTOR_NAMES_RHB)
                else:
                    mirrored_rows = []
                    for _, row in grouped.iterrows():
                        s = int(row['sector'])
                        display_idx = 9 - s
                        mirrored_rows.append({
                            'sector': s,
                            'Sector Name': SECTOR_NAMES_RHB.get(display_idx, f"Sector {display_idx}"),
                            'runs': int(row['runs']),
                            'pct_runs': round(row['pct_runs'],2),
                            'fours': int(row['fours']),
                            'sixes': int(row['sixes']),
                            'balls': int(row['balls'])
                        })
                    ww_display = pd.DataFrame(mirrored_rows)
            
                ww_display = ww_display[['sector','Sector Name','runs','pct_runs','fours','sixes','balls']].rename(columns={
                    'sector':'Sector','runs':'Runs','pct_runs':'Pct of Runs','fours':'4s','sixes':'6s','balls':'Balls'
                })
                ww_display['Pct of Runs'] = ww_display['Pct of Runs'].apply(lambda x: f"{x:.2f}%")
            
                # draw figure using requested style and mapping
                fig = draw_cricket_field_with_run_totals_requested(final_df, batsman_selected)
            
                # attempt safe plotting in Streamlit if available; fallbacks included
                plotted = False
                # try:
                #     safe_st_pyplot  # if you have a utility named this in your app
                # except NameError:
                #     safe_st_pyplot = None
            
                if safe_st_pyplot:
                    try:
                        safe_st_pyplot(fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
                        plotted = True
                    except Exception:
                        plotted = False
            
                if not plotted and st:
                    try:
                        st.pyplot(fig)
                        plotted = True
                    except Exception:
                        plotted = False
            
                if not plotted:
                    plt.show()
            
                # show table
                if st:
                    st.dataframe(ww_display)
                    side_label = "LHB" if is_lhb_table else "RHB"
                    st.markdown(f"<div style='text-align:center; margin-top:6px;'><strong>{batsman_selected}'s Wagon Chart ({side_label})</strong></div>", unsafe_allow_html=True)
                else:
                    print(ww_display.to_string(index=False))
                    side_label = "LHB" if is_lhb_table else "RHB"
                    print(f"\n{batsman_selected}'s Wagon Chart ({side_label})")





            # -------------------------
            # Pitchmaps below Wagon: two heatmaps (dots vs scoring) - increased height
            # -------------------------
# --- Batsman pitchmaps: Dot Balls & Runs (6 lengths including Full Toss), mirrored for LHB ---
            import numpy as np
            import matplotlib.pyplot as plt
            
            # safety: ensure streamlit variable 'st' may or may not exist
            try:
                import streamlit as st  # noqa
            except Exception:
                st = None
            
            # create 6x5 grids (lengths bottom->top: Short, Back of Length, Good, Full, Yorker, Full Toss)
            N_ROWS = 6
            N_COLS = 5
            dot_grid = np.zeros((N_ROWS, N_COLS), dtype=int)
            run_grid = np.zeros((N_ROWS, N_COLS), dtype=float)
            wkt_grid = np.zeros((N_ROWS, N_COLS), dtype=int)
            
            # only proceed if required columns exist
            if line_col in final_df.columns and length_col in final_df.columns:
                plot_df = final_df[[line_col, length_col, run_col]].copy().dropna(subset=[line_col, length_col])
            
                # helper mapping with fallbacks (tries global maps then heuristics)
                def get_line_index(val):
                    if val is None:
                        return None
                    # try global line_map if present
                    lm = globals().get('line_map', None)
                    if lm is not None:
                        try:
                            idx = lm.get(val)
                            if idx is not None:
                                return int(idx)
                        except Exception:
                            pass
                    s = str(val).strip().lower()
                    # heuristics left->right for RHB:
                    # 0: wide outside off, 1: outside off, 2: on stumps, 3: down leg, 4: wide down leg
                    if 'wide' in s and 'outside' in s and 'off' in s:
                        return 0
                    if 'outside off' in s or ('outside' in s and 'off' in s):
                        return 1
                    if ('on' in s and ('stump' in s or 'stumps' in s)) or 'middle' in s:
                        return 2
                    if 'down' in s and 'leg' in s:
                        return 3
                    if 'wide' in s and 'leg' in s:
                        return 4
                    # fallback by keywords
                    if 'off' in s:
                        return 1
                    if 'leg' in s:
                        return 3
                    return None
            
                def get_length_index(val):
                    """Map textual length to index 0..5 (0 short ... 5 full toss top)."""
                    if val is None:
                        return None
                    # try global length_map first
                    lm = globals().get('length_map', None)
                    if lm is not None:
                        try:
                            idx = lm.get(val)
                            if idx is not None:
                                ii = int(idx)
                                # if global map uses 0..4, keep only if within 0..5 else attempt heuristics
                                if 0 <= ii < N_ROWS:
                                    return ii
                        except Exception:
                            pass
                    s = str(val).strip().lower()
                    # priority: full toss -> top row
                    if ('full' in s and 'toss' in s) or 'fulltoss' in s or 'full_toss' in s:
                        return 5
                    if 'york' in s:  # yorker
                        return 4
                    # full (but not full toss)
                    if s == 'full' or ('full' in s and 'toss' not in s):
                        return 3
                    # good length / length
                    if 'good length' in s or ('good' in s and 'length' in s) or (s == 'length' or 'length' in s):
                        return 2
                    # short good length
                    if 'short good' in s or 'back of length' in s or 'short_good' in s:
                        return 1
                    if 'short' in s:
                        return 0
                    # fallback None
                    return None
            
                # wicket types to mark (exclude run-out etc)
                wicket_set = {'caught', 'bowled', 'stumped', 'lbw','leg before wicket','hit wicket'}
            
                # iterate and fill grids
                for _, r in plot_df.iterrows():
                    li = get_line_index(r[line_col])
                    le = get_length_index(r[length_col])
                    if li is None or le is None:
                        continue
                    # safety bounds
                    if not (0 <= li < N_COLS and 0 <= le < N_ROWS):
                        continue
                    # runs
                    try:
                        rv = int(r[run_col] if r[run_col] is not None else 0)
                    except Exception:
                        try:
                            rv = int(float(r[run_col]))
                        except Exception:
                            rv = 0
                    run_grid[le, li] += rv
                    if rv == 0:
                        dot_grid[le, li] += 1
                    # check dismissal if present in final_df
                    if 'dismissal' in final_df.columns:
                        d = r.get('dismissal', '')
                        if isinstance(d, str) and d.strip().lower() in wicket_set:
                            # mark wicket (for batsman chart, likely 0 or 1)
                            wkt_grid[le, li] += 1
            
                # Determine batter handedness for mirroring
                try:
                    bat_hand_col_name = bat_hand_col
                except NameError:
                    bat_hand_col_name = 'bat_hand'
                batting_style_val = None
                is_lhb = False
                if bat_hand_col_name in final_df.columns and not final_df[bat_hand_col_name].dropna().empty:
                    batting_style_val = final_df[bat_hand_col_name].dropna().iloc[0]
                    is_lhb = isinstance(batting_style_val, str) and batting_style_val.strip().upper().startswith('L')
            
                # If LHB, mirror arrays left-right so off/leg sides swap about the vertical axis
                if is_lhb:
                    run_grid = np.fliplr(run_grid)
                    dot_grid = np.fliplr(dot_grid)
                    wkt_grid = np.fliplr(wkt_grid)
            
                # xtick labels (line positions left->right for RHB). For LHB we reverse labels to match flipped arrays
                xticklabels = ['Wide Out Off', 'Outside Off', 'On Stumps', 'Down Leg', 'Wide Down Leg']
                if is_lhb:
                    xticklabels = xticklabels[::-1]
                yticklabels = ['Short', 'Back of Length', 'Good', 'Full', 'Yorker', 'Full Toss']  # bottom->top
            
                # Plotting: two tall heatmaps side-by-side (Dot Balls, Runs)
                st.markdown("### Pitchmaps")
                c1, c2 = st.columns([1, 1])
            
                # dot heatmap (no numbers in cells)
                with c1:
                    st.markdown("**Dot Balls**")
                    fig1, ax1 = plt.subplots(figsize=(8, 14), dpi=150)
                    im1 = ax1.imshow(dot_grid, origin='lower', cmap='Blues')
                    ax1.set_xticks(range(N_COLS))
                    ax1.set_yticks(range(N_ROWS))
                    ax1.set_xticklabels(xticklabels, rotation=45, ha='right')
                    ax1.set_yticklabels(yticklabels)
                    # black minor gridlines for borders
                    ax1.set_xticks(np.arange(-0.5, N_COLS, 1), minor=True)
                    ax1.set_yticks(np.arange(-0.5, N_ROWS, 1), minor=True)
                    ax1.grid(which='minor', color='black', linewidth=0.6, alpha=0.8)
                    ax1.tick_params(which='minor', bottom=False, left=False)
                    # annotate only single 'W' if wicket exists in that cell
                    for i in range(N_ROWS):
                        for j in range(N_COLS):
                            if int(wkt_grid[i, j]) > 0:
                                # show 'W' (single char), not 'N W'
                                ax1.text(j, i, "W", ha='center', va='center', fontsize=16, weight='bold',
                                         color='gold', bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
                    cbar1 = fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
                    cbar1.set_label('Dot count', fontsize=10)
                    plt.tight_layout(pad=3.0)
                    if st:
                        st.pyplot(fig1)
                    else:
                        plt.show()
                    plt.close(fig1)
            
                # runs heatmap (no numbers in cells)
                with c2:
                    st.markdown("**Scoring Balls**")
                    fig2, ax2 = plt.subplots(figsize=(8, 14), dpi=150)
                    im2 = ax2.imshow(run_grid, origin='lower', cmap='Reds')
                    ax2.set_xticks(range(N_COLS))
                    ax2.set_yticks(range(N_ROWS))
                    ax2.set_xticklabels(xticklabels, rotation=45, ha='right')
                    ax2.set_yticklabels(yticklabels)
                    # black minor gridlines
                    ax2.set_xticks(np.arange(-0.5, N_COLS, 1), minor=True)
                    ax2.set_yticks(np.arange(-0.5, N_ROWS, 1), minor=True)
                    ax2.grid(which='minor', color='black', linewidth=0.6, alpha=0.8)
                    ax2.tick_params(which='minor', bottom=False, left=False)
                    # annotate 'W' only where wicket occurred
                    for i in range(N_ROWS):
                        for j in range(N_COLS):
                            if int(wkt_grid[i, j]) > 0:
                                ax2.text(j, i, "W", ha='center', va='center', fontsize=16, weight='bold',
                                         color='gold', bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
                    cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                    cbar2.set_label('Runs (sum)', fontsize=10)
                    plt.tight_layout(pad=3.0)
                    if st:
                        st.pyplot(fig2)
                    else:
                        plt.show()
                    plt.close(fig2)
            else:
                if st:
                    st.info("Pitchmap requires both 'line' and 'length' columns in dataset; skipping pitchmaps.")
                else:
                    print("Pitchmap requires both 'line' and 'length' columns in dataset; skipping pitchmaps.")



                    
            #     # import base64
            #     # from io import BytesIO
            #     # from PIL import Image
                
            #     # # Pixel height for pitchmaps (change this value to whatever visible height you want)
            #     # HEIGHT_PITCHMAP_PX = 1600

            #     # Assuming dot_grid and run_grid are 5x5 numpy arrays already defined
                
            #     st.markdown("### Pitchmaps")
            #     c1, c2 = st.columns([1, 1])
                
            #     with c1:
            #         st.markdown("**Dot Balls**")
            #         fig1, ax1 = plt.subplots(figsize=(8, 14), dpi=150)  # Increased height from 10 to 12
            #         im1 = ax1.imshow(dot_grid, origin='lower', cmap='Blues')
            #         ax1.set_xticks(range(5))
            #         ax1.set_yticks(range(5))
            #         ax1.set_xticklabels(['Wide Out Off', 'Outside Off', 'On Stumps', 'Down Leg', 'Wide Down Leg'],
            #                             rotation=45, ha='right')
            #         ax1.set_yticklabels(['Short', 'Back of Length', 'Good', 'Full', 'Yorker'])
            #         for i in range(5):
            #             for j in range(5):
            #                 ax1.text(j, i, int(dot_grid[i, j]), ha='center', va='center', color='black', fontsize=12)
            #         fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            #         plt.tight_layout(pad=3.0)
            #         st.pyplot(fig1)
                
            #     with c2:
            #         st.markdown("**Scoring Balls**")
            #         fig2, ax2 = plt.subplots(figsize=(8, 14), dpi=150)  # Increased height from 10 to 12
            #         im2 = ax2.imshow(run_grid, origin='lower', cmap='Reds')
            #         ax2.set_xticks(range(5))
            #         ax2.set_yticks(range(5))
            #         ax2.set_xticklabels(['Wide Out Off', 'Outside Off', 'On Stumps', 'Down Leg', 'Wide Down Leg'],
            #                             rotation=45, ha='right')
            #         ax2.set_yticklabels(['Short', 'Back of Length', 'Good', 'Full', 'Yorker'])
            #         for i in range(5):
            #             for j in range(5):
            #                 ax2.text(j, i, int(run_grid[i, j]), ha='center', va='center', color='black', fontsize=12)
            #         fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            #         plt.tight_layout(pad=3.0)
            #         st.pyplot(fig2)

            #             # Pitchmaps below Wagon: two heatmaps (dots vs scoring) - increased height
            # # -------------------------
            # run_grid = np.zeros((5,5), dtype=float)
            # dot_grid = np.zeros((5,5), dtype=int)

            # if line_col in final_df.columns and length_col in final_df.columns:
            #     plot_df = final_df[[line_col,length_col,run_col]].copy().dropna(subset=[line_col,length_col])
            #     for _, r in plot_df.iterrows():
            #         li = line_map.get(r[line_col], None)
            #         le = length_map.get(r[length_col], None)
            #         if li is None or le is None:
            #             continue
            #         runs_here = int(r[run_col])
            #         run_grid[le, li] += runs_here
            #         if runs_here == 0:
            #             dot_grid[le, li] += 1

            #     # import base64
            #     # from io import BytesIO
            #     # from PIL import Image

            #     # # Pixel height for pitchmaps (change this value to whatever visible height you want)
            #     # HEIGHT_PITCHMAP_PX = 1600

            #     # Assuming dot_grid and run_grid are 5x5 numpy arrays already defined

            #     st.markdown("### Pitchmaps")
            #     c1, c2 = st.columns([1, 1])

            #     with c1:
            #         st.markdown("**Dot Balls**")
            #         fig1, ax1 = plt.subplots(figsize=(8, 14), dpi=150)  # Increased height from 10 to 12
            #         im1 = ax1.imshow(dot_grid, origin='lower', cmap='Blues')
            #         ax1.set_xticks(range(5))
            #         ax1.set_yticks(range(5))
            #         ax1.set_xticklabels(['Wide Out Off', 'Outside Off', 'On Stumps', 'Down Leg', 'Wide Down Leg'],
            #                             rotation=45, ha='right')
            #         ax1.set_yticklabels(['Short', 'Back of Length', 'Good', 'Full', 'Yorker'])
            #         for i in range(5):
            #             for j in range(5):
            #                 ax1.text(j, i, int(dot_grid[i, j]), ha='center', va='center', color='black', fontsize=12)
            #         fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            #         plt.tight_layout(pad=3.0)
            #         st.pyplot(fig1)

            #     with c2:
            #         st.markdown("**Scoring Balls**")
            #         fig2, ax2 = plt.subplots(figsize=(8, 14), dpi=150)  # Increased height from 10 to 12
            #         im2 = ax2.imshow(run_grid, origin='lower', cmap='Reds')
            #         ax2.set_xticks(range(5))
            #         ax2.set_yticks(range(5))
            #         ax2.set_xticklabels(['Wide Out Off', 'Outside Off', 'On Stumps', 'Down Leg', 'Wide Down Leg'],
            #                             rotation=45, ha='right')
            #         ax2.set_yticklabels(['Short', 'Back of Length', 'Good', 'Full', 'Yorker'])
            #         for i in range(5):
            #             for j in range(5):
            #                 ax2.text(j, i, int(run_grid[i, j]), ha='center', va='center', color='black', fontsize=12)
            #         fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            #         plt.tight_layout(pad=3.0)
            #         st.pyplot(fig2)

            # else:
            #     st.info("Pitchmap requires both 'line' and 'length' columns in dataset; skipping pitchmaps.")
    
            # else:
            #     st.info("Pitchmap requires both 'line' and 'length' columns in dataset; skipping pitchmaps.")

            # Adapted shot-productivity + control charts for your dataset (uses `bdf`)
            # ---------- Shot productivity + SR + Dismissals + BallsPerDismissal (for selected batter only) ----------
            import plotly.express as px
            import pandas as pd
            import numpy as np
            pf=filtered_df
            # Defensive checks
            # if 'pf' not in globals():
            #     st.error("Player frame `pf` not found. Filter your DataFrame for the selected batsman into `pf` first.")
            #     st.stop()
            
            # Ensure required columns exist
            if 'batruns' not in pf.columns:
                st.error("Column 'batruns' is required but not found in player frame `pf`.")
                st.stop()
            if 'shot' not in pf.columns:
                st.error("Column 'shot' is required but not found in player frame `pf`.")
                st.stop()
            
            # Build working df for this batter (keep rows that have shot info)
            df_local = pf.dropna(subset=['shot']).copy()
            if df_local.empty:
                st.info("No deliveries with 'shot' recorded for this batsman.")
            else:
                # Ensure numeric batruns
                df_local['batruns'] = pd.to_numeric(df_local['batruns'], errors='coerce').fillna(0).astype(int)
            
                # Compute dismissal flag (is_wkt) if not already present
                if 'is_wkt' not in df_local.columns:
                    # compute out_flag and dismissal_clean locally
                    df_local['out_flag'] = pd.to_numeric(df_local.get('out', 0), errors='coerce').fillna(0).astype(int)
                    df_local['dismissal_clean'] = df_local.get('dismissal', "").astype(str).str.lower().str.strip().replace({'nan':'','none':''})
                    special_runout_types = set(['run out','runout','retired','retired not out','retired out','obstructing the field'])
                    df_local['is_wkt'] = df_local.apply(
                        lambda r: 1 if (int(r.get('out_flag',0)) == 1 and str(r.get('dismissal_clean','')).strip() not in special_runout_types and str(r.get('dismissal_clean','')).strip() != '') else 0,
                        axis=1
                    )
                else:
                    df_local['is_wkt'] = pd.to_numeric(df_local['is_wkt'], errors='coerce').fillna(0).astype(int)
            
                # Total runs (batruns) by this batter (used for percentage)
                total_runs = df_local['batruns'].sum()
            
                # Group by shot to compute runs, balls, dismissals
                shot_grp = df_local.groupby('shot').agg(
                    runs_by_shot = ('batruns', 'sum'),
                    balls = ('shot', 'size'),
                    dismissals = ('is_wkt', 'sum')
                ).reset_index()
            
                # Compute derived metrics
                shot_grp['% of Runs'] = shot_grp['runs_by_shot'].apply(lambda r: (r / total_runs * 100.0) if total_runs > 0 else 0.0)
                shot_grp['SR'] = shot_grp.apply(lambda r: (r['runs_by_shot'] / r['balls'] * 100.0) if r['balls']>0 else np.nan, axis=1)
                # Balls per dismissal: if dismissals == 0 -> NaN (we'll show '-' later)
                shot_grp['BallsPerDismissal'] = shot_grp.apply(lambda r: (r['balls'] / r['dismissals']) if r['dismissals']>0 else np.nan, axis=1)
            
                # Round nicely for display
                shot_grp['% of Runs'] = shot_grp['% of Runs'].round(2)
                shot_grp['SR'] = shot_grp['SR'].round(2)
                shot_grp['BallsPerDismissal'] = shot_grp['BallsPerDismissal'].round(2)
            
                # Sort for plotting (so the biggest % of runs appear at top)
                productive_shot_df = shot_grp.sort_values('% of Runs', ascending=True)
            
                # -------------------------
                # Control % (if control column exists)
                # -------------------------
                control_df = None
                if 'control' in df_local.columns:
                    # normalize control (robust coercion to 0/1)
                    def _to_control_num(x):
                        if pd.isna(x):
                            return 0
                        try:
                            n = float(x)
                            return 1 if int(n) != 0 else 0
                        except:
                            s = str(x).strip().lower()
                            if s in ('1','true','t','y','yes','controlled','c','ok'):
                                return 1
                            return 0
                    df_local['control_num'] = df_local['control'].apply(_to_control_num).astype(int)
                    control_grp = df_local.groupby('shot').agg(
                        total_shots = ('control_num','size'),
                        controlled_shots = ('control_num','sum')
                    ).reset_index()
                    control_grp['Control Percentage'] = control_grp.apply(
                        lambda r: (r['controlled_shots'] / r['total_shots'] * 100.0) if r['total_shots']>0 else 0.0,
                        axis=1
                    )
                    control_grp['Control Percentage'] = control_grp['Control Percentage'].round(2)
                    control_df = control_grp.sort_values('Control Percentage', ascending=True)
            
                # -------------------------
                # Plotting: left = Productive Shots (% of runs) with hover showing SR, Dismissals & BallsPerDismissal
                # -------------------------
                col1, col2 = st.columns(2)
            
                with col1:
                    st.markdown("###  Most Productive Shots ")
                    if productive_shot_df.empty:
                        st.info("No shot data to plot.")
                    else:
                        fig1 = px.bar(
                            productive_shot_df,
                            x='% of Runs',
                            y='shot',
                            orientation='h',
                            color='% of Runs',
                            labels={'shot': 'Shot Type', '% of Runs': '% of Runs'},
                            height=600,
                            hover_data={
                                'runs_by_shot': True,
                                'balls': True,
                                'SR': True,
                                'dismissals': True,
                                'BallsPerDismissal': True,
                                '% of Runs': ':.2f'
                            }
                        )
                        fig1.update_layout(
                            margin=dict(l=180, r=40, t=40, b=40),
                            xaxis_title='% of Runs',
                            yaxis_title=None,
                        )
                        # show percentage inside bars with 2 decimals
                        fig1.update_traces(texttemplate='%{x:.2f}%', textposition='inside', hovertemplate=None)
                        fig1.update_yaxes(categoryorder='total ascending')
                        st.plotly_chart(fig1, use_container_width=True)
            
                # -------------------------
                # Plotting: right = Control % by shot (if available)
                # -------------------------
                with col2:
                    st.markdown("###  Control Percentage by Shot")
                    if control_df is None:
                        st.info("No `control` column available for this batter; skipping Control % chart.")
                    else:
                        fig2 = px.bar(
                            control_df,
                            x='Control Percentage',
                            y='shot',
                            orientation='h',
                            color='Control Percentage',
                            labels={'shot': 'Shot Type', 'Control Percentage': '% Controlled'},
                            height=520,
                            hover_data={'total_shots': True, 'controlled_shots': True, 'Control Percentage': ':.2f'}
                        )
                        fig2.update_layout(
                            margin=dict(l=160, r=30, t=40, b=40),
                            xaxis_title='Control Percentage',
                            yaxis_title=None,
                        )
                        fig2.update_traces(texttemplate='%{x:.2f}%', textposition='inside', hovertemplate=None)
                        fig2.update_yaxes(categoryorder='total ascending')
                        st.plotly_chart(fig2, use_container_width=True)
            
                # ---------- Underlying numbers (rounded) ----------
                st.markdown("#### Underlying numbers (rounded)")
                prod_show = productive_shot_df[['shot', 'runs_by_shot', 'balls', 'SR', 'dismissals', 'BallsPerDismissal', '% of Runs']].copy()
                prod_show = prod_show.rename(columns={
                    'shot': 'Shot',
                    'runs_by_shot': 'Runs (batruns)',
                    'balls': 'Balls',
                    'SR': 'SR (%)',
                    'dismissals': 'Dismissals',
                    'BallsPerDismissal': 'Balls per Dismissal',
                    '% of Runs': '% of Runs'
                })
                # Replace NaN BallsPerDismissal with '-' for display
                prod_show['Balls per Dismissal'] = prod_show['Balls per Dismissal'].apply(lambda x: '-' if pd.isna(x) else (round(x, 2) if not isinstance(x, str) else x)).astype(str)  # Convert entire column to string
                prod_show['SR (%)'] = prod_show['SR (%)'].apply(lambda x: '-' if pd.isna(x) else round(x,2))
                prod_show['% of Runs'] = prod_show['% of Runs'].round(2)
                prod_show = prod_show.sort_values('% of Runs', ascending=False).reset_index(drop=True)
            
                st.dataframe(prod_show, use_container_width=True)
            
                # If control table exists, show it too
                if control_df is not None:
                    ctrl_show = control_df[['shot','total_shots','controlled_shots','Control Percentage']].copy()
                    ctrl_show = ctrl_show.rename(columns={
                        'shot': 'Shot',
                        'total_shots': 'Total Shots',
                        'controlled_shots': 'Controlled Shots',
                        'Control Percentage': '% Controlled'
                    })
                    st.dataframe(ctrl_show, use_container_width=True)
            # ---------- end shot productivity snippet ----------


    
    # ---------------------------
    # Bowler Analysis (match-level)
                # ---------------------------
    # elif option == "Bowler Analysis":
    #     # -------------------------
    #     # Bowler selection & base metrics (unchanged)
    #     # -------------------------
    #     bowler_choices = sorted([x for x in temp_df[bowler_col].dropna().unique() if str(x).strip() not in ("","0")])
    #     if not bowler_choices:
    #         st.info("No bowlers found in this match.")
    #     else:
    #         bowler_selected = st.selectbox("Select Bowler", options=bowler_choices, index=0)
    #         filtered_df = temp_df[temp_df[bowler_col] == bowler_selected].copy()
    
    #         # Legal balls definition: both wide & noball == 0
    #         filtered_df['noball'] = pd.to_numeric(filtered_df.get('noball', 0), errors='coerce').fillna(0).astype(int)
    #         filtered_df['wide'] = pd.to_numeric(filtered_df.get('wide', 0), errors='coerce').fillna(0).astype(int)
    #         filtered_df['legal_ball'] = ((filtered_df['noball'] == 0) & (filtered_df['wide'] == 0)).astype(int)
    
    #         # runs conceded should be sum of score/batruns when byes & legbyes ==0
    #         if 'score' in filtered_df.columns:
    #             cond = (~filtered_df.get('byes', 0).astype(bool)) & (~filtered_df.get('legbyes', 0).astype(bool))
    #             runs_given = int(filtered_df.loc[cond, 'score'].sum() if 'score' in filtered_df.columns else 0)
    #         elif 'batruns' in filtered_df.columns:
    #             cond = (~filtered_df.get('byes', 0).astype(bool)) & (~filtered_df.get('legbyes', 0).astype(bool))
    #             runs_given = int(filtered_df.loc[cond, 'batruns'].sum())
    #         else:
    #             runs_given = int(filtered_df.get('bowlruns', filtered_df.get('total_runs', 0)).sum())
    
    #         balls_bowled = int(filtered_df['legal_ball'].sum())
    #         wickets = int(filtered_df['is_wkt'].sum()) if 'is_wkt' in filtered_df.columns else 0
    #         econ = (runs_given * 6.0 / balls_bowled) if balls_bowled > 0 else float('nan')
    #         avg = (runs_given / wickets) if wickets > 0 else float('nan')
    #         sr = (balls_bowled / wickets) if wickets > 0 else float('nan')
    
    #         st.markdown(f"### Bowling Analysis for {bowler_selected}")
    #         col1, col2 = st.columns(2)
    #         with col1:
    #             st.write(f"Runs conceded: {runs_given}")
    #             st.write(f"Balls: {balls_bowled}")
    #             st.write(f"Wickets: {wickets}")
    #         with col2:
    #             st.write(f"Econ: {econ:.2f}" if not np.isnan(econ) else "Econ: -")
    #             st.write(f"Avg: {avg:.2f}" if not np.isnan(avg) else "Avg: -")
    #             st.write(f"SR: {sr:.2f}" if not np.isnan(sr) else "SR: -")
    
    #         # -------------------------
    #         # Pitchmaps: one 3x2 figure (Percent of balls, Dots, Runs) vs LHB / RHB
    #         # -------------------------
    #         required_cols = [line_col, length_col]
    #         missing = [c for c in required_cols if c not in filtered_df.columns]
    #         if missing:
    #             st.info(f"Pitchmap requires columns {missing} in dataset; skipping pitchmaps.")
    #         else:
    #             if 'line_map' not in globals() or 'length_map' not in globals():
    #                 st.error("line_map and length_map mappings must be defined in the app.")
    #             else:
    #                 df_legal = filtered_df[filtered_df['legal_ball'] == 1].copy()
    #                 if df_legal.empty:
    #                     st.info("No legal deliveries for this bowler in this match to plot pitchmaps.")
    #                 else:
    #                     try:
    #                         bh_col = bat_hand_col
    #                     except NameError:
    #                         bh_col = 'bat_hand'
    
    #                     if bh_col not in df_legal.columns:
    #                         df_legal[bh_col] = ''
    #                     else:
    #                         df_legal[bh_col] = df_legal[bh_col].astype(str).str.strip()
    
    #                     def build_grids(df_sub):
    #                         count_grid = np.zeros((5, 5), dtype=int)
    #                         dot_grid = np.zeros((5, 5), dtype=int)
    #                         runs_grid = np.zeros((5, 5), dtype=float)
    #                         wkt_grid = np.zeros((5, 5), dtype=int)
    
    #                         if 'batruns' in df_sub.columns:
    #                             rv_col = 'batruns'
    #                         elif 'score' in df_sub.columns:
    #                             rv_col = 'score'
    #                         else:
    #                             rv_col = None
    
    #                         dismissal_col = 'dismissal' if 'dismissal' in df_sub.columns else None
    #                         wicket_set = {'caught', 'bowled', 'stumped', 'lbw','leg before wicket','hit wicket'}
    
    #                         for _, rr in df_sub.iterrows():
    #                             li = line_map.get(rr[line_col], None)
    #                             le = length_map.get(rr[length_col], None)
    #                             if li is None or le is None:
    #                                 continue
    
    #                             count_grid[le, li] += 1
    #                             rv = float(rr.get(rv_col, 0) or 0) if rv_col else 0
    #                             runs_grid[le, li] += rv
    #                             if rv == 0:
    #                                 dot_grid[le, li] += 1
    #                             if dismissal_col and str(rr.get(dismissal_col, '')).lower() in wicket_set:
    #                                 wkt_grid[le, li] += 1
    
    #                         return count_grid, dot_grid, runs_grid, wkt_grid
    
    #                     df_lhb = df_legal[df_legal[bh_col].str.upper().str.startswith('L')]
    #                     df_rhb = df_legal[df_legal[bh_col].str.upper().str.startswith('R')]
    
    #                     count_l, dot_l, runs_l, wkt_l = build_grids(df_lhb)
    #                     count_r, dot_r, runs_r, wkt_r = build_grids(df_rhb)
    
    #                     perc_l = (count_l / count_l.sum() * 100) if count_l.sum() else np.zeros_like(count_l)
    #                     perc_r = (count_r / count_r.sum() * 100) if count_r.sum() else np.zeros_like(count_r)
    
    #                     disp = {
    #                         'perc_l': np.fliplr(perc_l),
    #                         'dot_l': np.fliplr(dot_l),
    #                         'run_l': np.fliplr(runs_l),
    #                         'wkt_l': np.fliplr(wkt_l),
    #                         'perc_r': perc_r,
    #                         'dot_r': dot_r,
    #                         'run_r': runs_r,
    #                         'wkt_r': wkt_r
    #                     }
    
    #                     xticks_r = ['Wide Out Off', 'Outside Off', 'On Stumps', 'Down Leg', 'Wide Down Leg']
    #                     xticks_l = xticks_r[::-1]
    #                     yticklabels = ['Short', 'Back of Length', 'Good', 'Full', 'Yorker']
    
    #                     fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    #                     plt.suptitle(f"{bowler_selected} — Pitchmaps vs LHB / RHB", fontsize=16, weight='bold')
    
    #                     plot_defs = [
    #                         ('perc_l', '% of balls vs LHB', 'Blues', xticks_l),
    #                         ('perc_r', '% of balls vs RHB', 'Blues', xticks_r),
    #                         ('dot_l', 'Dot balls vs LHB', 'Blues', xticks_l),
    #                         ('dot_r', 'Dot balls vs RHB', 'Blues', xticks_r),
    #                         ('run_l', 'Runs conceded vs LHB', 'Reds', xticks_l),
    #                         ('run_r', 'Runs conceded vs RHB', 'Reds', xticks_r),
    #                     ]
    
    #                     for ax, (k, title, cmap, xt) in zip(axes.flat, plot_defs):
    #                         im = ax.imshow(disp[k], origin='lower', cmap=cmap)
    #                         ax.set_title(title)
    #                         ax.set_xticks(range(5))
    #                         ax.set_yticks(range(5))
    #                         ax.set_xticklabels(xt, rotation=45, ha='right')
    #                         ax.set_yticklabels(yticklabels)
    
    #                         # ✅ LIGHT CELL BORDERS
    #                         ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    #                         ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    #                         ax.grid(which='minor', color='black', linewidth=0.6, alpha=0.6)
    #                         ax.tick_params(which='minor', bottom=False, left=False)
    
    #                         wkt_grid = disp[k.replace('perc', 'wkt').replace('dot', 'wkt').replace('run', 'wkt')]
    #                         for i in range(5):
    #                             for j in range(5):
    #                                 if wkt_grid[i, j] > 0:
    #                                     ax.text(j, i, f"{wkt_grid[i, j]} W",
    #                                             ha='center', va='center',
    #                                             fontsize=14, weight='bold',
    #                                             color='gold',
    #                                             bbox=dict(facecolor='black', alpha=0.6,
    #                                                       boxstyle='round,pad=0.2'))
    
    #                         fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    
    #                     plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    #                     st.pyplot(fig, clear_figure=True)
    #                     plt.close(fig)

    # elif option == "Bowler Analysis":
    #     # -------------------------
    #     # Bowler selection & base metrics (unchanged)
    #     # -------------------------
    #     bowler_choices = sorted([x for x in temp_df[bowler_col].dropna().unique() if str(x).strip() not in ("","0")])
    #     if not bowler_choices:
    #         st.info("No bowlers found in this match.")
    #     else:
    #         bowler_selected = st.selectbox("Select Bowler", options=bowler_choices, index=0)
    #         filtered_df = temp_df[temp_df[bowler_col] == bowler_selected].copy()
    
    #         # Legal balls definition: both wide & noball == 0
    #         filtered_df['noball'] = pd.to_numeric(filtered_df.get('noball', 0), errors='coerce').fillna(0).astype(int)
    #         filtered_df['wide'] = pd.to_numeric(filtered_df.get('wide', 0), errors='coerce').fillna(0).astype(int)
    #         filtered_df['legal_ball'] = ((filtered_df['noball'] == 0) & (filtered_df['wide'] == 0)).astype(int)
    
    #         # runs conceded should be sum of score/batruns when byes & legbyes ==0
    #         if 'score' in filtered_df.columns:
    #             cond = (~filtered_df.get('byes', 0).astype(bool)) & (~filtered_df.get('legbyes', 0).astype(bool))
    #             runs_given = int(filtered_df.loc[cond, 'score'].sum() if 'score' in filtered_df.columns else 0)
    #         elif 'batruns' in filtered_df.columns:
    #             cond = (~filtered_df.get('byes', 0).astype(bool)) & (~filtered_df.get('legbyes', 0).astype(bool))
    #             runs_given = int(filtered_df.loc[cond, 'batruns'].sum())
    #         else:
    #             runs_given = int(filtered_df.get('bowlruns', filtered_df.get('total_runs', 0)).sum())
    
    #         balls_bowled = int(filtered_df['legal_ball'].sum())
    #         wickets = int(filtered_df['is_wkt'].sum()) if 'is_wkt' in filtered_df.columns else 0
    #         econ = (runs_given * 6.0 / balls_bowled) if balls_bowled > 0 else float('nan')
    #         avg = (runs_given / wickets) if wickets > 0 else float('nan')
    #         sr = (balls_bowled / wickets) if wickets > 0 else float('nan')
    
    #         st.markdown(f"### Bowling Analysis for {bowler_selected}")
    #         col1, col2 = st.columns(2)
    #         with col1:
    #             st.write(f"Runs conceded: {runs_given}")
    #             st.write(f"Balls: {balls_bowled}")
    #             st.write(f"Wickets: {wickets}")
    #         with col2:
    #             st.write(f"Econ: {econ:.2f}" if not np.isnan(econ) else "Econ: -")
    #             st.write(f"Avg: {avg:.2f}" if not np.isnan(avg) else "Avg: -")
    #             st.write(f"SR: {sr:.2f}" if not np.isnan(sr) else "SR: -")
    
    #         # -------------------------
    #         # Pitchmaps: one 3x2 figure (Percent of balls, Dots, Runs) vs LHB / RHB
    #         # -------------------------
    #         # Validate mapping variables exist: line_map, length_map, and column names
    #         required_cols = [line_col, length_col]
    #         missing = [c for c in required_cols if c not in filtered_df.columns]
    #         if missing:
    #             st.info(f"Pitchmap requires columns {missing} in dataset; skipping pitchmaps.")
    #         else:
    #             # Check line_map / length_map existence
    #             if 'line_map' not in globals() or 'length_map' not in globals():
    #                 st.error("line_map and length_map mappings must be defined in the app (map textual 'line'/'length' -> 0..4 indices). Skipping pitchmaps.")
    #             else:
    #                 # Work only with legal deliveries
    #                 df_legal = filtered_df[filtered_df['legal_ball'] == 1].copy()
    #                 if df_legal.empty:
    #                     st.info("No legal deliveries for this bowler in this match to plot pitchmaps.")
    #                 else:
    #                     # Determine bat-hand column name
    #                     try:
    #                         bh_col = bat_hand_col
    #                     except NameError:
    #                         bh_col = 'bat_hand'
    
    #                     if bh_col not in df_legal.columns:
    #                         df_legal[bh_col] = ''
    #                     else:
    #                         df_legal[bh_col] = df_legal[bh_col].astype(str).str.strip()
    
    #                     # Helper to build grids
    #                     def build_grids(df_sub):
    #                         """
    #                         returns:
    #                           count_grid (5x5), dot_grid (5x5), runs_grid (5x5), wkt_grid (5x5)
    #                         indexing: [length_idx, line_idx] where length_idx 0..4 (short->yorker),
    #                                   line_idx 0..4 (wide out off -> wide down leg)
    #                         """
    #                         count_grid = np.zeros((5, 5), dtype=int)
    #                         dot_grid = np.zeros((5, 5), dtype=int)
    #                         runs_grid = np.zeros((5, 5), dtype=float)
    #                         wkt_grid = np.zeros((5, 5), dtype=int)
    
    #                         # pick run column preference
    #                         if 'batruns' in df_sub.columns:
    #                             rv_col = 'batruns'
    #                         elif 'score' in df_sub.columns:
    #                             rv_col = 'score'
    #                         else:
    #                             rv_col = None
    
    #                         dismissal_col = 'dismissal' if 'dismissal' in df_sub.columns else None
    #                         wicket_set = {'caught', 'bowled', 'stumped', 'lbw','leg before wicket','hit wicket'}
    
    #                         for _, rr in df_sub.iterrows():
    #                             li = line_map.get(rr[line_col], None)
    #                             le = length_map.get(rr[length_col], None)
    #                             if li is None or le is None:
    #                                 # skip if mapping missing
    #                                 continue
    #                             # increment counts
    #                             count_grid[le, li] += 1
    
    #                             rv = 0.0
    #                             if rv_col:
    #                                 try:
    #                                     rv = float(rr.get(rv_col, 0) or 0)
    #                                 except:
    #                                     rv = 0.0
    #                             runs_grid[le, li] += rv
    
    #                             if rv == 0:
    #                                 dot_grid[le, li] += 1
    
    #                             if dismissal_col:
    #                                 d = rr.get(dismissal_col, '')
    #                                 if isinstance(d, str) and d.strip().lower() in wicket_set:
    #                                     wkt_grid[le, li] += 1
    
    #                         return count_grid, dot_grid, runs_grid, wkt_grid
    
    #                     # split by batter hand
    #                     df_lhb = df_legal[df_legal[bh_col].fillna('').astype(str).str.upper().str.startswith('L')].copy()
    #                     df_rhb = df_legal[df_legal[bh_col].fillna('').astype(str).str.upper().str.startswith('R')].copy()
    
    #                     count_l, dot_l, runs_l, wkt_l = build_grids(df_lhb)
    #                     count_r, dot_r, runs_r, wkt_r = build_grids(df_rhb)
    
    #                     total_l = int(count_l.sum())
    #                     total_r = int(count_r.sum())
    
    #                     # compute percent arrays safely (avoid div by zero)
    #                     perc_l = (count_l.astype(float) / total_l * 100.0) if total_l > 0 else np.zeros_like(count_l, dtype=float)
    #                     perc_r = (count_r.astype(float) / total_r * 100.0) if total_r > 0 else np.zeros_like(count_r, dtype=float)
    
    #                     # For LHB: display mirrored horizontally so off/leg swap visually
    #                     disp_count_l = np.fliplr(count_l)
    #                     disp_dot_l = np.fliplr(dot_l)
    #                     disp_runs_l = np.fliplr(runs_l)
    #                     disp_wkt_l = np.fliplr(wkt_l)
    #                     disp_perc_l = np.fliplr(perc_l)
    
    #                     disp_count_r = count_r.copy()
    #                     disp_dot_r = dot_r.copy()
    #                     disp_runs_r = runs_r.copy()
    #                     disp_wkt_r = wkt_r.copy()
    #                     disp_perc_r = perc_r.copy()
    
    #                     # xticklabels mapping: left->right for RHB. For LHB we'll reverse labels to match flipped arrays
    #                     xticks_r = ['Wide Out Off', 'Outside Off', 'On Stumps', 'Down Leg', 'Wide Down Leg']
    #                     xticks_l = xticks_r[::-1]
    #                     yticklabels = ['Short', 'Back of Length', 'Good', 'Full', 'Yorker']
    
    #                     # One 3x2 figure
    #                     fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    #                     plt.suptitle(f"{bowler_selected} — Pitchmaps vs LHB / RHB (legal deliveries only)", fontsize=16, weight='bold')
    
    #                     # Row 1: percent of balls (color only; no numbers)
    #                     ax = axes[0, 0]
    #                     vmax_pct = max(float(np.nanmax(disp_perc_l)), float(np.nanmax(disp_perc_r)), 1e-6)
    #                     im = ax.imshow(disp_perc_l, origin='lower', cmap='Blues', vmin=0, vmax=vmax_pct)
    #                     ax.set_title("Percent of balls vs LHB", fontsize=12)
    #                     ax.set_xticks(range(5)); ax.set_yticks(range(5))
    #                     ax.set_xticklabels(xticks_l, rotation=45, ha='right')
    #                     ax.set_yticklabels(yticklabels)
    #                     # Only annotate wickets (N W) if >0
    #                     for i in range(5):
    #                         for j in range(5):
    #                             wval = int(disp_wkt_l[i, j])
    #                             if wval > 0:
    #                                 ax.text(j, i, f"{wval} W", ha='center', va='center', fontsize=14, color='gold', weight='bold',
    #                                         bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
    #                     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    #                     cbar.set_label('% of balls', fontsize=10)
    
    #                     ax = axes[0, 1]
    #                     im = ax.imshow(disp_perc_r, origin='lower', cmap='Blues', vmin=0, vmax=vmax_pct)
    #                     ax.set_title("Percent of balls vs RHB", fontsize=12)
    #                     ax.set_xticks(range(5)); ax.set_yticks(range(5))
    #                     ax.set_xticklabels(xticks_r, rotation=45, ha='right')
    #                     ax.set_yticklabels(yticklabels)
    #                     for i in range(5):
    #                         for j in range(5):
    #                             wval = int(disp_wkt_r[i, j])
    #                             if wval > 0:
    #                                 ax.text(j, i, f"{wval} W", ha='center', va='center', fontsize=14, color='gold', weight='bold',
    #                                         bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
    #                     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    #                     cbar.set_label('% of balls', fontsize=10)
    
    #                     # Row 2: dot balls (color only; no numbers)
    #                     ax = axes[1, 0]
    #                     vmax_dot = max(float(np.nanmax(disp_dot_l)), float(np.nanmax(disp_dot_r)), 1e-6)
    #                     im = ax.imshow(disp_dot_l, origin='lower', cmap='Blues', vmin=0, vmax=vmax_dot)
    #                     ax.set_title("Dot balls vs LHB (count)", fontsize=12)
    #                     ax.set_xticks(range(5)); ax.set_yticks(range(5))
    #                     ax.set_xticklabels(xticks_l, rotation=45, ha='right')
    #                     ax.set_yticklabels(yticklabels)
    #                     for i in range(5):
    #                         for j in range(5):
    #                             wval = int(disp_wkt_l[i, j])
    #                             if wval > 0:
    #                                 ax.text(j, i, f"{wval} W", ha='center', va='center', fontsize=14, color='gold', weight='bold',
    #                                         bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
    #                     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    #                     cbar.set_label('Dot count', fontsize=10)
    
    #                     ax = axes[1, 1]
    #                     im = ax.imshow(disp_dot_r, origin='lower', cmap='Blues', vmin=0, vmax=vmax_dot)
    #                     ax.set_title("Dot balls vs RHB (count)", fontsize=12)
    #                     ax.set_xticks(range(5)); ax.set_yticks(range(5))
    #                     ax.set_xticklabels(xticks_r, rotation=45, ha='right')
    #                     ax.set_yticklabels(yticklabels)
    #                     for i in range(5):
    #                         for j in range(5):
    #                             wval = int(disp_wkt_r[i, j])
    #                             if wval > 0:
    #                                 ax.text(j, i, f"{wval} W", ha='center', va='center', fontsize=14, color='gold', weight='bold',
    #                                         bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
    #                     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    #                     cbar.set_label('Dot count', fontsize=10)
    
    #                     # Row 3: runs conceded (color only; no numbers)
    #                     ax = axes[2, 0]
    #                     vmax_runs = max(float(np.nanmax(disp_runs_l)), float(np.nanmax(disp_runs_r)), 1e-6)
    #                     im = ax.imshow(disp_runs_l, origin='lower', cmap='Reds', vmin=0, vmax=vmax_runs)
    #                     ax.set_title("Runs conceded vs LHB (sum)", fontsize=12)
    #                     ax.set_xticks(range(5)); ax.set_yticks(range(5))
    #                     ax.set_xticklabels(xticks_l, rotation=45, ha='right')
    #                     ax.set_yticklabels(yticklabels)
    #                     for i in range(5):
    #                         for j in range(5):
    #                             wval = int(disp_wkt_l[i, j])
    #                             if wval > 0:
    #                                 ax.text(j, i, f"{wval} W", ha='center', va='center', fontsize=14, color='gold', weight='bold',
    #                                         bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
    #                     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    #                     cbar.set_label('Runs (sum)', fontsize=10)
    
    #                     ax = axes[2, 1]
    #                     im = ax.imshow(disp_runs_r, origin='lower', cmap='Reds', vmin=0, vmax=vmax_runs)
    #                     ax.set_title("Runs conceded vs RHB (sum)", fontsize=12)
    #                     ax.set_xticks(range(5)); ax.set_yticks(range(5))
    #                     ax.set_xticklabels(xticks_r, rotation=45, ha='right')
    #                     ax.set_yticklabels(yticklabels)
    #                     for i in range(5):
    #                         for j in range(5):
    #                             wval = int(disp_wkt_r[i, j])
    #                             if wval > 0:
    #                                 ax.text(j, i, f"{wval} W", ha='center', va='center', fontsize=14, color='gold', weight='bold',
    #                                         bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
    #                     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    #                     cbar.set_label('Runs (sum)', fontsize=10)
    
    #                     plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    #                     # Display once (Streamlit preferred)
    #                     try:
    #                         st.pyplot(fig, clear_figure=True)
    #                     except Exception:
    #                         # fallback to matplotlib display
    #                         plt.show()
    #                     finally:
    #                         plt.close(fig)

    elif option == "Bowler Analysis":
        # -------------------------
        # Bowler selection & base metrics (unchanged)
        # -------------------------
        bowler_choices = sorted([x for x in temp_df[bowler_col].dropna().unique() if str(x).strip() not in ("","0")])
        if not bowler_choices:
            st.info("No bowlers found in this match.")
        else:
            bowler_selected = st.selectbox("Select Bowler", options=bowler_choices, index=0)
            filtered_df = temp_df[temp_df[bowler_col] == bowler_selected].copy()

            # Legal balls definition: both wide & noball == 0
            filtered_df['noball'] = pd.to_numeric(filtered_df.get('noball', 0), errors='coerce').fillna(0).astype(int)
            filtered_df['wide'] = pd.to_numeric(filtered_df.get('wide', 0), errors='coerce').fillna(0).astype(int)
            filtered_df['legal_ball'] = ((filtered_df['noball'] == 0) & (filtered_df['wide'] == 0)).astype(int)

            # runs conceded should be sum of score/batruns when byes & legbyes ==0
            if 'score' in filtered_df.columns:
                cond = (~filtered_df.get('byes', 0).astype(bool)) & (~filtered_df.get('legbyes', 0).astype(bool))
                runs_given = int(filtered_df.loc[cond, 'score'].sum() if 'score' in filtered_df.columns else 0)
            elif 'batruns' in filtered_df.columns:
                cond = (~filtered_df.get('byes', 0).astype(bool)) & (~filtered_df.get('legbyes', 0).astype(bool))
                runs_given = int(filtered_df.loc[cond, 'batruns'].sum())
            else:
                runs_given = int(filtered_df.get('bowlruns', filtered_df.get('total_runs', 0)).sum())

            balls_bowled = int(filtered_df['legal_ball'].sum())
            wickets = int(filtered_df['is_wkt'].sum()) if 'is_wkt' in filtered_df.columns else 0
            econ = (runs_given * 6.0 / balls_bowled) if balls_bowled > 0 else float('nan')
            avg = (runs_given / wickets) if wickets > 0 else float('nan')
            sr = (balls_bowled / wickets) if wickets > 0 else float('nan')

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

            # -------------------------
            # Pitchmaps: one 3x2 figure (Percent of balls, Dots, Runs) vs LHB / RHB
            # -------------------------
            required_cols = [line_col, length_col]
            missing = [c for c in required_cols if c not in filtered_df.columns]
            if missing:
                st.info(f"Pitchmap requires columns {missing} in dataset; skipping pitchmaps.")
            else:
                if 'line_map' not in globals() or 'length_map' not in globals():
                    # we can still proceed but will attempt robust keyword mapping
                    st.warning("line_map and/or length_map not present in globals — using keyword fallbacks for mapping.")
                df_legal = filtered_df[filtered_df['legal_ball'] == 1].copy()
                if df_legal.empty:
                    st.info("No legal deliveries for this bowler in this match to plot pitchmaps.")
                else:
                    try:
                        bh_col = bat_hand_col
                    except NameError:
                        bh_col = 'bat_hand'

                    if bh_col not in df_legal.columns:
                        df_legal[bh_col] = ''
                    else:
                        df_legal[bh_col] = df_legal[bh_col].astype(str).str.strip()

                    # --- robust mapping helpers (fall back to keywords if global maps don't contain value) ---
                    def get_line_index(val):
                        """Return column index 0..4 for a line string. Try global line_map first, then heuristics."""
                        if val is None:
                            return None
                        # try global mapping if exists
                        try:
                            lm = globals().get('line_map', None)
                            if lm is not None:
                                # supports dict-like
                                idx = lm.get(val)
                                if idx is not None:
                                    return int(idx)
                        except Exception:
                            pass
                        s = str(val).strip().lower()
                        # heuristics - match most likely phrase
                        if 'wide' in s and 'off' in s and 'outside' in s:
                            return 0
                        if 'outside off' in s or ('outside' in s and 'off' in s and 'wide' not in s):
                            return 1
                        if 'on' in s and ('stump' in s or 'stumps' in s):
                            return 2
                        if ('down' in s and 'leg' in s) or ('leg' in s and 'down' not in s and 'mid' in s):
                            return 3
                        if 'wide' in s and 'down' in s and 'leg' in s:
                            return 4
                        # fallback: try to detect off/leg words
                        if 'off' in s:
                            return 1
                        if 'leg' in s:
                            return 3
                        return None

                    def get_length_index(val, n_rows=6):
                        """Return row index 0..n_rows-1 mapping bottom->top:
                           0: Short
                           1: Short Good Length (back of length)
                           2: Length / Good
                           3: Full
                           4: Yorker
                           5: Full Toss
                           Try global length_map first, else heuristics.
                        """
                        if val is None:
                            return None
                        # try global mapping first
                        try:
                            lm = globals().get('length_map', None)
                            if lm is not None:
                                idx = lm.get(val)
                                if idx is not None:
                                    # If global map returned 0..4 we need to be careful:
                                    # we'll accept global idx if it's within [0,n_rows-1]
                                    try:
                                        ii = int(idx)
                                        if 0 <= ii < n_rows:
                                            return ii
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        s = str(val).strip().lower()
                        # handle variants:
                        if 'full' in s and ('toss' in s or 'full_toss' in s or 'fulltoss' in s):
                            return n_rows - 1  # top row: Full Toss
                        if 'york' in s:  # yorker
                            return n_rows - 2
                        # match exact 'full' but not full toss
                        if s == 'full' or ('full' in s and 'toss' not in s):
                            return n_rows - 3
                        # 'length' or 'good length'
                        if 'good length' in s or ('short' not in s and ('length' in s or 'good' in s)):
                            # map to middle (index 2)
                            return 2
                        # short good length variants
                        if 'short good' in s or 'short_good' in s or ('good' in s and 'short' in s):
                            return 1
                        # short
                        if 'short' in s:
                            return 0
                        # fallback: if numeric or unknown, return None
                        return None

                    def build_grids(df_sub, n_rows=6, n_cols=5):
                        """
                        returns:
                          count_grid (n_rows x n_cols), dot_grid, runs_grid, wkt_grid
                        indexing: [length_idx, line_idx] where length_idx 0..n_rows-1 (short->full toss),
                                  line_idx 0..n_cols-1 (wide out off -> wide down leg)
                        """
                        count_grid = np.zeros((n_rows, n_cols), dtype=int)
                        dot_grid = np.zeros((n_rows, n_cols), dtype=int)
                        runs_grid = np.zeros((n_rows, n_cols), dtype=float)
                        wkt_grid = np.zeros((n_rows, n_cols), dtype=int)

                        if 'batruns' in df_sub.columns:
                            rv_col = 'batruns'
                        elif 'score' in df_sub.columns:
                            rv_col = 'score'
                        else:
                            rv_col = None

                        dismissal_col = 'dismissal' if 'dismissal' in df_sub.columns else None
                        wicket_set = {'caught', 'bowled', 'stumped', 'lbw','leg before wicket','hit wicket'}

                        for _, rr in df_sub.iterrows():
                            li = get_line_index(rr.get(line_col, None))
                            le = get_length_index(rr.get(length_col, None), n_rows=n_rows)
                            if li is None or le is None:
                                # skip if mapping missing
                                continue

                            # safety bounds:
                            if not (0 <= li < n_cols and 0 <= le < n_rows):
                                continue

                            count_grid[le, li] += 1
                            rv = 0.0
                            if rv_col:
                                try:
                                    rv = float(rr.get(rv_col, 0) or 0)
                                except:
                                    rv = 0.0
                            runs_grid[le, li] += rv

                            if rv == 0:
                                dot_grid[le, li] += 1

                            if dismissal_col:
                                d = rr.get(dismissal_col, '')
                                if isinstance(d, str) and d.strip().lower() in wicket_set:
                                    wkt_grid[le, li] += 1

                        return count_grid, dot_grid, runs_grid, wkt_grid

                    # NEW: n_rows = 6 to include Full Toss ahead of Yorker (top row)
                    N_ROWS = 6
                    N_COLS = 5

                    # split by batter hand - be robust about empty strings
                    df_lhb = df_legal[df_legal[bh_col].fillna('').astype(str).str.upper().str.startswith('L')].copy()
                    df_rhb = df_legal[df_legal[bh_col].fillna('').astype(str).str.upper().str.startswith('R')].copy()

                    count_l, dot_l, runs_l, wkt_l = build_grids(df_lhb, n_rows=N_ROWS, n_cols=N_COLS)
                    count_r, dot_r, runs_r, wkt_r = build_grids(df_rhb, n_rows=N_ROWS, n_cols=N_COLS)

                    # percent arrays (avoid div by zero)
                    perc_l = (count_l.astype(float) / count_l.sum() * 100.0) if count_l.sum() else np.zeros_like(count_l, dtype=float)
                    perc_r = (count_r.astype(float) / count_r.sum() * 100.0) if count_r.sum() else np.zeros_like(count_r, dtype=float)

                    # For LHB: display mirrored horizontally so off/leg swap visually
                    disp_count_l = np.fliplr(count_l)
                    disp_dot_l = np.fliplr(dot_l)
                    disp_runs_l = np.fliplr(runs_l)
                    disp_wkt_l = np.fliplr(wkt_l)
                    disp_perc_l = np.fliplr(perc_l)

                    disp_count_r = count_r.copy()
                    disp_dot_r = dot_r.copy()
                    disp_runs_r = runs_r.copy()
                    disp_wkt_r = wkt_r.copy()
                    disp_perc_r = perc_r.copy()

                    disp = {
                        'perc_l': disp_perc_l,
                        'dot_l': disp_dot_l,
                        'run_l': disp_runs_l,
                        'wkt_l': disp_wkt_l,
                        'perc_r': disp_perc_r,
                        'dot_r': disp_dot_r,
                        'run_r': disp_runs_r,
                        'wkt_r': disp_wkt_r
                    }

                    # xticks for RHB left->right; LHB uses reversed labels to match flipped array
                    xticks_r = ['Wide Out Off', 'Outside Off', 'On Stumps', 'Down Leg', 'Wide Down Leg']
                    xticks_l = xticks_r[::-1]
                    # yticklabels bottom->top: Short -> Back of Length -> Good -> Full -> Yorker -> Full Toss
                    yticklabels = ['Short', 'Back of Length', 'Good', 'Full', 'Yorker', 'Full Toss']

                    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
                    plt.suptitle(f"{bowler_selected} — Pitchmaps vs LHB / RHB", fontsize=16, weight='bold')

                    plot_defs = [
                        ('perc_l', '% of balls vs LHB', 'Blues', xticks_l),
                        ('perc_r', '% of balls vs RHB', 'Blues', xticks_r),
                        ('dot_l', 'Dot balls vs LHB', 'Blues', xticks_l),
                        ('dot_r', 'Dot balls vs RHB', 'Blues', xticks_r),
                        ('run_l', 'Runs conceded vs LHB', 'Reds', xticks_l),
                        ('run_r', 'Runs conceded vs RHB', 'Reds', xticks_r),
                    ]

                    for ax, (k, title, cmap, xt) in zip(axes.flat, plot_defs):
                        im = ax.imshow(disp[k], origin='lower', cmap=cmap)
                        ax.set_title(title)
                        ax.set_xticks(range(N_COLS))
                        ax.set_yticks(range(N_ROWS))
                        ax.set_xticklabels(xt, rotation=45, ha='right')
                        ax.set_yticklabels(yticklabels)

                        # -----------------------------
                        # LIGHT / CLEAR CELL BORDERS: use black color (as requested)
                        # -----------------------------
                        ax.set_xticks(np.arange(-0.5, N_COLS, 1), minor=True)
                        ax.set_yticks(np.arange(-0.5, N_ROWS, 1), minor=True)
                        ax.grid(which='minor', color='black', linewidth=0.6, alpha=0.6)
                        ax.tick_params(which='minor', bottom=False, left=False)

                        # Annotate wickets only (N W) if >0
                        if k.endswith('_l'):
                            wkt_grid = disp['wkt_l']
                        else:
                            wkt_grid = disp['wkt_r']

                        for i in range(N_ROWS):
                            for j in range(N_COLS):
                                if int(wkt_grid[i, j]) > 0:
                                    ax.text(j, i, f"{int(wkt_grid[i, j])} W",
                                            ha='center', va='center',
                                            fontsize=14, weight='bold',
                                            color='gold',
                                            bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))

                        # colorbar
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

                    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

                    try:
                        st.pyplot(fig, clear_figure=True)
                    except Exception:
                        plt.show()
                    finally:
                        plt.close(fig)

    







# SEARCH FOR THIS LINE IN YOUR FILE:
# st.header("Strength and Weakness Analysis")
## new code starts here
elif sidebar_option == "Strength vs Weakness":
    st.header("Strength vs Weakness Analysis")
    # strength_weakness_streamlit.py
    # strength_weakness_streamlit_broadcast_wkt_fix.py
    import streamlit as st
    import pandas as pd
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    from io import BytesIO
    import base64
    from PIL import Image
    import warnings
    warnings.filterwarnings("ignore")
    # NOTE: Do NOT call st.set_page_config() here — that must be the first Streamlit command in the entry file.

    # ---------- Display sizes (tweakable) ----------
    HEIGHT_WAGON_PX = 640
    HEIGHT_PITCHMAP_PX = 880

    # ---------- HTML embed helper (renders matplotlib figure at fixed pixel height) ----------
    def display_figure_fixed_height_html(fig, height_px=800, bg='white', margin_px=0):
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=200, facecolor=fig.get_facecolor())
        buf.seek(0)
        img = Image.open(buf).convert('RGBA')
        if bg is not None:
            bg_img = Image.new('RGB', img.size, bg)
            bg_img.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
            img = bg_img
        out_buf = BytesIO()
        img.save(out_buf, format='PNG')
        b64 = base64.b64encode(out_buf.getvalue()).decode('ascii')
        html = (
            f'<img src="data:image/png;base64,{b64}" '
            f'style="height:{int(height_px)}px; max-width:100%; width:auto; display:block; margin:{margin_px}px auto;" />'
        )
        st.markdown(f'<div style="max-width:100%; padding:0; margin:0;">{html}</div>', unsafe_allow_html=True)
        plt.close(fig)

    # ---------- Small helpers ----------
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

    # Helper: robust detection of legal delivery
    def is_legal_ball_from_row(row):
        """
        Determine whether a delivery is a legal ball.
        Works defensively against different dataset schemas.
        """
        # Prefer explicit flag if present
        if 'is_legal' in row.index:
            try:
                return int(row['is_legal']) == 1
            except Exception:
                pass

        # Wides / No-balls logic
        wides_cols = ['wides', 'wide', 'n_wides']
        noball_cols = ['noballs', 'no_ball', 'no-balls', 'n_noballs']

        wides_val = None
        noball_val = None

        for c in wides_cols:
            if c in row.index:
                wides_val = row[c]
                break
        for c in noball_cols:
            if c in row.index:
                noball_val = row[c]
                break

        if wides_val is not None or noball_val is not None:
            try:
                w = int(wides_val) if wides_val is not None else 0
            except Exception:
                w = 0
            try:
                nb = int(noball_val) if noball_val is not None else 0
            except Exception:
                nb = 0
            return (w == 0) and (nb == 0)

        # extras_type fallback
        if 'extras_type' in row.index:
            et = str(row.get('extras_type', '')).lower()
            if et in ['wide', 'wides', 'no ball', 'noball', 'nb', 'no-ball']:
                return False

        # If nothing else, assume legal (safer for selection of first N faced balls)
        return True

    # ---------- Work on a copy ----------
    bdf = as_dataframe(df)
    RAA_data=bdf.copy()

    # ---------- Column names (explicitly mapped to your provided schema) ----------
    COL_BAT = 'bat'  # batsman column
    COL_BOWL = 'bowl'  # bowler column
    COL_BAT_HAND = 'bat_hand'  # batting handedness (LHB/RHB)
    COL_BOWL_KIND = 'bowl_kind'  # pace/spin
    COL_BOWL_STYLE = 'bowl_style'  # specific style
    COL_RUNS = 'batruns'  # batsman runs per ball
    COL_WAGON_ZONE = 'wagonZone'  # wagon zone id (1..8)
    COL_LINE = 'line'  # line (WIDE_OUTSIDE_OFFSTUMP etc.)
    COL_LENGTH = 'length'  # length (SHORT, GOOD_LENGTH, etc.)
    COL_OUT = 'out'  # out flag (0/1)
    COL_DISMISSAL = 'dismissal'  # dismissal text
    COL_PHASE = 'PHASE'
    COL_MATCH = 'p_match'  # match id column used for grouping

    # sanity checks
    if COL_BAT not in bdf.columns or COL_BOWL not in bdf.columns:
        st.error(f"Expected columns '{COL_BAT}' and '{COL_BOWL}' in the DataFrame.")
        st.stop()

    # ---------- UI header and player selection ----------
    role = st.selectbox("Select Role", ["Batting", "Bowling"], index=0)
    if role == "Batting":
        players = sorted([x for x in bdf[COL_BAT].dropna().unique() if str(x).strip() not in ("", "0")])
    else:
        players = sorted([x for x in bdf[COL_BOWL].dropna().unique() if str(x).strip() not in ("", "0")])
    if not players:
        st.error("No players found in the chosen role column.")
        st.stop()
    player_selected = st.selectbox("Search for a player", players, index=0)

    # ---------- PHASE SELECTBOX RIGHT HERE (AFTER ROLE AND PLAYER) ----------
    phase_opts = ['Overall', 'Powerplay', 'Middle 1', 'Middle 2', 'Death']
    chosen_phase = st.selectbox("Phase", options=phase_opts, index=0)  # Default Overall

    # ---------- OPTIONAL: FIRST 10 BALLS CHECKBOX (only shown for Batting role) ----------
    first_10_balls_only = False
    if role == "Batting":
        first_10_balls_only = st.checkbox("First 10 balls", value=False)

    # ---------- Prepare player frame ----------
    if role == "Batting":
        pf = bdf[bdf[COL_BAT] == player_selected].copy()
    else:
        pf = bdf[bdf[COL_BOWL] == player_selected].copy()
    if pf.empty:
        st.info("No ball-by-ball rows for the selected player.")
        st.stop()

    # normalize runs and flags
    if role == "Batting":
        if COL_RUNS in pf.columns:
            pf[COL_RUNS] = pd.to_numeric(pf[COL_RUNS], errors='coerce').fillna(0).astype(int)
        else:
            alt = safe_get_col(pf, ['score', 'runs'])
            if alt:
                pf[COL_RUNS] = pd.to_numeric(pf[alt], errors='coerce').fillna(0).astype(int)
            else:
                pf[COL_RUNS] = 0
    else:
        # For Bowling, use 'bowlruns' or similar
        COL_RUNS_BOWL = safe_get_col(pf, ['bowlruns', 'score', 'runs'])
        if COL_RUNS_BOWL:
            pf[COL_RUNS_BOWL] = pd.to_numeric(pf[COL_RUNS_BOWL], errors='coerce').fillna(0).astype(int)
        else:
            pf['bowlruns'] = 0  # Placeholder
            COL_RUNS_BOWL = 'bowlruns'

    if COL_OUT in pf.columns:
        pf['out_flag'] = pd.to_numeric(pf[COL_OUT], errors='coerce').fillna(0).astype(int)
    else:
        pf['out_flag'] = 0

    if COL_DISMISSAL in pf.columns:
        pf['dismissal_clean'] = pf[COL_DISMISSAL].astype(str).str.lower().str.strip().replace({'nan': '', 'none': ''})
    else:
        pf['dismissal_clean'] = ''

    # ---------- CORRECTED WICKET LOGIC ----------
    WICKET_TYPES = [
        'bowled',
        'caught',
        'hit wicket',
        'stumped',
        'leg before wicket',
        'lbw'
    ]

    def is_bowler_wicket(out_flag_val, dismissal_text):
        try:
            if int(out_flag_val) != 1:
                return False
        except Exception:
            if not out_flag_val:
                return False
        if not dismissal_text or str(dismissal_text).strip() == '':
            return False
        dd = str(dismissal_text).lower()
        for token in WICKET_TYPES:
            if token in dd:
                return True
        return False

    pf['is_wkt'] = pf.apply(lambda r: 1 if is_bowler_wicket(r.get('out_flag', 0), r.get('dismissal_clean', '')) else 0, axis=1)

    # boundary flag (for batting, for bowling it's boundaries conceded)
    # use get to handle missing run columns gracefully
    runs_col = COL_RUNS if role == "Batting" else safe_get_col(pf, ['bowlruns', 'runs', 'score'], default=None)
    if runs_col and runs_col in pf.columns:
        pf['is_boundary'] = pf[runs_col].isin([4, 6]).astype(int)
    else:
        pf['is_boundary'] = 0

    # detect LHB for mirroring
    is_lhb = False
    if COL_BAT_HAND in pf.columns:
        nonull = pf[COL_BAT_HAND].dropna()
        if not nonull.empty and str(nonull.iloc[0]).strip().upper().startswith('L'):
            is_lhb = True

    # ---------- PHASE CALCULATION (run only once if not already present) ----------
    def assign_phase(over):
        if pd.isna(over):
            return 'Unknown'
        try:
            over_int = int(float(over))  # handle float or string
            if 1 <= over_int <= 6:
                return 'Powerplay'
            elif 7 <= over_int <= 11:
                return 'Middle 1'
            elif 12 <= over_int <= 16:
                return 'Middle 2'
            elif 17 <= over_int <= 20:
                return 'Death'
            else:
                return 'Unknown'
        except:
            return 'Unknown'

    for df_iter in [pf, bdf]:
        if 'PHASE' not in df_iter.columns:
            if 'over' in df_iter.columns:
                df_iter['PHASE'] = df_iter['over'].apply(assign_phase)
            else:
                df_iter['PHASE'] = 'Unknown'  # fallback if no over column
        df_iter['PHASE'] = df_iter['PHASE'].astype(str)

    # ---------- FILTER PF AND BDF BY CHOSEN PHASE (AFFECTS ALL DOWNSTREAM) ----------
    if chosen_phase != 'Overall':
        pf = pf[pf['PHASE'] == chosen_phase].copy()
        bdf = bdf[bdf['PHASE'] == chosen_phase].copy()
        # bdf = bdf[bdf['PHASE'] == chosen_phase].copy()

    # ---------- FIRST 10 LEGAL BALLS FILTER (PER MATCH, PER BATTER) ----------
    if role == "Batting" and first_10_balls_only:
        if COL_MATCH not in pf.columns:
            st.error("Column 'p_match' is required for First 10 Balls filter.")
            st.stop()

        # mark legal deliveries in the player frame (pf)
        pf = pf.copy()
        pf['is_legal_ball'] = pf.apply(is_legal_ball_from_row, axis=1)
        bdf['is_legal_ball'] = bdf.apply(is_legal_ball_from_row, axis=1)

        # prepare ordering columns to get chronological order within each match
        ordering_cols = []
        if 'innings' in pf.columns:
            ordering_cols.append('innings')
        if 'over' in pf.columns:
            ordering_cols.append('over')
        # common ball fields - try to use them if present
        if 'ball' in pf.columns:
            ordering_cols.append('ball')
        elif 'ball_in_over' in pf.columns:
            ordering_cols.append('ball_in_over')
        elif 'ball_id' in pf.columns:
            ordering_cols.append('ball_id')

        # Collect indices to keep: first 10 legal deliveries per p_match where this batter is on strike
        keep_indices = []
        # Group by match - preserve original order where possible
        for match_id, group in pf.groupby(COL_MATCH, sort=False):
            if ordering_cols:
                # stable sort to preserve earlier relative ordering where not specified by columns
                try:
                    group_sorted = group.sort_values(by=ordering_cols, kind='stable')
                except Exception:
                    group_sorted = group
            else:
                group_sorted = group

            legal_deliveries = group_sorted[group_sorted['is_legal_ball'] == True]
            first10 = legal_deliveries.head(10)
            keep_indices.extend(first10.index.tolist())

        # If no deliveries match the criteria, warn and stop gracefully
        if not keep_indices:
            st.warning("No legal deliveries found for the selected player under the applied filters.")
            st.stop()

        # Filter pf to only those first10 deliveries
        pf = pf.loc[keep_indices].copy()

        # For downstream visuals that might rely on bdf context, filter bdf to the same delivery rows (these are deliveries where the batter faced)
        # This keeps only rows corresponding to the selected batter's first-10 legal balls per match.
        bdf = bdf.loc[bdf.index.isin(keep_indices)].copy()

    # ---------- At this point, `pf` contains the player deliveries (possibly first 10 legal balls per match),
    # ---------- and `bdf` contains the matching deliveries (same indices) if First 10 filter was applied.
    # Continue with your existing plotting/analysis code below using `pf` and `bdf`.

# elif sidebar_option == "Strength vs Weakness":
#     st.header("Strength vs Weakness Analysis")
#     # strength_weakness_streamlit.py
#     # strength_weakness_streamlit_broadcast_wkt_fix.py
#     import streamlit as st
#     import pandas as pd
#     import numpy as np
#     import math
#     import matplotlib.pyplot as plt
#     from matplotlib.patches import Circle, Rectangle
#     from io import BytesIO
#     import base64
#     from PIL import Image
#     import warnings

#     warnings.filterwarnings("ignore")
#     st.set_page_config(layout="wide")

#     # ---------- Display sizes (tweakable) ----------
#     HEIGHT_WAGON_PX = 640
#     HEIGHT_PITCHMAP_PX = 880

#     # ---------- HTML embed helper (renders matplotlib figure at fixed pixel height) ----------
#     def display_figure_fixed_height_html(fig, height_px=800, bg='white', margin_px=0):
#         buf = BytesIO()
#         fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=200, facecolor=fig.get_facecolor())
#         buf.seek(0)
#         img = Image.open(buf).convert('RGBA')
#         if bg is not None:
#             bg_img = Image.new('RGB', img.size, bg)
#             bg_img.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
#             img = bg_img
#         out_buf = BytesIO()
#         img.save(out_buf, format='PNG')
#         b64 = base64.b64encode(out_buf.getvalue()).decode('ascii')
#         html = (
#             f'<img src="data:image/png;base64,{b64}" '
#             f'style="height:{int(height_px)}px; max-width:100%; width:auto; display:block; margin:{margin_px}px auto;" />'
#         )
#         st.markdown(f'<div style="max-width:100%; padding:0; margin:0;">{html}</div>', unsafe_allow_html=True)
#         plt.close(fig)

#     # ---------- Small helpers ----------
#     def as_dataframe(x):
#         if isinstance(x, pd.Series):
#             return x.to_frame().T.reset_index(drop=True)
#         elif isinstance(x, pd.DataFrame):
#             return x.copy()
#         else:
#             return pd.DataFrame(x)

#     def safe_get_col(df_local, candidates, default=None):
#         for c in candidates:
#             if c in df_local.columns:
#                 return c
#         return default


#     # work on a copy
#     bdf = as_dataframe(df)

#     # ---------- Column names (explicitly mapped to your provided schema) ----------
#     COL_BAT = 'bat' # batsman column
#     COL_BOWL = 'bowl' # bowler column
#     COL_BAT_HAND = 'bat_hand' # batting handedness (LHB/RHB)
#     COL_BOWL_KIND = 'bowl_kind' # pace/spin
#     COL_BOWL_STYLE = 'bowl_style' # specific style
#     COL_RUNS = 'batruns' # batsman runs per ball
#     COL_WAGON_ZONE = 'wagonZone' # wagon zone id (1..8)
#     COL_LINE = 'line' # line (WIDE_OUTSIDE_OFFSTUMP etc.)
#     COL_LENGTH = 'length' # length (SHORT, GOOD_LENGTH, etc.)
#     COL_OUT = 'out' # out flag (0/1)
#     COL_DISMISSAL = 'dismissal' # dismissal text
#     COL_PHASE = 'PHASE'

#     # sanity checks
#     if COL_BAT not in bdf.columns or COL_BOWL not in bdf.columns:
#         st.error(f"Expected columns '{COL_BAT}' and '{COL_BOWL}' in the DataFrame.")
#         st.stop()

#     # ---------- UI header and player selection ----------
#     # st.title("Strength & Weakness — Broadcast View")
#     role = st.selectbox("Select Role", ["Batting", "Bowling"], index=0)

#     if role == "Batting":
#         players = sorted([x for x in bdf[COL_BAT].dropna().unique() if str(x).strip() not in ("", "0")])
#     else:
#         players = sorted([x for x in bdf[COL_BOWL].dropna().unique() if str(x).strip() not in ("", "0")])

#     if not players:
#         st.error("No players found in the chosen role column.")
#         st.stop()

#     player_selected = st.selectbox("Search for a player", players, index=0)

#     # ---------- PHASE SELECTBOX RIGHT HERE (AFTER ROLE AND PLAYER) ----------
#     phase_opts = ['Overall', 'Powerplay', 'Middle 1', 'Middle 2', 'Death']
#     chosen_phase = st.selectbox("Phase", options=phase_opts, index=0) # Default Overall

#     # ---------- Prepare player frame ----------
#     if role == "Batting":
#         pf = bdf[bdf[COL_BAT] == player_selected].copy()
#     else:
#         pf = bdf[bdf[COL_BOWL] == player_selected].copy()

#     if pf.empty:
#         st.info("No ball-by-ball rows for the selected player.")
#         st.stop()

#     # normalize runs and flags
#     if role == "Batting":
#         if COL_RUNS in pf.columns:
#             pf[COL_RUNS] = pd.to_numeric(pf[COL_RUNS], errors='coerce').fillna(0).astype(int)
#         else:
#             alt = safe_get_col(pf, ['score','runs'])
#             if alt:
#                 pf[COL_RUNS] = pd.to_numeric(pf[alt], errors='coerce').fillna(0).astype(int)
#             else:
#                 pf[COL_RUNS] = 0
#     else:
#         # For Bowling, use 'bowlruns' or similar
#         COL_RUNS_BOWL = safe_get_col(pf, ['bowlruns', 'score', 'runs'])
#         if COL_RUNS_BOWL:
#             pf[COL_RUNS_BOWL] = pd.to_numeric(pf[COL_RUNS_BOWL], errors='coerce').fillna(0).astype(int)
#         else:
#             pf['bowlruns'] = 0 # Placeholder
#             COL_RUNS_BOWL = 'bowlruns'

#     if COL_OUT in pf.columns:
#         pf['out_flag'] = pd.to_numeric(pf[COL_OUT], errors='coerce').fillna(0).astype(int)
#     else:
#         pf['out_flag'] = 0

#     if COL_DISMISSAL in pf.columns:
#         pf['dismissal_clean'] = pf[COL_DISMISSAL].astype(str).str.lower().str.strip().replace({'nan':'','none':''})
#     else:
#         pf['dismissal_clean'] = ''

#     # ---------- CORRECTED WICKET LOGIC ----------
#     # Only these dismissal types count as bowler wickets:
#     WICKET_TYPES = [
#         'bowled',
#         'caught',
#         'hit wicket',
#         'stumped',
#         'leg before wicket', # full name
#         'lbw' # common abbreviation
#     ]

#     def is_bowler_wicket(out_flag_val, dismissal_text):
#         """
#         Return True if the delivery is credited as a bowler wicket:
#         - out_flag (truthy) AND dismissal contains one of the accepted wicket tokens.
#         """
#         try:
#             if int(out_flag_val) != 1:
#                 return False
#         except Exception:
#             # non-numeric: treat falsy
#             if not out_flag_val:
#                 return False
#         if not dismissal_text or str(dismissal_text).strip() == '':
#             return False
#         dd = str(dismissal_text).lower()
#         # check any wicket token is present as substring
#         for token in WICKET_TYPES:
#             if token in dd:
#                 return True
#         return False

#     pf['is_wkt'] = pf.apply(lambda r: 1 if is_bowler_wicket(r.get('out_flag',0), r.get('dismissal_clean','')) else 0, axis=1)

#     # boundary flag (for batting, for bowling it's boundaries conceded)
#     pf['is_boundary'] = pf.get(COL_RUNS if role == "Batting" else COL_RUNS_BOWL, 0).isin([4,6]).astype(int)

#     # detect LHB for mirroring
#     is_lhb = False
#     if COL_BAT_HAND in pf.columns:
#         nonull = pf[COL_BAT_HAND].dropna()
#         if not nonull.empty and str(nonull.iloc[0]).strip().upper().startswith('L'):
#             is_lhb = True

#     # ---------- PHASE CALCULATION (run only once if not already present) ----------
#     def assign_phase(over):
#         if pd.isna(over):
#             return 'Unknown'
#         try:
#             over_int = int(float(over)) # handle float or string
#             if 1 <= over_int <= 6:
#                 return 'Powerplay'
#             elif 7 <= over_int <= 11:
#                 return 'Middle 1'
#             elif 12 <= over_int <= 16:
#                 return 'Middle 2'
#             elif 17 <= over_int <= 20:
#                 return 'Death'
#             else:
#                 return 'Unknown'
#         except:
#             return 'Unknown'

#     for df in [pf, bdf]:
#         if 'PHASE' not in df.columns:
#             if 'over' in df.columns:
#                 df['PHASE'] = df['over'].apply(assign_phase)
#             else:
#                 df['PHASE'] = 'Unknown' # fallback if no over column
#         df['PHASE'] = df['PHASE'].astype(str)

#     # ---------- FILTER PF AND BDF BY CHOSEN PHASE (AFFECTS ALL DOWNSTREAM) ----------
#     if chosen_phase != 'Overall':
#         pf = pf[pf['PHASE'] == chosen_phase].copy()
#         bdf = bdf[bdf['PHASE'] == chosen_phase].copy()
    # Now all computations below use the phase-filtered pf/bdf

    # ---------- Metrics helpers ----------
    def compute_batting_metrics(gdf, run_col=COL_RUNS):
        runs = int(gdf[run_col].sum())
        balls = int(gdf.shape[0])
        fours = int((gdf[run_col] == 4).sum())
        sixes = int((gdf[run_col] == 6).sum())
        wkts = int(gdf['is_wkt'].sum()) if 'is_wkt' in gdf.columns else 0
        avg = (runs / wkts) if wkts>0 else np.nan
        sr = (runs / balls * 100) if balls>0 else np.nan
        bound_pct = ((fours + sixes) / balls * 100) if balls>0 else 0.0
        return {'Runs': runs, 'Balls': balls, '4s': fours, '6s': sixes, 'Dismissals': wkts,
                'Avg': np.round(avg,2) if not np.isnan(avg) else '-', 'SR': np.round(sr,2) if not np.isnan(sr) else '-', 'Bound%': f"{bound_pct:.2f}%"}

    def compute_bowling_metrics(gdf, run_col):
        runs = int(gdf[run_col].sum())
        balls = int(((gdf.get('noball',0).fillna(0).astype(int) == 0) & (gdf.get('wide',0).fillna(0).astype(int) == 0)).sum())
        wkts = int(gdf['is_wkt'].sum()) if 'is_wkt' in gdf.columns else 0
        econ = (runs * 6.0 / balls) if balls>0 else np.nan
        avg = (runs / wkts) if wkts>0 else np.nan
        sr = (balls / wkts) if wkts>0 else np.nan
        fours = int((gdf[run_col] == 4).sum())
        sixes = int((gdf[run_col] == 6).sum())
        bound_pct = ((fours + sixes) / balls * 100) if balls>0 else 0.0
        return {'Runs': runs, 'Balls': balls, 'Wkts': wkts, 'Econ': np.round(econ,2) if not np.isnan(econ) else '-', 'Avg': np.round(avg,2) if not np.isnan(avg) else '-', 'SR': np.round(sr,2) if not np.isnan(sr) else '-', 'Bound%': f"{bound_pct:.2f}%"}

    # ---------- Visual building blocks (unchanged) ----------
    sector_names = {
        1: "Third Man", 2: "Point", 3: "Covers", 4: "Mid Off",
        5: "Mid On", 6: "Mid Wicket", 7: "Square Leg", 8: "Fine Leg"
    }
    base_angles = {1: 112.5, 2:157.5, 3:202.5, 4:247.5, 5:292.5, 6:337.5, 7:22.5, 8:67.5}
    def sector_angle_rad(zone, is_lhb_local):
        a = float(base_angles.get(int(zone), 0.0))
        if is_lhb_local:
            a = (180.0 - a) % 360.0
        return math.radians(a)

    def draw_wagon(df_local, label, is_lhb_local):
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_aspect('equal'); ax.axis('off')
        ax.add_patch(Circle((0,0),1, fill=True, color='#1b8f1b', alpha=1))
        ax.add_patch(Circle((0,0),1, fill=False, color='black', linewidth=3))
        ax.add_patch(Circle((0,0),0.52, fill=True, color='#66bb6a'))
        ax.add_patch(Circle((0,0),0.52, fill=False, color='white', linewidth=1))
        ax.add_patch(Rectangle((-0.04, -0.08), 0.08, 0.16, color='tan', alpha=1, zorder=8))
        for a in np.linspace(0,2*np.pi,9)[:-1]:
            ax.plot([0, math.cos(a)], [0, math.sin(a)], color='white', alpha=0.25, linewidth=1)
        zcol = COL_WAGON_ZONE if COL_WAGON_ZONE in df_local.columns else safe_get_col(df_local, ['wagon_zone','wagon_zone_id'])
        if zcol is None:
            ax.text(0,0, "No wagonZone column", ha='center', va='center', color='white')
            return fig
        tmp = df_local.copy()
        tmp['zone_int'] = pd.to_numeric(tmp[zcol], errors='coerce').astype('Int64')
        runs_by_zone = tmp.groupby('zone_int')[COL_RUNS if role == "Batting" else COL_RUNS_BOWL].sum().to_dict()
        fours_by_zone = tmp.groupby('zone_int')[COL_RUNS if role == "Batting" else COL_RUNS_BOWL].apply(lambda s: int((s==4).sum())).to_dict()
        sixes_by_zone = tmp.groupby('zone_int')[COL_RUNS if role == "Batting" else COL_RUNS_BOWL].apply(lambda s: int((s==6).sum())).to_dict()
        total_runs = sum(int(v) for v in runs_by_zone.values())
        for zone in range(1,9):
            ang = sector_angle_rad(zone, is_lhb_local)
            x = 0.60*math.cos(ang)
            y = 0.60*math.sin(ang)
            runs = int(runs_by_zone.get(zone,0))
            pct = (runs/total_runs*100) if total_runs>0 else 0.0
            ax.text(x, y+0.03, f"{pct:.2f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=14)
            ax.text(x, y-0.03, f"{runs} runs", ha='center', va='center', color='white', fontsize=10)
            ax.text(x, y-0.12, f"4s:{fours_by_zone.get(zone,0)} 6s:{sixes_by_zone.get(zone,0)}", ha='center', va='center', color='white', fontsize=8)
            sx = 0.80*math.cos(ang); sy = 0.80*math.sin(ang)
            ax.text(sx, sy, sector_names[zone], ha='center', va='center', color='white', fontsize=8)
        ax.set_xlim(-1.2,1.2); ax.set_ylim(-1.2,1.2)
        plt.tight_layout(pad=0)
        fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99)
        return fig

    LINE_MAP = {
        'WIDE_OUTSIDE_OFFSTUMP': 0,
        'OUTSIDE_OFFSTUMP': 1,
        'ON_THE_STUMPS': 2,
        'DOWN_LEG': 3,
        'WIDE_DOWN_LEG': 4
    }
    LENGTH_MAP = {
        'SHORT': 0,
        'SHORT_OF_A_GOOD_LENGTH': 1,
        'GOOD_LENGTH': 2,
        'FULL': 3,
        'YORKER': 4,
        'FULL_TOSS': 5
    }
    line_map = LINE_MAP
    length_map = LENGTH_MAP

    def build_boundaries_grid_local(df_local):
        grid = np.zeros((6,5), dtype=int)
        if df_local.shape[0] == 0: return grid
        if COL_LINE not in df_local.columns or COL_LENGTH not in df_local.columns: return grid
        plot_df = df_local[[COL_LINE, COL_LENGTH, COL_RUNS if role == "Batting" else COL_RUNS_BOWL]].dropna(subset=[COL_LINE, COL_LENGTH])
        for _, r in plot_df.iterrows():
            li = LINE_MAP.get(r[COL_LINE], None)
            le = LENGTH_MAP.get(r[COL_LENGTH], None)
            if li is None or le is None: continue
            try:
                runs_here = int(r[COL_RUNS if role == "Batting" else COL_RUNS_BOWL])
            except:
                runs_here = 0
            if runs_here in (4,6):
                grid[le, li] += 1
        return grid

    def build_dismissals_grid_local(df_local):
        grid = np.zeros((6,5), dtype=int)
        if df_local.shape[0] == 0: return grid
        if COL_LINE not in df_local.columns or COL_LENGTH not in df_local.columns: return grid
        plot_df = df_local[[COL_LINE, COL_LENGTH, 'is_wkt']].dropna(subset=[COL_LINE, COL_LENGTH])
        for _, r in plot_df.iterrows():
            li = LINE_MAP.get(r[COL_LINE], None)
            le = LENGTH_MAP.get(r[COL_LENGTH], None)
            if li is None or le is None: continue
            isw = int(r.get('is_wkt',0))
            if isw == 1:
                grid[le, li] += 1
        return grid

    def plot_grid_mat(grid, title, cmap='Oranges', mirror=False):
        disp = np.fliplr(grid) if mirror else grid.copy()
        xticks_base = ['Wide Out Off','Outside Off','On Stumps','Down Leg','Wide Down Leg']
        xticks = list(reversed(xticks_base)) if mirror else xticks_base
        fig, ax = plt.subplots(figsize=(6,9), dpi=150)
        im = ax.imshow(disp, origin='lower', cmap=cmap)
        ax.set_xticks(range(5)); ax.set_yticks(range(6))
        ax.set_xticklabels(xticks, rotation=40, ha='right')
        ax.set_yticklabels(['Short','Back of Length','Good','Full','Yorker', 'Full Toss'])
        for i in range(6):
            for j in range(5):
                ax.text(j, i, int(disp[i,j]), ha='center', va='center', color='black', fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        plt.tight_layout(pad=0)
        fig.subplots_adjust(top=0.99, bottom=0.01, left=0.06, right=0.99)
        return fig

    # ---------- Render Batting or Bowling views ----------
    if role == "Batting":
        st.markdown(f"<div style='font-size:20px; font-weight:800; color:#111;'> Batting — {player_selected}</div>", unsafe_allow_html=True)

        # require bdf in globals
        if 'bdf' not in globals():
            bdf = as_dataframe(df) # Use df if bdf not defined

        # build player frame pf using canonical batter column (COL_BAT expected to be 'bat')
        pf = bdf[bdf.get(COL_BAT, 'bat') == player_selected].copy()
        if pf.empty:
            st.info("No ball-by-ball rows for the selected batter.")
            st.stop()

        # run column detection (prefer 'batruns') - use safe_get_col if available
        preferred = 'batruns'
        alt = safe_get_col(bdf, ['batruns', 'batsman_runs', 'score', 'runs'])
        runs_col = preferred if preferred in bdf.columns else (alt if alt else None)
        if runs_col is None:
            st.error("No runs column found (expected 'batruns' or 'batsman_runs' or 'score' or 'runs').")
            st.stop()

        # coerce runs col
        bdf[runs_col] = pd.to_numeric(bdf.get(runs_col, 0), errors='coerce').fillna(0).astype(int)
        pf[runs_col] = pd.to_numeric(pf.get(runs_col, 0), errors='coerce').fillna(0).astype(int)

        # detect left hand for mirroring
        is_lhb = False
        if 'bat_hand' in pf.columns:
            nonull = pf['bat_hand'].dropna().astype(str)
            if not nonull.empty and nonull.iloc[0].strip().upper().startswith('L'):
                is_lhb = True

        # basic batting metrics function (uses runs_col)
        def compute_batting_metrics(gdf, run_col=runs_col):
            runs = int(gdf[run_col].sum()) if run_col in gdf.columns else 0
            balls = int(gdf.shape[0])
            fours = int((gdf[run_col] == 4).sum()) if run_col in gdf.columns else 0
            sixes = int((gdf[run_col] == 6).sum()) if run_col in gdf.columns else 0
            sr = (runs / balls * 100.0) if balls > 0 else np.nan
            return {'Runs': runs, 'Balls': balls, '4s': fours, '6s': sixes, 'SR': np.round(sr, 2) if not np.isnan(sr) else '-'}

        # ────────────────────────────────────────────────
        # FILTER PF AND BDF BY PHASE (affects all)
        # ────────────────────────────────────────────────
        if chosen_phase != 'Overall':
            pf = pf[pf['PHASE'] == chosen_phase].copy()
            bdf = bdf[bdf['PHASE'] == chosen_phase].copy()

        # ------------------ bowling type summary for this batter ------------------
        if COL_BOWL_KIND in pf.columns:
            pf[COL_BOWL_KIND] = pf[COL_BOWL_KIND].astype(str).str.lower().fillna('unknown')
            kinds = sorted(pf[COL_BOWL_KIND].dropna().unique().tolist())
        else:
            kinds = []

        rows = []
        if kinds:
            for k in kinds:
                g = pf[pf[COL_BOWL_KIND] == k]
                m = compute_batting_metrics(g)
                m['bowl_kind'] = k
                rows.append(m)
        else:
            m = compute_batting_metrics(pf)
            m['bowl_kind'] = 'unknown'
            rows.append(m)
        bk_df = pd.DataFrame(rows).set_index('bowl_kind')
        bk_df.index.name = 'Bowler Kind'

        # ------------------ bowling style summary for this batter -----------------
        if COL_BOWL_STYLE in pf.columns:
            styles = sorted([s for s in pf[COL_BOWL_STYLE].dropna().unique() if str(s).strip() != ''])
            bs_rows = []
            if styles:
                for s in styles:
                    g = pf[pf[COL_BOWL_STYLE] == s]
                    m = compute_batting_metrics(g)
                    m['bowl_style'] = s
                    bs_rows.append(m)
                bs_df = pd.DataFrame(bs_rows).set_index('bowl_style')
                bs_df.index.name = 'Bowler Style'
            else:
                bs_df = pd.DataFrame(columns=['bowl_style']).set_index('bowl_style')
                bs_df.index.name = 'Bowler Style'
        else:
            bs_df = None

        # ------------------ ensure top7_flag exists (attempt to create if absent) ------------------

        # if 'top7_flag' not in bdf.columns:
        #     bdf = bdf.copy()
        #     if 'p_bat' in bdf.columns and pd.api.types.is_numeric_dtype(bdf['p_bat']):
        #         bdf['top7_flag'] = (pd.to_numeric(bdf['p_bat'], errors='coerce').fillna(9999) <= 7).astype(int)
        #     else:
        #         # derive by first appearance (best effort)
        #         order_col = 'ball_id' if 'ball_id' in bdf.columns else ('ball' if 'ball' in bdf.columns else None)
        #         if order_col is None:
        #             bdf = bdf.reset_index().rename(columns={'index': '_order_idx'})
        #             order_col = '_order_idx'
        #         if all(c in bdf.columns for c in ['p_match', 'inns', COL_BAT]):
        #             tmp = bdf.dropna(subset=[COL_BAT, 'p_match', 'inns']).copy()
        #             first_app = tmp.groupby(['p_match', 'inns', COL_BAT], as_index=False)[order_col].min().rename(columns={order_col: 'first_ball'})
        #             top7_records = []
        #             for (m, inn), grp in first_app.groupby(['p_match', 'inns']):
        #                 top7 = grp.sort_values('first_ball').head(7)[COL_BAT].tolist()
        #                 for b in top7:
        #                     top7_records.append((m, inn, b))
        #             if top7_records:
        #                 top7_df = pd.DataFrame(top7_records, columns=['p_match', 'inns', COL_BAT])
        #                 top7_df['top7_flag'] = 1
        #                 bdf = bdf.merge(top7_df, how='left', on=['p_match', 'inns', COL_BAT])
        #                 bdf['top7_flag'] = bdf['top7_flag'].fillna(0).astype(int)
        #             else:
        #                 bdf['top7_flag'] = 0
        #         else:
        #             bdf['top7_flag'] = 0
        #     bdf['top7_flag'] = bdf['top7_flag'].fillna(0).astype(int)
        # def compute_RAA_DAA_for_group_column(group_col):
        #     out = {}
        #     if group_col not in bdf.columns:
        #         return out
           
        #     working = bdf.copy()
        #     working[group_col] = working[group_col].astype(str).str.lower().fillna('unknown')
        #     working[runs_col] = pd.to_numeric(working.get(runs_col, 0), errors='coerce').fillna(0).astype(int)
        #     working['out_flag_tmp'] = pd.to_numeric(working.get('out', 0), errors='coerce').fillna(0).astype(int)
        #     working['dismissal_clean_tmp'] = working.get('dismissal', "").astype(str).str.lower().str.strip().replace({'nan': '', 'none': ''})
           
        #     WICKET_TYPES = ['bowled', 'caught', 'hit wicket', 'stumped', 'leg before wicket', 'lbw']
        #     def is_bowler_wicket_local(out_flag_val, dismissal_text):
        #         try:
        #             if int(out_flag_val) != 1:
        #                 return False
        #         except:
        #             if not out_flag_val:
        #                 return False
        #         if not dismissal_text or str(dismissal_text).strip() == '':
        #             return False
        #         dd = str(dismissal_text).lower()
        #         for token in WICKET_TYPES:
        #             if token in dd:
        #                 return True
        #         return False
           
        #     working['is_wkt_tmp'] = working.apply(lambda r: 1 if is_bowler_wicket_local(r.get('out_flag_tmp', 0), r.get('dismissal_clean_tmp', '')) else 0, axis=1)
           
        #     # Assume top7_flag exists to ignore bowlers
        #     top7 = working[working.get('top7_flag', 0) == 1].copy()
        #     if top7.empty:
        #         return out # No data to benchmark
           
        #     # NEW: Add position groups (from your earlier logic)
        #     # Assume 'final_batting_position' is already computed in cumulator
        #     if 'final_batting_position' not in top7.columns:
        #         return out # Need positions for grouping
           
        #     def get_group(pos):
        #         if pos in [1, 2]:
        #             return 'Openers'
        #         elif pos in [3]:
        #             return 'Top Order'
        #         elif pos in [4, 5]:
        #             return 'Middle Order'
        #         elif pos in [6, 7, 8]:
        #             return 'Lower Order'
        #         else:
        #             return 'Unknown'
           
        #     top7['group'] = top7['final_batting_position'].apply(get_group)
           
        #     # Compute per-match stats (runs, balls, dismissals) grouped by tournament, batsman, group
        #     gb_keys = ['tournament', 'batsman', 'match_id', 'group'] if 'tournament' in top7.columns else ['batsman', 'match_id', 'group']
        #     per_mb = top7.groupby(gb_keys, as_index=False).agg(
        #         runs=(runs_col, 'sum'),
        #         balls=(runs_col, 'count'),
        #         dismissals=('is_wkt_tmp', 'sum')
        #     )
           
        #     per_mb['SR'] = per_mb.apply(lambda r: (r['runs'] / r['balls'] * 100.0) if r['balls'] > 0 else np.nan, axis=1)
        #     per_mb['BPD'] = per_mb.apply(lambda r: (r['balls'] / r['dismissals']) if r['dismissals'] > 0 else np.nan, axis=1)
           
        #     # Compute average SR/BPD per group per tournament (benchmark)
        #     avg_by_group_tournament = per_mb.groupby(['tournament', 'group']).agg(
        #         avg_SR_top7=('SR', 'mean'),
        #         avg_BPD_top7=('BPD', 'mean')
        #     ).reset_index()
           
        #     # For selected batter (sel/pf), compute per-tournament RAA/DAA
        #     sel = pf.copy()
        #     sel[group_col] = sel.get(group_col, "").astype(str).str.lower().fillna('unknown')
        #     sel[runs_col] = pd.to_numeric(sel.get(runs_col, 0), errors='coerce').fillna(0).astype(int)
        #     sel['out_flag_tmp'] = pd.to_numeric(sel.get('out', 0), errors='coerce').fillna(0).astype(int)
        #     sel['dismissal_clean_tmp'] = sel.get('dismissal', "").astype(str).str.lower().str.strip().replace({'nan': '', 'none': ''})
        #     sel['is_wkt_tmp'] = sel.apply(lambda r: 1 if is_bowler_wicket_local(r.get('out_flag_tmp', 0), r.get('dismissal_clean_tmp', '')) else 0, axis=1)
           
        #     # Add groups to sel
        #     sel['group'] = sel['final_batting_position'].apply(get_group)
           
        #     sel_grp = sel.groupby(['tournament', 'group']).agg(
        #         runs=(runs_col, 'sum'),
        #         balls=(runs_col, 'count'),
        #         dismissals=('is_wkt_tmp', 'sum')
        #     ).reset_index()
           
        #     sel_grp['SR'] = sel_grp.apply(lambda r: (r['runs'] / r['balls'] * 100.0) if r['balls'] > 0 else np.nan, axis=1)
        #     sel_grp['BPD'] = sel_grp.apply(lambda r: (r['balls'] / r['dismissals']) if r['dismissals'] > 0 else np.nan, axis=1)
           
        #     # Merge with averages and compute per-tournament RAA/DAA
        #     merged = pd.merge(sel_grp, avg_by_group_tournament, how='left', on=['tournament', 'group'])
        #     merged['RAA'] = merged['SR'] - merged['avg_SR_top7']
        #     merged['DAA'] = merged['BPD'] - merged['avg_BPD_top7']
           
        #     # Innings per tournament for weighting
        #     innings_per_tournament = merged.groupby('tournament')['innings'].sum().reset_index() # assuming innings computed earlier
           
        #     # Weighted average RAA/DAA (overall)
        #     def weighted_avg(series, weights):
        #         return np.average(series, weights=weights)
           
        #     overall_raa = weighted_avg(merged['RAA'], merged['innings'])
        #     overall_daa = weighted_avg(merged['DAA'], merged['innings'])
           
        #     # Output dict (per group or tournament if needed)
        #     for _, row in merged.iterrows():
        #         g = row[group_col]
        #         out[str(g).lower()] = {
        #             'selected_SR': row['SR'],
        #             'selected_BPD': row['BPD'],
        #             'avg_SR_top7': row['avg_SR_top7'],
        #             'avg_BPD_top7': row['avg_BPD_top7'],
        #             'RAA': row['RAA'],
        #             'DAA': row['DAA']
        #         }
           
        #     out['overall'] = {'RAA': overall_raa, 'DAA': overall_daa}
        #     return out
       
        # def _fmt(x):
        #     return f"{x:.2f}" if (not pd.isna(x)) else '-'
        # # attach to bk_df
        # if bk_df is not None and not bk_df.empty:
        #     if COL_BOWL_KIND in bdf.columns:
        #         bk_raadaa = compute_RAA_DAA_for_group_column(COL_BOWL_KIND)
        #         new_RAA = []
        #         new_DAA = []
        #         for idx in bk_df.index:
        #             key = str(idx).lower()
        #             val = bk_raadaa.get(key, {})
        #             new_RAA.append(_fmt(val.get('RAA', np.nan)))
        #             new_DAA.append(_fmt(val.get('DAA', np.nan)))
        #         bk_df['RAA'] = new_RAA
        #         bk_df['DAA'] = new_DAA
        #     else:
        #         bk_df['RAA'] = '-'
        #         bk_df['DAA'] = '-'
        # st.markdown("<div style='font-weight:700; font-size:15px;'> Performance by bowling type </div>", unsafe_allow_html=True)
        # st.dataframe(bk_df, use_container_width=True)
        # # attach to bs_df
        # if bs_df is not None:
        #     if COL_BOWL_STYLE in bdf.columns:
        #         bs_raadaa = compute_RAA_DAA_for_group_column(COL_BOWL_STYLE)
        #         new_RAA = []
        #         new_DAA = []
        #         for idx in bs_df.index:
        #             key = str(idx).lower()
        #             val = bs_raadaa.get(key, {})
        #             new_RAA.append(_fmt(val.get('RAA', np.nan)))
        #             new_DAA.append(_fmt(val.get('DAA', np.nan)))
        #         bs_df['RAA'] = new_RAA
        #         bs_df['DAA'] = new_DAA
        #     else:
        #         bs_df['RAA'] = '-'
        #         bs_df['DAA'] = '-'
        #     st.markdown("<div style='font-weight:700; font-size:15px;'> Performance by bowling style </div>", unsafe_allow_html=True)
        #     st.dataframe(bs_df, use_container_width=True)

        # ------------------ ensure top7_flag exists (attempt to create if absent) ------------------

        if 'top7_flag' not in bdf.columns:
            bdf = bdf.copy()
            if 'p_bat' in bdf.columns and pd.api.types.is_numeric_dtype(bdf['p_bat']):
                bdf['top7_flag'] = (pd.to_numeric(bdf['p_bat'], errors='coerce').fillna(9999) <= 7).astype(int)
            else:
                # derive by first appearance (best effort)
                order_col = 'ball_id' if 'ball_id' in bdf.columns else ('ball' if 'ball' in bdf.columns else None)
                if order_col is None:
                    bdf = bdf.reset_index().rename(columns={'index': '_order_idx'})
                    order_col = '_order_idx'
                if all(c in bdf.columns for c in ['p_match', 'inns', COL_BAT]):
                    tmp = bdf.dropna(subset=[COL_BAT, 'p_match', 'inns']).copy()
                    first_app = tmp.groupby(['p_match', 'inns', COL_BAT], as_index=False)[order_col].min().rename(columns={order_col: 'first_ball'})
                    top7_records = []
                    for (m, inn), grp in first_app.groupby(['p_match', 'inns']):
                        top7 = grp.sort_values('first_ball').head(7)[COL_BAT].tolist()
                        for b in top7:
                            top7_records.append((m, inn, b))
                    if top7_records:
                        top7_df = pd.DataFrame(top7_records, columns=['p_match', 'inns', COL_BAT])
                        top7_df['top7_flag'] = 1
                        bdf = bdf.merge(top7_df, how='left', on=['p_match', 'inns', COL_BAT])
                        bdf['top7_flag'] = bdf['top7_flag'].fillna(0).astype(int)
                    else:
                        bdf['top7_flag'] = 0
                else:
                    bdf['top7_flag'] = 0
            bdf['top7_flag'] = bdf['top7_flag'].fillna(0).astype(int)


          
        # ------------------ compute RAA / DAA (robustly) ------------------
        def compute_RAA_DAA_for_group_column(group_col):
            """
            Compute RAA (Runs Above Average) and DAA (Dismissals Above Average) for a given group column.
            
            CHANGES MADE:
            1. Added dismissals calculation using wkt_tokens = {'caught', 'bowled', 'stumped', 'lbw','leg before wicket','hit wicket'}
            2. Added AVG (runs per dismissal)
            3. Added BPB (balls per boundary, where boundary = 4 or 6)
            4. Changed average calculation to WEIGHTED average based on runs scored:
               - Weighted SR = Σ(runs_i * SR_i) / Σ(runs_i)
               - Weighted BPD = Σ(runs_i * BPD_i) / Σ(runs_i)
               - Weighted AVG = Σ(runs_i * AVG_i) / Σ(runs_i)
               - Weighted BPB = Σ(runs_i * BPB_i) / Σ(runs_i)
            """
            out = {}
            if group_col not in bdf.columns:
                return out
        
            working = bdf.copy()
            working[group_col] = working[group_col].astype(str).str.lower().fillna('unknown')
            working[runs_col] = pd.to_numeric(working.get(runs_col, 0), errors='coerce').fillna(0).astype(int)
            working['out_flag_tmp'] = pd.to_numeric(working.get('out', 0), errors='coerce').fillna(0).astype(int)
            working['dismissal_clean_tmp'] = working.get('dismissal', "").astype(str).str.lower().str.strip().replace({'nan': '', 'none': ''})
        
            # CHANGE 1: Use specific wicket tokens for dismissals
            WKT_TOKENS = {'caught', 'bowled', 'stumped', 'lbw','leg before wicket','hit wicket'}
            
            def is_bowler_wicket_local(out_flag_val, dismissal_text):
                try:
                    if int(out_flag_val) != 1:
                        return False
                except:
                    if not out_flag_val:
                        return False
                if not dismissal_text or str(dismissal_text).strip() == '':
                    return False
                dd = str(dismissal_text).lower()
                # Check if any wicket token is in dismissal text
                for token in WKT_TOKENS:
                    if token in dd:
                        return True
                return False
        
            working['is_wkt_tmp'] = working.apply(
                lambda r: 1 if is_bowler_wicket_local(r.get('out_flag_tmp', 0), r.get('dismissal_clean_tmp', '')) else 0, 
                axis=1
            )
            
            # CHANGE 2: Calculate boundaries (4s and 6s)
            working['is_boundary'] = working[runs_col].isin([4, 6]).astype(int)
        
            # Get top 7 batters
            top7 = working[working.get('top7_flag', 0) == 1].copy()
        
            if top7.empty:
                if 'p_bat' in working.columns and pd.api.types.is_numeric_dtype(working['p_bat']):
                    top7 = working[pd.to_numeric(working['p_bat'], errors='coerce').fillna(9999) <= 7].copy()
        
            if top7.empty:
                if all(c in working.columns for c in ['p_match', 'inns', COL_BAT]):
                    tmp = working.dropna(subset=[COL_BAT, 'p_match', 'inns']).copy()
                    order_col = 'ball_id' if 'ball_id' in tmp.columns else ('ball' if 'ball' in tmp.columns else tmp.columns[0])
                    first_app = tmp.groupby(['p_match', 'inns', COL_BAT], as_index=False)[order_col].min().rename(columns={order_col: 'first_ball'})
                    recs = []
                    for (m, inn), grp in first_app.groupby(['p_match', 'inns']):
                        top7_names = grp.sort_values('first_ball').head(7)[COL_BAT].tolist()
                        for b in top7_names:
                            recs.append((m, inn, b))
                    if recs:
                        top7_df = pd.DataFrame(recs, columns=['p_match', 'inns', COL_BAT])
                        top7_df['top7_flag'] = 1
                        top7 = working.merge(top7_df, how='inner', on=['p_match', 'inns', COL_BAT])
            
            if top7.empty:
                return out
        
            # Aggregate per match/batter/group
            gb_keys = ['p_match', 'inns', COL_BAT, group_col] if all(c in top7.columns for c in ['p_match', 'inns']) else ['p_match', COL_BAT, group_col]
            
            per_mb = (top7.groupby(gb_keys, as_index=False)
                      .agg(
                          runs=(runs_col, 'sum'),
                          balls=(runs_col, 'count'),
                          dismissals=('is_wkt_tmp', 'sum'),
                          boundaries=('is_boundary', 'sum')  # CHANGE 3: Add boundaries
                      ))
            
            # CHANGE 4: Calculate all metrics including AVG and BPB
            per_mb['SR'] = per_mb.apply(lambda r: (r['runs'] / r['balls'] * 100.0) if r['balls'] > 0 else np.nan, axis=1)
            per_mb['BPD'] = per_mb.apply(lambda r: (r['balls'] / r['dismissals']) if r['dismissals'] > 0 else np.nan, axis=1)
            per_mb['AVG'] = per_mb.apply(lambda r: (r['runs'] / r['dismissals']) if r['dismissals'] > 0 else np.nan, axis=1)
            per_mb['BPB'] = per_mb.apply(lambda r: (r['balls'] / r['boundaries']) if r['boundaries'] > 0 else np.nan, axis=1)
        
            # Aggregate per batter per group (keep runs for weighting)
            per_batter_group = per_mb.groupby([COL_BAT, group_col], as_index=False).agg(
                total_runs=('runs', 'sum'),
                mean_SR=('SR', 'mean'), 
                mean_BPD=('BPD', 'mean'),
                mean_AVG=('AVG', 'mean'),
                mean_BPB=('BPB', 'mean')
            )
        
            # CHANGE 5: Calculate WEIGHTED averages by group (weighted by runs scored)
            def weighted_avg(df, value_col):
                """Calculate weighted average: Σ(runs_i * value_i) / Σ(runs_i)"""
                df_clean = df.dropna(subset=[value_col, 'total_runs'])
                if df_clean.empty or df_clean['total_runs'].sum() == 0:
                    return np.nan
                return (df_clean['total_runs'] * df_clean[value_col]).sum() / df_clean['total_runs'].sum()
            
            avg_by_group = per_batter_group.groupby(group_col).apply(
                lambda g: pd.Series({
                    'avg_SR_top7': weighted_avg(g, 'mean_SR'),
                    'avg_BPD_top7': weighted_avg(g, 'mean_BPD'),
                    'avg_AVG_top7': weighted_avg(g, 'mean_AVG'),
                    'avg_BPB_top7': weighted_avg(g, 'mean_BPB')
                })
            ).reset_index()
        
            # Process selected player data
            sel = pf.copy()
            sel[group_col] = sel.get(group_col, "").astype(str).str.lower().fillna('unknown')
            sel[runs_col] = pd.to_numeric(sel.get(runs_col, 0), errors='coerce').fillna(0).astype(int)
            sel['out_flag_tmp'] = pd.to_numeric(sel.get('out', 0), errors='coerce').fillna(0).astype(int)
            sel['dismissal_clean_tmp'] = sel.get('dismissal', "").astype(str).str.lower().str.strip().replace({'nan': '', 'none': ''})
            sel['is_wkt_tmp'] = sel.apply(
                lambda r: 1 if is_bowler_wicket_local(r.get('out_flag_tmp', 0), r.get('dismissal_clean_tmp', '')) else 0, 
                axis=1
            )
            sel['is_boundary'] = sel[runs_col].isin([4, 6]).astype(int)
        
            # Aggregate selected player by group
            sel_grp = sel.groupby(group_col).agg(
                runs=(runs_col, 'sum'), 
                balls=(runs_col, 'count'), 
                dismissals=('is_wkt_tmp', 'sum'),
                boundaries=('is_boundary', 'sum')
            ).reset_index()
            
            sel_grp['SR'] = sel_grp.apply(lambda r: (r['runs'] / r['balls'] * 100.0) if r['balls'] > 0 else np.nan, axis=1)
            sel_grp['BPD'] = sel_grp.apply(lambda r: (r['balls'] / r['dismissals']) if r['dismissals'] > 0 else np.nan, axis=1)
            sel_grp['AVG'] = sel_grp.apply(lambda r: (r['runs'] / r['dismissals']) if r['dismissals'] > 0 else np.nan, axis=1)
            sel_grp['BPB'] = sel_grp.apply(lambda r: (r['balls'] / r['boundaries']) if r['boundaries'] > 0 else np.nan, axis=1)
        
            # Merge with group averages
            merged = pd.merge(sel_grp, avg_by_group, how='left', on=group_col)
        
            # Calculate differences
            for _, row in merged.iterrows():
                g = row[group_col]
                sel_sr = row['SR']
                sel_bpd = row['BPD']
                sel_avg = row['AVG']
                sel_bpb = row['BPB']
                
                avg_sr = row.get('avg_SR_top7', np.nan)
                avg_bpd = row.get('avg_BPD_top7', np.nan)
                avg_avg = row.get('avg_AVG_top7', np.nan)
                avg_bpb = row.get('avg_BPB_top7', np.nan)
                
                # Safe float conversion
                try:
                    avg_sr = float(avg_sr) if not pd.isna(avg_sr) else np.nan
                except:
                    avg_sr = np.nan
                try:
                    avg_bpd = float(avg_bpd) if not pd.isna(avg_bpd) else np.nan
                except:
                    avg_bpd = np.nan
                try:
                    avg_avg = float(avg_avg) if not pd.isna(avg_avg) else np.nan
                except:
                    avg_avg = np.nan
                try:
                    avg_bpb = float(avg_bpb) if not pd.isna(avg_bpb) else np.nan
                except:
                    avg_bpb = np.nan
                
                # Calculate differences (Above Average metrics)
                RAA = (sel_sr - avg_sr) if (not np.isnan(sel_sr) and not np.isnan(avg_sr)) else np.nan
                DAA = (sel_bpd - avg_bpd) if (not np.isnan(sel_bpd) and not np.isnan(avg_bpd)) else np.nan
                AAA = (sel_avg - avg_avg) if (not np.isnan(sel_avg) and not np.isnan(avg_avg)) else np.nan  # Average Above Average
                BAA = (sel_bpb - avg_bpb) if (not np.isnan(sel_bpb) and not np.isnan(avg_bpb)) else np.nan  # Boundary Above Average
                
                out[str(g).lower()] = {
                    'selected_SR': sel_sr,
                    'selected_BPD': sel_bpd,
                    'selected_AVG': sel_avg,
                    'selected_BPB': sel_bpb,
                    'RAA': RAA,
                    'DAA': DAA  # Boundary Above Average (lower is better)
                }
        
            return out
        # def compute_RAA_DAA_for_group_column(group_col):
        #     out = {}
        #     if group_col not in bdf.columns:
        #         return out

        #     working = bdf.copy()
        #     working[group_col] = working[group_col].astype(str).str.lower().fillna('unknown')
        #     working[runs_col] = pd.to_numeric(working.get(runs_col, 0), errors='coerce').fillna(0).astype(int)
        #     working['out_flag_tmp'] = pd.to_numeric(working.get('out', 0), errors='coerce').fillna(0).astype(int)
        #     working['dismissal_clean_tmp'] = working.get('dismissal', "").astype(str).str.lower().str.strip().replace({'nan': '', 'none': ''})

        #     WICKET_TYPES = ['bowled', 'caught', 'hit wicket', 'stumped', 'leg before wicket', 'lbw','leg before wicket','hit wicket']
        #     def is_bowler_wicket_local(out_flag_val, dismissal_text):
        #         try:
        #             if int(out_flag_val) != 1:
        #                 return False
        #         except:
        #             if not out_flag_val:
        #                 return False
        #         if not dismissal_text or str(dismissal_text).strip() == '':
        #             return False
        #         dd = str(dismissal_text).lower()
        #         for token in WICKET_TYPES:
        #             if token in dd:
        #                 return True
        #         return False

        #     working['is_wkt_tmp'] = working.apply(lambda r: 1 if is_bowler_wicket_local(r.get('out_flag_tmp', 0), r.get('dismissal_clean_tmp', '')) else 0, axis=1)

        #     top7 = working[working.get('top7_flag', 0) == 1].copy()

        #     if top7.empty:
        #         if 'p_bat' in working.columns and pd.api.types.is_numeric_dtype(working['p_bat']):
        #             top7 = working[pd.to_numeric(working['p_bat'], errors='coerce').fillna(9999) <= 7].copy()

        #     if top7.empty:
        #         if all(c in working.columns for c in ['p_match', 'inns', COL_BAT]):
        #             tmp = working.dropna(subset=[COL_BAT, 'p_match', 'inns']).copy()
        #             order_col = 'ball_id' if 'ball_id' in tmp.columns else ('ball' if 'ball' in tmp.columns else tmp.columns[0])
        #             first_app = tmp.groupby(['p_match', 'inns', COL_BAT], as_index=False)[order_col].min().rename(columns={order_col: 'first_ball'})
        #             recs = []
        #             for (m, inn), grp in first_app.groupby(['p_match', 'inns']):
        #                 top7_names = grp.sort_values('first_ball').head(7)[COL_BAT].tolist()
        #                 for b in top7_names:
        #                     recs.append((m, inn, b))
        #             if recs:
        #                 top7_df = pd.DataFrame(recs, columns=['p_match', 'inns', COL_BAT])
        #                 top7_df['top7_flag'] = 1
        #                 top7 = working.merge(top7_df, how='inner', on=['p_match', 'inns', COL_BAT])
        #     if top7.empty:
        #         return out

        #     gb_keys = ['p_match', 'inns', COL_BAT, group_col] if all(c in top7.columns for c in ['p_match', 'inns']) else ['p_match', COL_BAT, group_col]
        #     per_mb = (top7.groupby(gb_keys, as_index=False)
        #               .agg(runs=(runs_col, 'sum'),
        #                    balls=(runs_col, 'count'),
        #                    dismissals=('is_wkt_tmp', 'sum')))
        #     per_mb['SR'] = per_mb.apply(lambda r: (r['runs'] / r['balls'] * 100.0) if r['balls'] > 0 else np.nan, axis=1)
        #     per_mb['BPD'] = per_mb.apply(lambda r: (r['balls'] / r['dismissals']) if r['dismissals'] > 0 else np.nan, axis=1)

        #     per_batter_group = per_mb.groupby([COL_BAT, group_col], as_index=False).agg(mean_SR=('SR', 'mean'), mean_BPD=('BPD', 'mean'))

        #     avg_by_group = per_batter_group.groupby(group_col).agg(avg_SR_top7=('mean_SR', 'mean'), avg_BPD_top7=('mean_BPD', 'mean')).reset_index()

        #     sel = pf.copy()
        #     sel[group_col] = sel.get(group_col, "").astype(str).str.lower().fillna('unknown')
        #     sel[runs_col] = pd.to_numeric(sel.get(runs_col, 0), errors='coerce').fillna(0).astype(int)
        #     sel['out_flag_tmp'] = pd.to_numeric(sel.get('out', 0), errors='coerce').fillna(0).astype(int)
        #     sel['dismissal_clean_tmp'] = sel.get('dismissal', "").astype(str).str.lower().str.strip().replace({'nan': '', 'none': ''})
        #     sel['is_wkt_tmp'] = sel.apply(lambda r: 1 if is_bowler_wicket_local(r.get('out_flag_tmp', 0), r.get('dismissal_clean_tmp', '')) else 0, axis=1)

        #     sel_grp = sel.groupby(group_col).agg(runs=(runs_col, 'sum'), balls=(runs_col, 'count'), dismissals=('is_wkt_tmp', 'sum')).reset_index()
        #     sel_grp['SR'] = sel_grp.apply(lambda r: (r['runs'] / r['balls'] * 100.0) if r['balls'] > 0 else np.nan, axis=1)
        #     sel_grp['BPD'] = sel_grp.apply(lambda r: (r['balls'] / r['dismissals']) if r['dismissals'] > 0 else np.nan, axis=1)

        #     merged = pd.merge(sel_grp, avg_by_group, how='left', on=group_col)

        #     for _, row in merged.iterrows():
        #         g = row[group_col]
        #         sel_sr = row['SR']
        #         sel_bpd = row['BPD']
        #         avg_sr = row.get('avg_SR_top7', np.nan)
        #         avg_bpd = row.get('avg_BPD_top7', np.nan)
        #         try:
        #             avg_sr = float(avg_sr) if not pd.isna(avg_sr) else np.nan
        #         except:
        #             avg_sr = np.nan
        #         try:
        #             avg_bpd = float(avg_bpd) if not pd.isna(avg_bpd) else np.nan
        #         except:
        #             avg_bpd = np.nan
        #         RAA = (sel_sr - avg_sr) if (not np.isnan(sel_sr) and not np.isnan(avg_sr)) else np.nan
        #         DAA = (sel_bpd - avg_bpd) if (not np.isnan(sel_bpd) and not np.isnan(avg_bpd)) else np.nan
        #         out[str(g).lower()] = {
        #             'selected_SR': sel_sr,
        #             'selected_BPD': sel_bpd,
        #             'avg_SR_top7': avg_sr,
        #             'avg_BPD_top7': avg_bpd,
        #             'RAA': RAA,
        #             'DAA': DAA
        #         }

        #     return out

        # -------------------- attach RAA/DAA to bk_df and bs_df --------------------
        def _fmt(x):
            return f"{x:.2f}" if (not pd.isna(x)) else '-'

                # Ensure PHASE column exists (derive from 'over' if missing)
        if 'PHASE' not in pf.columns:
            if 'over' in pf.columns:
                pf['PHASE'] = pf['over'].apply(assign_phase)
            else:
                pf['PHASE'] = 'Unknown'

        if 'PHASE' in pf.columns:
            phases = sorted([p for p in pf['PHASE'].dropna().unique() if str(p).strip() != ''])
            phase_rows = []
            if phases:
                for p in phases:
                    g = pf[pf['PHASE'] == p]
                    m = compute_batting_metrics(g)
                    m['PHASE'] = p
                    phase_rows.append(m)
                phase_df = pd.DataFrame(phase_rows).set_index('PHASE')
                phase_df.index.name = 'Phase'
            else:
                phase_df = pd.DataFrame(columns=['PHASE']).set_index('PHASE')
                phase_df.index.name = 'Phase'
        else:
            phase_df = None

        # attach to phase_df
        if phase_df is not None:
            if 'PHASE' in bdf.columns:
                phase_raadaa = compute_RAA_DAA_for_group_column('PHASE')
                new_RAA = []
                new_DAA = []
                for idx in phase_df.index:
                    key = str(idx).lower()
                    val = phase_raadaa.get(key, {})
                    new_RAA.append(_fmt(val.get('RAA', np.nan)))
                    new_DAA.append(_fmt(val.get('DAA', np.nan)))
                phase_df['RAA'] = new_RAA
                phase_df['DAA'] = new_DAA
            else:
                phase_df['RAA'] = '-'
                phase_df['DAA'] = '-'
            st.markdown("<div style='font-weight:700; font-size:15px;'> Performance by Phase </div>", unsafe_allow_html=True)
            st.dataframe(phase_df, use_container_width=True)

        # attach to bk_df
        if bk_df is not None and not bk_df.empty:
            if COL_BOWL_KIND in bdf.columns:
                bk_raadaa = compute_RAA_DAA_for_group_column(COL_BOWL_KIND)
                new_RAA = []
                new_DAA = []
                for idx in bk_df.index:
                    key = str(idx).lower()
                    val = bk_raadaa.get(key, {})
                    new_RAA.append(_fmt(val.get('RAA', np.nan)))
                    new_DAA.append(_fmt(val.get('DAA', np.nan)))
                bk_df['RAA'] = new_RAA
                bk_df['DAA'] = new_DAA
            else:
                bk_df['RAA'] = '-'
                bk_df['DAA'] = '-'

        st.markdown("<div style='font-weight:700; font-size:15px;'> Performance by bowling type </div>", unsafe_allow_html=True)
        st.dataframe(bk_df, use_container_width=True)

        # attach to bs_df
        if bs_df is not None:
            if COL_BOWL_STYLE in bdf.columns:
                bs_raadaa = compute_RAA_DAA_for_group_column(COL_BOWL_STYLE)
                new_RAA = []
                new_DAA = []
                for idx in bs_df.index:
                    key = str(idx).lower()
                    val = bs_raadaa.get(key, {})
                    new_RAA.append(_fmt(val.get('RAA', np.nan)))
                    new_DAA.append(_fmt(val.get('DAA', np.nan)))
                bs_df['RAA'] = new_RAA
                bs_df['DAA'] = new_DAA
            else:
                bs_df['RAA'] = '-'
                bs_df['DAA'] = '-'

            st.markdown("<div style='font-weight:700; font-size:15px;'> Performance by bowling style </div>", unsafe_allow_html=True)
            st.dataframe(bs_df, use_container_width=True)

            # import numpy as np
            # import numpy as np
            # import pandas as pd
            # import matplotlib.pyplot as plt
            # import streamlit as st
            
            # Required objects check
# import numpy as np
            # import numpy as np
            # import pandas as pd
            # import matplotlib.pyplot as plt
            # import streamlit as st
            
            # Required objects check
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
            # import streamlit as st
            
            # Required objects check
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import streamlit as st
            # st.write(pf.bat.unique())
            # st.write(bdf.bat.unique())
            # Required objects check
            required = ['pf', 'bdf', 'player_selected']
            missing = [r for r in required if r not in globals()]
            if missing:
                st.error(f"Required objects missing from scope: {missing}. Place this block after the batter block that defines them.")
            else:
                # Collect bowl_kind and bowl_style values actually present (scan both pf and bdf)
                def unique_vals_union(col):
                    vals = []
                    for df in (pf, bdf):
                        if col in df.columns:
                            vals.extend(df[col].dropna().astype(str).str.strip().tolist())
                    vals = sorted({v for v in vals if str(v).strip() != ''})
                    return vals
            
                bowl_kinds_present = unique_vals_union('bowl_kind')  # e.g. ['pace', 'spin']
                # Limit to pace/spin only
                bowl_kinds_present = [k for k in bowl_kinds_present if 'pace' in k.lower() or 'spin' in k.lower()]
            
                bowl_styles_present = unique_vals_union('bowl_style')
            
                # UI controls
                st.markdown("## Batter — Bowler Kind / Style exploration")
                st.write("Select a Bowler Kind (value as stored) or select a Bowler Style (value as stored).")
                kind_opts = ['-- none --'] + bowl_kinds_present
                style_opts = ['-- none --'] + bowl_styles_present
            
                chosen_kind = st.selectbox("Bowler Kind", options=kind_opts, index=0)
                chosen_style = st.selectbox("Bowler Style", options=style_opts, index=0)
            
                # ---------- robust map-lookup helpers ----------
                def _norm_key(s):
                    if s is None:
                        return ''
                    return str(s).strip().upper().replace(' ', '_').replace('-', '_')
            
                def get_map_index(map_obj, raw_val):
                    if raw_val is None:
                        return None
                    sval = str(raw_val).strip()
                    if sval == '' or sval.lower() in ('nan', 'none'):
                        return None
            
                    if sval in map_obj:
                        return int(map_obj[sval])
                    s_norm = _norm_key(sval)
                    for k in map_obj:
                        try:
                            if isinstance(k, str) and _norm_key(k) == s_norm:
                                return int(map_obj[k])
                        except Exception:
                            continue
                    for k in map_obj:
                        try:
                            if isinstance(k, str) and (k.lower() in sval.lower() or sval.lower() in k.lower()):
                                return int(map_obj[k])
                        except Exception:
                            continue
                    return None
            
                # ---------- grids builder ----------
                def build_pitch_grids(df_in, line_col_name='line', length_col_name='length', runs_col_candidates=('batruns', 'score'),
                                      control_col='control', dismissal_col='dismissal'):
                    if 'length_map' in globals() and isinstance(length_map, dict) and len(length_map) > 0:
                        try:
                            max_idx = max(int(v) for v in length_map.values())
                            n_rows = max(5, max_idx + 1)
                        except Exception:
                            n_rows = 5
                    else:
                        n_rows = 5
                        st.warning("length_map not found; defaulting to 5 rows.")
            
                    length_vals = df_in.get(length_col_name, pd.Series()).dropna().astype(str).str.lower().unique()
                    if any('full toss' in val for val in length_vals):
                        n_rows = max(n_rows, 6)
            
                    n_cols = 5
            
                    count = np.zeros((n_rows, n_cols), dtype=int)
                    bounds = np.zeros((n_rows, n_cols), dtype=int)
                    dots = np.zeros((n_rows, n_cols), dtype=int)
                    runs = np.zeros((n_rows, n_cols), dtype=float)
                    wkt = np.zeros((n_rows, n_cols), dtype=int)
                    ctrl_not = np.zeros((n_rows, n_cols), dtype=int)
            
                    runs_col = None
                    for c in runs_col_candidates:
                        if c in df_in.columns:
                            runs_col = c
                            break
                    if runs_col is None:
                        runs_col = None  # will use 0
            
                    wkt_tokens = {'caught', 'bowled', 'stumped', 'lbw','leg before wicket','hit wicket'}
                    dismissal_series = df_in[dismissal_col].fillna('').astype(str).str.lower()
                    for _, row in df_in.iterrows():
                        li = get_map_index(line_map, row.get(line_col_name, None)) if 'line_map' in globals() else None
                        le = get_map_index(length_map, row.get(length_col_name, None)) if 'length_map' in globals() else None
                        if li is None or le is None:
                            continue
                        if not (0 <= le < n_rows and 0 <= li < n_cols):
                            continue
                        count[le, li] += 1
                        rv = 0
                        if runs_col:
                            try:
                                rv = int(row.get(runs_col, 0) or 0)
                            except:
                                rv = 0
                        runs[le, li] += rv
                        if rv >= 4:
                            bounds[le, li] += 1
                        if rv == 0:
                            dots[le, li] += 1
                        dval = str(row.get(dismissal_col, '') or '').lower()
                        if any(tok in dval for tok in wkt_tokens):
                            wkt[le, li] += 1
                        cval = row.get(control_col, None)
                        if cval is not None:
                            if isinstance(cval, str) and 'not' in cval.lower():
                                ctrl_not[le, li] += 1
                            elif isinstance(cval, (int, float)) and float(cval) == 0:
                                ctrl_not[le, li] += 1
            
                    sr = np.full(count.shape, np.nan)
                    ctrl_pct = np.full(count.shape, np.nan)
                    for i in range(n_rows):
                        for j in range(n_cols):
                            if count[i, j] > 0:
                                sr[i, j] = runs[i, j] / count[i, j] * 100.0
                                ctrl_pct[i, j] = (ctrl_not[i, j] / count[i, j]) * 100.0
            
                    return {
                        'count': count, 'bounds': bounds, 'dots': dots,
                        'runs': runs, 'sr': sr, 'ctrl_pct': ctrl_pct, 'wkt': wkt,
                        'n_rows': n_rows, 'n_cols': n_cols
                    }
            
                # ---------- display utility ----------
                import numpy as np
                import pandas as pd
                import matplotlib.pyplot as plt
                import matplotlib.colors as mcolors
                import streamlit as st
                                
                                                
                # =======================================================================================
                # ACTUALLY FIXED PITCH MAP FUNCTIONS - USES CORRECT LINE_MAP AND LENGTH_MAP
                # =======================================================================================

                def get_map_index(mapping_dict, key_val):
                    """Helper to get index from LINE_MAP or LENGTH_MAP, handles None and normalization"""
                    if key_val is None or mapping_dict is None:
                        return None
                    key_str = str(key_val).strip().upper()
                    return mapping_dict.get(key_str, None)
                
                
                def build_pitch_grids(df_in):
                    """
                    Build pitch grids using LINE_MAP and LENGTH_MAP from globals.
                    Uses runs_col from globals for correct runs column.
                    """
                    # Get from globals
                    LINE_MAP = globals().get('LINE_MAP', {})
                    LENGTH_MAP = globals().get('LENGTH_MAP', {})
                    runs_col = globals().get('runs_col', 'batruns')
                    COL_LINE = globals().get('COL_LINE', 'line')
                    COL_LENGTH = globals().get('COL_LENGTH', 'length')
                    COL_OUT = globals().get('COL_OUT', 'out')
                    
                    print(f"\n[build_pitch_grids] DEBUG:")
                    print(f"  runs_col = {runs_col}")
                    print(f"  COL_LINE = {COL_LINE}")
                    print(f"  COL_LENGTH = {COL_LENGTH}")
                    print(f"  df_in shape = {df_in.shape}")
                    print(f"  df_in columns = {df_in.columns.tolist()}")
                    print(f"  LINE_MAP = {LINE_MAP}")
                    print(f"  LENGTH_MAP = {LENGTH_MAP}")
                    
                    if LINE_MAP is None or len(LINE_MAP) == 0:
                        print("  ERROR: LINE_MAP not found in globals!")
                        return {
                            'count': np.zeros((6, 5)), 'bounds': np.zeros((6, 5)), 'dots': np.zeros((6, 5)),
                            'runs': np.zeros((6, 5)), 'sr': np.zeros((6, 5)), 'ctrl_pct': np.zeros((6, 5)),
                            'wkt': np.zeros((6, 5)), 'n_rows': 6, 'n_cols': 5
                        }
                    
                    if LENGTH_MAP is None or len(LENGTH_MAP) == 0:
                        print("  ERROR: LENGTH_MAP not found in globals!")
                        return {
                            'count': np.zeros((6, 5)), 'bounds': np.zeros((6, 5)), 'dots': np.zeros((6, 5)),
                            'runs': np.zeros((6, 5)), 'sr': np.zeros((6, 5)), 'ctrl_pct': np.zeros((6, 5)),
                            'wkt': np.zeros((6, 5)), 'n_rows': 6, 'n_cols': 5
                        }
                    
                    # Determine dimensions
                    try:
                        max_length_idx = max(int(v) for v in LENGTH_MAP.values())
                        n_rows = max(5, max_length_idx + 1)
                    except:
                        n_rows = 6
                    
                    try:
                        max_line_idx = max(int(v) for v in LINE_MAP.values())
                        n_cols = max(5, max_line_idx + 1)
                    except:
                        n_cols = 5
                    
                    print(f"  Grid dimensions: {n_rows} rows x {n_cols} cols")
                    
                    # Initialize grids
                    count = np.zeros((n_rows, n_cols), dtype=int)
                    runs = np.zeros((n_rows, n_cols), dtype=float)
                    bounds = np.zeros((n_rows, n_cols), dtype=int)
                    dots = np.zeros((n_rows, n_cols), dtype=int)
                    wkt = np.zeros((n_rows, n_cols), dtype=int)
                    controlled = np.zeros((n_rows, n_cols), dtype=int)
                    
                    # Wicket tokens
                    wkt_tokens = {'caught', 'bowled', 'stumped', 'lbw', 'leg before wicket', 'hit wicket'}
                    
                    # Sample some line/length values to debug
                    if COL_LINE in df_in.columns:
                        sample_lines = df_in[COL_LINE].dropna().head(10).tolist()
                        print(f"  Sample line values: {sample_lines}")
                    if COL_LENGTH in df_in.columns:
                        sample_lengths = df_in[COL_LENGTH].dropna().head(10).tolist()
                        print(f"  Sample length values: {sample_lengths}")
                    
                    # Populate grids
                    rows_processed = 0
                    rows_with_valid_line_length = 0
                    
                    for _, row in df_in.iterrows():
                        rows_processed += 1
                        
                        # Get line and length indices using the map
                        li = get_map_index(LINE_MAP, row.get(COL_LINE, None))
                        le = get_map_index(LENGTH_MAP, row.get(COL_LENGTH, None))
                        
                        if li is None or le is None:
                            continue
                        
                        if not (0 <= le < n_rows and 0 <= li < n_cols):
                            continue
                        
                        rows_with_valid_line_length += 1
                        count[le, li] += 1
                        
                        # Get runs value
                        rv = 0
                        if runs_col and runs_col in row.index:
                            try:
                                rv = int(pd.to_numeric(row.get(runs_col, 0), errors='coerce') or 0)
                            except:
                                rv = 0
                        runs[le, li] += rv
                        
                        # Boundaries
                        if rv >= 4:
                            bounds[le, li] += 1
                        
                        # Dots
                        if rv == 0:
                            dots[le, li] += 1
                        
                        # Wickets
                        if COL_OUT in row.index:
                            try:
                                if int(pd.to_numeric(row.get(COL_OUT, 0), errors='coerce') or 0) == 1:
                                    wkt[le, li] += 1
                            except:
                                pass
                        
                        # Check dismissal column for wicket tokens
                        if 'dismissal' in row.index:
                            dval = str(row.get('dismissal', '') or '').lower()
                            if any(tok in dval for tok in wkt_tokens):
                                # Already counted above if out flag was set
                                pass
                        
                        # Control
                        if 'control' in row.index:
                            cval = str(row.get('control', '')).lower()
                            if cval in ['yes', '1', 'true']:
                                controlled[le, li] += 1
                    
                    print(f"  Rows processed: {rows_processed}")
                    print(f"  Rows with valid line/length: {rows_with_valid_line_length}")
                    print(f"  Total count: {count.sum()}")
                    print(f"  Total runs: {runs.sum()}")
                    print(f"  Count grid:\n{count}")
                    
                    # Calculate metrics
                    sr = np.zeros((n_rows, n_cols), dtype=float)
                    ctrl_pct = np.zeros((n_rows, n_cols), dtype=float)
                    
                    mask = count > 0
                    sr[mask] = (runs[mask] / count[mask]) * 100.0
                    ctrl_pct[mask] = ((count[mask] - controlled[mask]) / count[mask]) * 100.0
                    
                    return {
                        'count': count,
                        'runs': runs,
                        'bounds': bounds,
                        'dots': dots,
                        'wkt': wkt,
                        'sr': sr,
                        'ctrl_pct': ctrl_pct,
                        'n_rows': n_rows,
                        'n_cols': n_cols,
                        'LINE_MAP': LINE_MAP,
                        'LENGTH_MAP': LENGTH_MAP
                    }
                
                
                def calculate_raa_grid(df_player, grids, chosen_kind, chosen_style):
                    """
                    Calculate RAA (Runs Above Average) for each line-length combination.
                    """
                    print(f"\n=== RAA CALCULATION DEBUG ===")
                    print(f"chosen_kind: {chosen_kind}")
                    print(f"chosen_style: {chosen_style}")
                    print(f"df_player shape: {df_player.shape}")
                    
                    # Get from globals
                    LINE_MAP = grids.get('LINE_MAP', globals().get('LINE_MAP', {}))
                    LENGTH_MAP = grids.get('LENGTH_MAP', globals().get('LENGTH_MAP', {}))
                    COL_LINE = globals().get('COL_LINE', 'line')
                    COL_LENGTH = globals().get('COL_LENGTH', 'length')
                    COL_BOWL_KIND = globals().get('COL_BOWL_KIND', 'bowl_kind')
                    COL_BOWL_STYLE = globals().get('COL_BOWL_STYLE', 'bowl_style')
                    
                    # Initialize RAA grid
                    n_rows = grids['n_rows']
                    n_cols = grids['n_cols']
                    raa_grid = np.zeros((n_rows, n_cols), dtype=float)
                    
                    # Get the filter column and value
                    filter_col = None
                    filter_value = None
                    
                    if chosen_kind and chosen_kind != '-- none --':
                        filter_col = COL_BOWL_KIND
                        filter_value = chosen_kind
                    elif chosen_style and chosen_style != '-- none --':
                        filter_col = COL_BOWL_STYLE
                        filter_value = chosen_style
                    
                    if filter_col is None or filter_value is None:
                        print("No filter selected, returning zeros")
                        return raa_grid
                    
                    print(f"Filter: {filter_col} = {filter_value}")
                    
                    # Get bdf from globals
                    bdf = globals().get('bdf', pd.DataFrame())
                    if bdf.empty:
                        print("bdf is empty!")
                        return raa_grid
                    
                    print(f"bdf shape before filter: {bdf.shape}")
                    
                    # Filter bdf with the same bowl_kind/style
                    if filter_col not in bdf.columns:
                        print(f"{filter_col} not in bdf columns!")
                        return raa_grid
                    
                    bdf_filtered = bdf[
                        bdf[filter_col].astype(str).str.lower().str.contains(str(filter_value).lower(), na=False)
                    ].copy()
                    
                    print(f"bdf shape after filter: {bdf_filtered.shape}")
                    
                    if bdf_filtered.empty:
                        print("bdf_filtered is empty!")
                        return raa_grid
                    
                    # Create line_length_combo column using the maps
                    def make_combo_key(row):
                        li = get_map_index(LINE_MAP, row.get(COL_LINE, None))
                        le = get_map_index(LENGTH_MAP, row.get(COL_LENGTH, None))
                        if li is None or le is None:
                            return None
                        return f"line{li}_length{le}"
                    
                    bdf_filtered['line_length_combo'] = bdf_filtered.apply(make_combo_key, axis=1)
                    df_player = df_player.copy()
                    df_player['line_length_combo'] = df_player.apply(make_combo_key, axis=1)
                    
                    print(f"\nCalculating RAA for each cell...")
                    
                    # For each line-length combination
                    for line_key, line_idx in LINE_MAP.items():
                        for length_key, length_idx in LENGTH_MAP.items():
                            if length_idx >= n_rows or line_idx >= n_cols:
                                continue
                            
                            combo_key = f"line{line_idx}_length{length_idx}"
                            
                            # Filter bdf for this cell
                            bdf_cell = bdf_filtered[bdf_filtered['line_length_combo'] == combo_key].copy()
                            
                            # Filter player data for this cell
                            df_player_cell = df_player[df_player['line_length_combo'] == combo_key].copy()
                            
                            if bdf_cell.empty:
                                continue
                            
                            if df_player_cell.empty:
                                continue
                            
                            print(f"  {line_key}/{length_key} ({combo_key}): bdf={len(bdf_cell)}, player={len(df_player_cell)}")
                            
                            # Store original globals
                            original_bdf = globals().get('bdf')
                            original_pf = globals().get('pf')
                            
                            # Set globals for RAA function
                            globals()['bdf'] = bdf_cell
                            globals()['pf'] = df_player_cell
                            
                            try:
                                # Call RAA function
                                raa_result = compute_RAA_DAA_for_group_column('line_length_combo')
                                
                                # Extract RAA value
                                if combo_key in raa_result:
                                    raa_value = raa_result[combo_key].get('RAA', np.nan)
                                    print(f"    RAA = {raa_value}")
                                    if not np.isnan(raa_value):
                                        raa_grid[length_idx, line_idx] = raa_value
                                else:
                                    print(f"    combo_key not in result, keys: {list(raa_result.keys())}")
                                
                            except Exception as e:
                                print(f"    ERROR: {e}")
                                import traceback
                                traceback.print_exc()
                            finally:
                                # Restore globals
                                globals()['bdf'] = original_bdf
                                globals()['pf'] = original_pf
                    
                    print(f"\nRAA grid:")
                    print(raa_grid)
                    print("=== END RAA DEBUG ===\n")
                    
                    return raa_grid
                
                
                def display_pitchmaps_from_df(df_src, title_prefix):
                    if df_src is None or df_src.empty:
                        st.info(f"No deliveries to show for {title_prefix}")
                        return
                
                    grids = build_pitch_grids(df_src)
                
                    bh_col_name = globals().get('bat_hand_col', 'bat_hand')
                    is_lhb = False
                    if bh_col_name in df_src.columns:
                        hands = df_src[bh_col_name].dropna().astype(str).str.strip().unique()
                        if any(h.upper().startswith('L') for h in hands):
                            is_lhb = True
                
                    def maybe_flip(arr):
                        return np.fliplr(arr) if is_lhb else arr.copy()
                
                    count = maybe_flip(grids['count'])
                    bounds = maybe_flip(grids['bounds'])
                    dots = maybe_flip(grids['dots'])
                    sr = maybe_flip(grids['sr'])
                    wkt = maybe_flip(grids['wkt'])
                
                    total = count.sum() if count.sum() > 0 else 1.0
                    perc = count.astype(float) / total * 100.0
                
                    # Boundary % = boundaries / balls in cell × 100
                    bound_pct = np.zeros_like(bounds, dtype=float)
                    mask = count > 0
                    bound_pct[mask] = bounds[mask] / count[mask] * 100.0
                
                    # Dot % = dots / balls in cell × 100
                    dot_pct = np.zeros_like(dots, dtype=float)
                    dot_pct[mask] = dots[mask] / count[mask] * 100.0
                
                    # False Shot % — recompute from 'control' column per cell
                    false_shot_pct = np.zeros_like(count, dtype=float)
                    if 'control' in df_src.columns:
                        # Assume control is string or boolean; count 'not in control' or False
                        df_src['not_in_control'] = df_src['control'].astype(str).str.lower().str.contains('not in control|not', na=False).astype(int)
                        # Group by line/length indices (assume build_pitch_grids adds 'line_idx', 'length_idx')
                        if 'line_idx' in df_src.columns and 'length_idx' in df_src.columns:
                            ctrl_raw = df_src.groupby(['line_idx', 'length_idx'])['not_in_control'].mean() * 100
                            # Map back to grid (reshape to match n_rows x n_cols)
                            false_shot_pct = ctrl_raw.unstack(fill_value=0).reindex(
                                index=range(grids['n_rows']), columns=range(grids['n_cols']), fill_value=0
                            ).values
                        else:
                            # Fallback if no indices: use existing ctrl_pct if available
                            false_shot_pct = maybe_flip(grids.get('ctrl_pct', np.zeros_like(count)))
                    else:
                        false_shot_pct = maybe_flip(grids.get('ctrl_pct', np.zeros_like(count)))
                
                    xticks_base = ['Wide Out Off', 'Outside Off', 'On Stumps', 'Down Leg', 'Wide Down Leg']
                    xticks = xticks_base[::-1] if is_lhb else xticks_base
                
                    n_rows = grids['n_rows']
                    if n_rows >= 6:
                        yticklabels = ['Short', 'Back of Length', 'Good', 'Full', 'Yorker', 'Full Toss'][:n_rows]
                    else:
                        yticklabels = ['Short', 'Back of Length', 'Good', 'Full', 'Yorker'][:n_rows]
                
                    # RAA per cell — using your existing function (no changes)
                    raa_grid = np.full((n_rows, grids['n_cols']), np.nan)
                    if 'line' in df_src.columns and 'length' in df_src.columns and 'bdf' in globals() and isinstance(bdf, pd.DataFrame):
                        # Add combo to df_src (filtered) and bdf (full benchmark)
                        df_src['line_length_combo'] = df_src['line'].astype(str).str.lower().str.strip() + '_' + df_src['length'].astype(str).str.lower().str.strip()
                        bdf['line_length_combo'] = bdf['line'].astype(str).str.lower().str.strip() + '_' + bdf['length'].astype(str).str.lower().str.strip()
                
                        # Call your original RAA function with combo as group_col
                        raa_dict = compute_RAA_DAA_for_group_column('line_length_combo')
                
                        # Map dict back to grid using labels
                        for i in range(n_rows):
                            length_str = yticklabels[i].lower().strip()
                            for j in range(grids['n_cols']):
                                line_str = xticks[j].lower().strip()
                                combo = f"{line_str}_{length_str}"
                                raa_grid[i, j] = raa_dict.get(combo, {}).get('RAA', np.nan)
                    else:
                        st.warning("Cannot compute RAA map: missing 'line'/'length' columns or global 'bdf'.")
                
                    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
                    plt.suptitle(f"{player_selected} — {title_prefix}", fontsize=16, weight='bold')
                
                    plot_list = [
                        (perc, '% of Balls (heat)', 'Blues'),
                        (bound_pct, 'Boundary %', 'OrRd'),
                        (dot_pct, 'Dot %', 'Blues'),
                        (sr, 'SR (runs/100 balls)', 'Reds'),
                        (false_shot_pct, 'False Shot %', 'PuBu'),
                        (raa_grid, 'RAA', 'RdYlGn')  # Diverging cmap: green positive, red negative
                    ]
                
                    for ax_idx, (ax, (arr, ttl, cmap)) in enumerate(zip(axes.flat, plot_list)):
                        safe_arr = np.nan_to_num(arr.astype(float), nan=0.0)
                        flat = safe_arr.flatten()
                        if np.all(flat == 0):
                            vmin, vmax = 0, 1
                        else:
                            vmin = float(np.nanmin(flat))
                            vmax = float(np.nanpercentile(flat, 95))
                            if vmax <= vmin:
                                vmax = vmin + 1.0
                
                        im = ax.imshow(safe_arr, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
                        ax.set_title(ttl)
                        ax.set_xticks(range(grids['n_cols']))
                        ax.set_yticks(range(grids['n_rows']))
                        ax.set_xticklabels(xticks, rotation=45, ha='right')
                        ax.set_yticklabels(yticklabels)
                
                        ax.set_xticks(np.arange(-0.5, grids['n_cols'], 1), minor=True)
                        ax.set_yticks(np.arange(-0.5, grids['n_rows'], 1), minor=True)
                        ax.grid(which='minor', color='black', linewidth=0.6, alpha=0.95)
                        ax.tick_params(which='minor', bottom=False, left=False)
                
                        # 'W' annotations ONLY on first map (% of Balls)
                        if ax_idx == 0:
                            for i in range(grids['n_rows']):
                                for j in range(grids['n_cols']):
                                    w_count = int(wkt[i, j])
                                    if w_count > 0:
                                        w_text = f"{w_count} W" if w_count > 1 else 'W'
                                        ax.text(j, i, w_text, ha='center', va='center', fontsize=14, color='gold', weight='bold',
                                                bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
                
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                
                    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                
                    safe_fn = globals().get('safe_st_pyplot', None)
                    try:
                        if callable(safe_fn):
                            safe_fn(fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
                        else:
                            st.pyplot(fig)
                    except Exception:
                        st.pyplot(fig)
                    finally:
                        plt.close(fig)
                
                # def build_pitch_grids(df_src):
                #     """
                #     Build pitch grids for visualization.
                #     Assumes df_src has 'line' and 'length' columns.
                #     Returns dict with various aggregated grids.
                #     """
                #     # Define line and length categories
                #     line_order = ['wide down leg', 'down leg', 'on stumps', 'outside off', 'wide out off']
                #     length_order = ['short', 'back of length', 'good', 'full', 'yorker', 'full toss']
                    
                #     # Normalize line and length
                #     df = df_src.copy()
                #     df['line_norm'] = df.get('line', '').astype(str).str.lower().str.strip()
                #     df['length_norm'] = df.get('length', '').astype(str).str.lower().str.strip()
                    
                #     # Filter valid lines and lengths
                #     df = df[df['line_norm'].isin(line_order) & df['length_norm'].isin(length_order)]
                    
                #     n_cols = len(line_order)
                #     n_rows = len(length_order)
                    
                #     # Initialize grids
                #     count = np.zeros((n_rows, n_cols), dtype=int)
                #     runs = np.zeros((n_rows, n_cols), dtype=int)
                #     bounds = np.zeros((n_rows, n_cols), dtype=int)
                #     dots = np.zeros((n_rows, n_cols), dtype=int)
                #     wkt = np.zeros((n_rows, n_cols), dtype=int)
                #     controlled = np.zeros((n_rows, n_cols), dtype=int)
                    
                #     # Map categories to indices
                #     line_to_idx = {line: idx for idx, line in enumerate(line_order)}
                #     length_to_idx = {length: idx for idx, length in enumerate(length_order)}
                    
                #     # Populate grids
                #     for _, row in df.iterrows():
                #         i = length_to_idx.get(row['length_norm'])
                #         j = line_to_idx.get(row['line_norm'])
                        
                #         if i is None or j is None:
                #             continue
                        
                #         count[i, j] += 1
                #         runs[i, j] += int(row.get('runs', 0))
                        
                #         # Boundaries (4s and 6s)
                #         if int(row.get('runs', 0)) in [4, 6]:
                #             bounds[i, j] += 1
                        
                #         # Dots
                #         if int(row.get('runs', 0)) == 0:
                #             dots[i, j] += 1
                        
                #         # Wickets
                #         if int(row.get('out', 0)) == 1:
                #             wkt[i, j] += 1
                        
                #         # Control (shots played in control)
                #         if row.get('control', '').lower() in ['yes', '1', 'true']:
                #             controlled[i, j] += 1
                    
                #     # Calculate metrics
                #     sr = np.zeros((n_rows, n_cols), dtype=float)
                #     ctrl_pct = np.zeros((n_rows, n_cols), dtype=float)
                    
                #     mask = count > 0
                #     sr[mask] = (runs[mask] / count[mask]) * 100.0
                #     ctrl_pct[mask] = ((count[mask] - controlled[mask]) / count[mask]) * 100.0  # False shot %
                    
                #     return {
                #         'count': count,
                #         'runs': runs,
                #         'bounds': bounds,
                #         'dots': dots,
                #         'wkt': wkt,
                #         'sr': sr,
                #         'ctrl_pct': ctrl_pct,
                #         'n_rows': n_rows,
                #         'n_cols': n_cols
                #     }
                
                
                # def display_pitchmaps_from_df(df_src, bdf_full, title_prefix, player_selected, runs_col, COL_BAT, 
                #                                 bowl_kind=None, bowl_style=None):
                #     """
                #     Display pitch maps with 6 panels including RAA.
                    
                #     Parameters:
                #     -----------
                #     df_src : DataFrame
                #         Filtered data for the selected player
                #     bdf_full : DataFrame
                #         Full benchmark dataset with ALL batters
                #     title_prefix : str
                #         Title for the visualization
                #     player_selected : str
                #         Name of player being analyzed
                #     runs_col : str
                #         Column name for runs
                #     COL_BAT : str
                #         Column name for batter identifier
                #     bowl_kind : str, optional
                #         Bowling kind filter applied to df_src
                #     bowl_style : str, optional
                #         Bowling style filter applied to df_src
                #     """
                #     if df_src is None or df_src.empty:
                #         st.info(f"No deliveries to show for {title_prefix}")
                #         return
                
                #     grids = build_pitch_grids(df_src)
                
                #     bh_col_name = 'bat_hand'  # Adjust if different
                #     is_lhb = False
                #     if bh_col_name in df_src.columns:
                #         hands = df_src[bh_col_name].dropna().astype(str).str.strip().unique()
                #         if any(h.upper().startswith('L') for h in hands):
                #             is_lhb = True
                
                #     def maybe_flip(arr):
                #         return np.fliplr(arr) if is_lhb else arr.copy()
                
                #     count = maybe_flip(grids['count'])
                #     bounds = maybe_flip(grids['bounds'])
                #     dots = maybe_flip(grids['dots'])
                #     sr = maybe_flip(grids['sr'])
                #     ctrl = maybe_flip(grids['ctrl_pct'])
                #     wkt = maybe_flip(grids['wkt'])
                #     runs = maybe_flip(grids['runs'])
                
                #     total = count.sum() if count.sum() > 0 else 1.0
                #     perc = count.astype(float) / total * 100.0
                
                #     # Boundary % = boundaries in cell / balls in cell × 100
                #     bound_pct = np.zeros_like(bounds, dtype=float)
                #     mask = count > 0
                #     bound_pct[mask] = bounds[mask] / count[mask] * 100.0
                
                #     # Dot % = dots in cell / balls in cell × 100
                #     dot_pct = np.zeros_like(dots, dtype=float)
                #     dot_pct[mask] = dots[mask] / count[mask] * 100.0
                
                #     xticks_base = ['Wide Out Off', 'Outside Off', 'On Stumps', 'Down Leg', 'Wide Down Leg']
                #     xticks = xticks_base[::-1] if is_lhb else xticks_base
                
                #     n_rows = grids['n_rows']
                #     if n_rows >= 6:
                #         yticklabels = ['Short', 'Back of Length', 'Good', 'Full', 'Yorker', 'Full Toss'][:n_rows]
                #     else:
                #         yticklabels = ['Short', 'Back of Length', 'Good', 'Full', 'Yorker'][:n_rows]
                
                #     # RAA per cell - NOW USING FULL BENCHMARK
                #     raa_grid = np.full((n_rows, grids['n_cols']), np.nan)
                    
                #     if 'line' in df_src.columns and 'length' in df_src.columns and bdf_full is not None and isinstance(bdf_full, pd.DataFrame):
                #         # Pass the FULL benchmark dataset and the filters
                #         raa_dict = compute_pitchmap_raa(df_src, bdf_full, runs_col=runs_col, COL_BAT=COL_BAT, 
                #                                         bowl_kind=bowl_kind, bowl_style=bowl_style)
                
                #         for i in range(n_rows):
                #             length_str = yticklabels[i].lower().strip()
                #             for j in range(grids['n_cols']):
                #                 line_str = xticks[j].lower().strip()
                #                 combo = f"{line_str}_{length_str}"
                #                 raa_grid[i, j] = raa_dict.get(combo, {}).get('RAA', np.nan)
                        
                #         # Debugging output
                #         st.write("### RAA Debug Info:")
                #         st.write(f"Player: {player_selected}")
                #         st.write(f"Filters applied: bowl_kind={bowl_kind}, bowl_style={bowl_style}")
                #         st.write(f"RAA grid shape: {raa_grid.shape}")
                #         st.write(f"Non-NaN RAA values: {np.sum(~np.isnan(raa_grid))}")
                #         if np.any(~np.isnan(raa_grid)):
                #             st.write(f"RAA min: {np.nanmin(raa_grid):.2f}, max: {np.nanmax(raa_grid):.2f}, mean: {np.nanmean(raa_grid):.2f}")
                        
                #         # Show RAA values in a table
                #         raa_df = pd.DataFrame(raa_grid, columns=xticks, index=yticklabels)
                #         st.write("RAA Values by cell:")
                #         st.dataframe(raa_df.style.format("{:.1f}"))
                        
                #         # Show some combo details
                #         st.write("Sample combo details:")
                #         sample_combos = list(raa_dict.items())[:5]
                #         for combo, data in sample_combos:
                #             st.write(f"  {combo}: RAA={data['RAA']:.1f}, Player SR={data['player_SR']:.1f}, Benchmark SR={data['benchmark_SR']:.1f}")
                #     else:
                #         st.warning("Cannot compute RAA map: missing columns or benchmark data.")
                
                #     fig, axes = plt.subplots(3, 2, figsize=(14, 18))
                #     plt.suptitle(f"{player_selected} — {title_prefix}", fontsize=16, weight='bold')
                
                #     plot_list = [
                #         (perc, '% of Balls (heat)', 'Blues', False),
                #         (bound_pct, 'Boundary %', 'OrRd', False),
                #         (dot_pct, 'Dot %', 'Blues', False),
                #         (sr, 'SR (runs/100 balls)', 'Reds', False),
                #         (ctrl, 'False Shot % (not in control)', 'PuBu', False),
                #         (raa_grid, 'RAA (vs Top7 Avg)', 'RdYlGn', True)  # Diverging colormap
                #     ]
                
                #     for ax_idx, (ax, (arr, ttl, cmap, is_diverging)) in enumerate(zip(axes.flat, plot_list)):
                #         safe_arr = np.nan_to_num(arr.astype(float), nan=0.0)
                #         flat = safe_arr.flatten()
                        
                #         # Different normalization for diverging vs sequential colormaps
                #         if is_diverging:
                #             # For RAA: center at 0, use symmetric range
                #             non_nan_vals = raa_grid[~np.isnan(raa_grid)]
                #             if len(non_nan_vals) > 0:
                #                 abs_max = max(abs(np.nanmin(non_nan_vals)), abs(np.nanmax(non_nan_vals)))
                #                 if abs_max == 0:
                #                     abs_max = 10.0  # Default range if all zeros
                #             else:
                #                 abs_max = 10.0
                            
                #             norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
                #             im = ax.imshow(safe_arr, origin='lower', cmap=cmap, norm=norm)
                #         else:
                #             # For other metrics: use standard normalization
                #             if np.all(flat == 0):
                #                 vmin, vmax = 0, 1
                #             else:
                #                 vmin = float(np.nanmin(flat))
                #                 vmax = float(np.nanpercentile(flat, 95))
                #                 if vmax <= vmin:
                #                     vmax = vmin + 1.0
                            
                #             im = ax.imshow(safe_arr, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
                        
                #         ax.set_title(ttl, fontsize=12, weight='bold')
                #         ax.set_xticks(range(grids['n_cols']))
                #         ax.set_yticks(range(grids['n_rows']))
                #         ax.set_xticklabels(xticks, rotation=45, ha='right', fontsize=9)
                #         ax.set_yticklabels(yticklabels, fontsize=9)
                
                #         ax.set_xticks(np.arange(-0.5, grids['n_cols'], 1), minor=True)
                #         ax.set_yticks(np.arange(-0.5, grids['n_rows'], 1), minor=True)
                #         ax.grid(which='minor', color='black', linewidth=0.6, alpha=0.95)
                #         ax.tick_params(which='minor', bottom=False, left=False)
                
                #         # Add wicket annotations to first plot
                #         if ax_idx == 0:
                #             for i in range(grids['n_rows']):
                #                 for j in range(grids['n_cols']):
                #                     w_count = int(wkt[i, j])
                #                     if w_count > 0:
                #                         w_text = f"{w_count} W" if w_count > 1 else 'W'
                #                         ax.text(j, i, w_text, ha='center', va='center', fontsize=14, color='gold', weight='bold',
                #                                 bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
                        
                #         # Add RAA values as text annotations on RAA plot
                #         if is_diverging:
                #             for i in range(grids['n_rows']):
                #                 for j in range(grids['n_cols']):
                #                     val = raa_grid[i, j]
                #                     if not np.isnan(val) and count[i, j] > 0:  # Only show if there were balls
                #                         # Choose text color based on value
                #                         if abs(val) > abs_max * 0.5:
                #                             text_color = 'white'
                #                         else:
                #                             text_color = 'black'
                #                         ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                #                                fontsize=8, color=text_color, weight='bold')
                
                #         fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                
                #     plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                
                #     # Display using Streamlit
                #     try:
                #         st.pyplot(fig)
                #     except Exception as e:
                #         st.error(f"Error displaying plot: {e}")
                #     finally:
                #         plt.close(fig)
                
                
                # Example usage:
                # display_pitchmaps_from_df(
                #     df_src=filtered_player_data,  # Data for ONE player
                #     bdf_full=full_benchmark_data,  # Data for ALL players
                #     title_prefix="vs Pace Bowlers",
                #     player_selected="Abhishek Sharma",
                #     runs_col='runs',
                #     COL_BAT='batter',
                #     bowl_kind='pace bowler',
                #     bowl_style=None
                # )
                                                            
                # ---------- Wagon chart (existing - runs) ----------
                # def draw_wagon_if_available(df_wagon, batter_name):
                #     if 'draw_cricket_field_with_run_totals_requested' in globals() and callable(globals()['draw_cricket_field_with_run_totals_requested']):
                #         try:
                #             fig_w = draw_cricket_field_with_run_totals_requested(df_wagon, batter_name)
                #             safe_fn = globals().get('safe_st_pyplot', None)
                #             if callable(safe_fn):
                #                 safe_fn(fig_w, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
                #             else:
                #                 st.pyplot(fig_w)
                #         except Exception as e:
                #             st.error(f"Wagon drawing function exists but raised: {e}")
                #     else:
                #         st.warning("Wagon chart function not found; please ensure `draw_cricket_field_with_run_totals_requested` is defined earlier.")


                # Required imports (place near top of your module)
                ##PROBLEM BIG PROBLEM
                def draw_wagon_if_available(df_wagon, batter_name, normalize_to_rhb=True):
                  
                    """
                    Wrapper that calls draw_cricket_field_with_run_totals_requested consistently.
                    - normalize_to_rhb: True => request RHB-normalised output (legacy behaviour).
                                        False => request true handedness visualization (LHB will appear mirrored).
                    This wrapper tries to call the function with the new parameter if available (backwards compatible).
                    """
                    import matplotlib.pyplot as plt
                    import streamlit as st
                    import inspect
               
                    # Defensive check
                    if not isinstance(df_wagon, pd.DataFrame) or df_wagon.empty:
                        st.warning("No wagon data available to draw.")
                        return
               
                    # Decide handedness (for UI messages / debugging)
                    batting_style_val = None
                    if 'bat_hand' in df_wagon.columns:
                        try:
                            batting_style_val = df_wagon['bat_hand'].dropna().iloc[0]
                        except Exception:
                            batting_style_val = None
                    is_lhb = isinstance(batting_style_val, str) and batting_style_val.strip().upper().startswith('L')
               
                    # NEW: For LHB, flip X coordinates in data (assumes 'wagonX' is the column; adjust if different)
                    df_wagon_copy = df_wagon.copy() # Avoid modifying original
                    if is_lhb:
                        if 'wagonX' in df_wagon_copy.columns:
                            df_wagon_copy['wagonX'] = -df_wagon_copy['wagonX'] # Flip X coords (data swap left/right)
                        # If using wagonZone (1-8), swap zones symmetrically
                        # Assume standard zones: 1 (sq leg) <-> 8 (third man), 2 <-> 7, 3 <-> 6, 4 <-> 5, etc.
                        if 'wagonZone' in df_wagon_copy.columns:
                            zone_map = {1: 8, 8: 1, 2: 7, 7: 2, 3: 6, 6: 3, 4: 5, 5: 4} # Symmetric flip
                            df_wagon_copy['wagonZone'] = df_wagon_copy['wagonZone'].map(zone_map).fillna(df_wagon_copy['wagonZone'])
               
                    # Check function signature
                    draw_fn = globals().get('draw_cricket_field_with_run_totals_requested', None)
                    if draw_fn is None or not callable(draw_fn):
                        st.warning("Wagon chart function not found; please ensure `draw_cricket_field_with_run_totals_requested` is defined earlier.")
                        return
               
                    try:
                        sig = inspect.signature(draw_fn)
                        if 'normalize_to_rhb' in sig.parameters:
                            # call with the explicit flag (preferred)
                            fig = draw_fn(df_wagon_copy, batter_name, normalize_to_rhb=normalize_to_rhb)
                        else:
                            # older signature: call without flag (maintain legacy behaviour)
                            fig = draw_fn(df_wagon_copy, batter_name)
               
                        # If the function returned a Matplotlib fig — display it
                        if isinstance(fig, plt.Figure):
                            safe_fn = globals().get('safe_st_pyplot', None)
                            if callable(safe_fn):
                                try:
                                    safe_fn(fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
                                except Exception:
                                    st.pyplot(fig)
                            else:
                                st.pyplot(fig)
                            return
               
                        # If function returned None, it may have drawn to current fig; capture that
                        if fig is None:
                            mpl_fig = plt.gcf()
                            # If figure has axes and content, display it
                            if isinstance(mpl_fig, plt.Figure) and len(mpl_fig.axes) > 0:
                                safe_fn = globals().get('safe_st_pyplot', None)
                                if callable(safe_fn):
                                    try:
                                        safe_fn(mpl_fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
                                    except Exception:
                                        st.pyplot(mpl_fig)
                                else:
                                    st.pyplot(mpl_fig)
                                return
               
                        # If function returned a Plotly figure (rare), display it
                        if 'plotly' in str(type(fig)).lower():
                            try:
                                fig.update_yaxes(scaleanchor="x", scaleratio=1)
                            except Exception:
                                pass
                            st.plotly_chart(fig, use_container_width=True)
                            return
               
                        # Unknown return — just state it
                        st.warning("Wagon draw function executed but returned an unexpected type; nothing displayed.")
                    except Exception as e:
                        st.error(f"Wagon drawing function raised: {e}")
                                
                # def draw_wagon_if_available(df_wagon, batter_name, normalize_to_rhb=True):
                #     """
                #     Wrapper that calls draw_cricket_field_with_run_totals_requested consistently.
                #     - normalize_to_rhb: True => request RHB-normalised output (legacy behaviour).
                #                         False => request true handedness visualization (LHB will appear mirrored).
                #     This wrapper tries to call the function with the new parameter if available (backwards compatible).
                #     """
                #     import matplotlib.pyplot as plt
                #     import streamlit as st
                #     import inspect
               
                #     # Defensive check
                #     if not isinstance(df_wagon, pd.DataFrame) or df_wagon.empty:
                #         st.warning("No wagon data available to draw.")
                #         return
               
                #     # Decide handedness (for UI messages / debugging)
                #     batting_style_val = None
                #     if 'bat_hand' in df_wagon.columns:
                #         try:
                #             batting_style_val = df_wagon['bat_hand'].dropna().iloc[0]
                #         except Exception:
                #             batting_style_val = None
                #     is_lhb = isinstance(batting_style_val, str) and batting_style_val.strip().upper().startswith('L')
               
                #     # NEW: For LHB, flip X coordinates in data (assumes 'wagonX' is the column; adjust if different)
                #     df_wagon_copy = df_wagon.copy()  # Avoid modifying original
                #     if is_lhb:
                #         if 'wagonX' in df_wagon_copy.columns:
                #             df_wagon_copy['wagonX'] = -df_wagon_copy['wagonX']  # Flip X coords (data swap left/right)
                #         # If using wagonZone (1-8), swap zones symmetrically
                #         # Assume standard zones: 1 (sq leg) <-> 8 (third man), 2 <-> 7, 3 <-> 6, 4 <-> 5, etc.
                #         if 'wagonZone' in df_wagon_copy.columns:
                #             zone_map = {1: 8, 8: 1, 2: 7, 7: 2, 3: 6, 6: 3, 4: 5, 5: 4}  # Symmetric flip
                #             df_wagon_copy['wagonZone'] = df_wagon_copy['wagonZone'].map(zone_map).fillna(df_wagon_copy['wagonZone'])
               
                #     # Check function signature
                #     draw_fn = globals().get('draw_cricket_field_with_run_totals_requested', None)
                #     if draw_fn is None or not callable(draw_fn):
                #         st.warning("Wagon chart function not found; please ensure `draw_cricket_field_with_run_totals_requested` is defined earlier.")
                #         return
               
                #     try:
                #         sig = inspect.signature(draw_fn)
                #         if 'normalize_to_rhb' in sig.parameters:
                #             # call with the explicit flag (preferred)
                #             fig = draw_fn(df_wagon_copy, batter_name, normalize_to_rhb=normalize_to_rhb)
                #         else:
                #             # older signature: call without flag (maintain legacy behaviour)
                #             fig = draw_fn(df_wagon_copy, batter_name)
               
                #         # If the function returned a Matplotlib fig — display it
                #         if isinstance(fig, plt.Figure):
                #             safe_fn = globals().get('safe_st_pyplot', None)
                #             if callable(safe_fn):
                #                 try:
                #                     safe_fn(fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
                #                 except Exception:
                #                     st.pyplot(fig)
                #             else:
                #                 st.pyplot(fig)
                #             return
               
                #         # If function returned None, it may have drawn to current fig; capture that
                #         if fig is None:
                #             mpl_fig = plt.gcf()
                #             # If figure has axes and content, display it
                #             if isinstance(mpl_fig, plt.Figure) and len(mpl_fig.axes) > 0:
                #                 safe_fn = globals().get('safe_st_pyplot', None)
                #                 if callable(safe_fn):
                #                     try:
                #                         safe_fn(mpl_fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
                #                     except Exception:
                #                         st.pyplot(mpl_fig)
                #                 else:
                #                     st.pyplot(mpl_fig)
                #                 return
               
                #         # If function returned a Plotly figure (rare), display it
                #         if 'plotly' in str(type(fig)).lower():
                #             try:
                #                 fig.update_yaxes(scaleanchor="x", scaleratio=1)
                #             except Exception:
                #                 pass
                #             st.plotly_chart(fig, use_container_width=True)
                #             return
               
                #         # Unknown return — just state it
                #         st.warning("Wagon draw function executed but returned an unexpected type; nothing displayed.")
                #     except Exception as e:
                #         st.error(f"Wagon drawing function raised: {e}")

                # def draw_wagon_if_available(df_wagon, batter_name, normalize_to_rhb=True):
                #     """
                #     Wrapper that calls draw_cricket_field_with_run_totals_requested consistently.
                #     - normalize_to_rhb: True => request RHB-normalised output (legacy behaviour).
                #                         False => request true handedness visualization (LHB will appear mirrored).
                #     This wrapper tries to call the function with the new parameter if available (backwards compatible).
                #     """
                #     import matplotlib.pyplot as plt
                #     import streamlit as st
                #     import inspect
                
                #     # Defensive check
                #     if not isinstance(df_wagon, pd.DataFrame) or df_wagon.empty:
                #         st.warning("No wagon data available to draw.")
                #         return
                
                #     # Decide handedness (for UI messages / debugging)
                #     batting_style_val = None
                #     if 'bat_hand' in df_wagon.columns:
                #         try:
                #             batting_style_val = df_wagon['bat_hand'].dropna().iloc[0]
                #         except Exception:
                #             batting_style_val = None
                #     is_lhb = isinstance(batting_style_val, str) and batting_style_val.strip().upper().startswith('L')
                
                #     # Check function signature
                #     draw_fn = globals().get('draw_cricket_field_with_run_totals_requested', None)
                #     if draw_fn is None or not callable(draw_fn):
                #         st.warning("Wagon chart function not found; please ensure `draw_cricket_field_with_run_totals_requested` is defined earlier.")
                #         return
                
                #     try:
                #         sig = inspect.signature(draw_fn)
                #         if 'normalize_to_rhb' in sig.parameters:
                #             # call with the explicit flag (preferred)
                #             fig = draw_fn(df_wagon, batter_name, normalize_to_rhb=normalize_to_rhb)
                #         else:
                #             # older signature: call without flag (maintain legacy behaviour)
                #             fig = draw_fn(df_wagon, batter_name)
                
                #         # If the function returned a Matplotlib fig — display it
                #         if isinstance(fig, MplFigure):
                #             safe_fn = globals().get('safe_st_pyplot', None)
                #             if callable(safe_fn):
                #                 try:
                #                     safe_fn(fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
                #                 except Exception:
                #                     st.pyplot(fig)
                #             else:
                #                 st.pyplot(fig)
                #             return
                
                #         # If function returned None, it may have drawn to current fig; capture that
                #         if fig is None:
                #             mpl_fig = plt.gcf()
                #             # If figure has axes and content, display it
                #             if isinstance(mpl_fig, MplFigure) and len(mpl_fig.axes) > 0:
                #                 safe_fn = globals().get('safe_st_pyplot', None)
                #                 if callable(safe_fn):
                #                     try:
                #                         safe_fn(mpl_fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
                #                     except Exception:
                #                         st.pyplot(mpl_fig)
                #                 else:
                #                     st.pyplot(mpl_fig)
                #                 return
                
                #         # If function returned a Plotly figure (rare), display it
                #         if isinstance(fig, go.Figure):
                #             try:
                #                 fig.update_yaxes(scaleanchor="x", scaleratio=1)
                #             except Exception:
                #                 pass
                #             st.plotly_chart(fig, use_container_width=True)
                #             return
                
                #         # Unknown return — just state it
                #         st.warning("Wagon draw function executed but returned an unexpected type; nothing displayed.")
                #     except Exception as e:
                #         st.error(f"Wagon drawing function raised: {e}")
                
                
                # -------------------------
                # 3) Updated draw_caught_dismissals_wagon (Plotly)
                # -------------------------
                def draw_caught_dismissals_wagon(df_wagon, batter_name, normalize_to_rhb=True):
                    """
                    Plotly wagon chart for caught dismissals.
                    - normalize_to_rhb=True: show points in RHB frame (no flipping)
                    - normalize_to_rhb=False: show true handedness; if batter is LHB, flip x coordinates to mirror
                    """
                    # Defensive checks
                    if not isinstance(df_wagon, pd.DataFrame):
                        st.warning("draw_caught_dismissals_wagon expects a DataFrame")
                        return
                
                    if 'dismissal' not in df_wagon.columns:
                        st.warning("No 'dismissal' column present — cannot filter caught dismissals.")
                        return
                
                    caught_df = df_wagon[df_wagon['dismissal'].astype(str).str.lower().str.contains('caught', na=False)].copy()
                    if caught_df.empty:
                        st.info(f"No caught dismissals for {batter_name}.")
                        return
                
                    if 'wagonX' not in caught_df.columns or 'wagonY' not in caught_df.columns:
                        st.warning("Missing 'wagonX' or 'wagonY' columns.")
                        return
                
                    # Numeric conversion and drop invalid
                    caught_df['wagonX'] = pd.to_numeric(caught_df['wagonX'], errors='coerce')
                    caught_df['wagonY'] = pd.to_numeric(caught_df['wagonY'], errors='coerce')
                    caught_df = caught_df.dropna(subset=['wagonX', 'wagonY']).copy()
                    if caught_df.empty:
                        st.info("No valid wagon coordinates for caught dismissals.")
                        return
                
                    # Normalize coords to [-1,1] (match your scaling)
                    center_x = 184.0
                    center_y = 184.0
                    radius = 184.0
                    caught_df['x_plot'] = (caught_df['wagonX'].astype(float) - center_x) / radius
                    caught_df['y_plot'] = - (caught_df['wagonY'].astype(float) - center_y) / radius  # flip Y so positive is up
                
                    # Detect LHB if any
                    batting_style_val = None
                    if 'bat_hand' in df_wagon.columns and not df_wagon.empty:
                        try:
                            batting_style_val = df_wagon['bat_hand'].dropna().iloc[0]
                        except Exception:
                            batting_style_val = None
                    is_lhb = False
                    if batting_style_val is not None:
                        try:
                            is_lhb = str(batting_style_val).strip().upper().startswith('L')
                        except Exception:
                            is_lhb = False
                
                    # Apply display rule:
                    # - If normalize_to_rhb True => do NOT flip (everyone in RHB frame)
                    # - If normalize_to_rhb False and batter is LHB => flip x for true mirror
                    x_vals = caught_df['x_plot'].values
                    y_vals = caught_df['y_plot'].values
                    if (not normalize_to_rhb) and is_lhb:
                        x_vals = -x_vals
                
                    # Filter to inside circle
                    caught_df['distance'] = np.sqrt(x_vals**2 + y_vals**2)
                    caught_df = caught_df[caught_df['distance'] <= 1].copy()
                    if caught_df.empty:
                        st.info(f"No caught dismissals inside the field for {batter_name}.")
                        return
                
                    # Choose hover columns
                    hover_candidates = ['bowler', 'bowl_style', 'line', 'length', 'shot', 'dismissal']
                    customdata_cols = [c for c in hover_candidates if c in caught_df.columns]
                    # Build customdata rows
                    customdata = []
                    for _, row in caught_df.iterrows():
                        customdata.append([("" if pd.isna(row.get(c, "")) else str(row.get(c, ""))) for c in customdata_cols])
                
                    # Build the figure
                    fig = go.Figure()
                
                    # Circles and pitch: if normalize_to_rhb True, use canonical RHB shapes; if False and LHB, flip x coords for shapes
                    def add_circle_shape(fig_obj, x0_raw, y0, x1_raw, y1, **kwargs):
                        if (not normalize_to_rhb) and is_lhb:
                            tx0 = -x1_raw
                            tx1 = -x0_raw
                        else:
                            tx0 = x0_raw
                            tx1 = x1_raw
                        x0_, x1_ = min(tx0, tx1), max(tx0, tx1)
                        fig_obj.add_shape(type="circle", xref="x", yref="y", x0=x0_, y0=y0, x1=x1_, y1=y1, **kwargs)
                
                    add_circle_shape(fig, -1, -1, 1, 1, fillcolor="#228B22", line_color="black", opacity=1, layer="below")
                    add_circle_shape(fig, -0.5, -0.5, 0.5, 0.5, fillcolor="#66bb6a", line_color="white", opacity=1, layer="below")
                
                    # Pitch rectangle (tan)
                    pitch_x0, pitch_x1 = (-0.04, 0.04)
                    if (not normalize_to_rhb) and is_lhb:
                        pitch_x0, pitch_x1 = (-pitch_x1, -pitch_x0)
                    fig.add_shape(type="rect", x0=pitch_x0, y0=-0.08, x1=pitch_x1, y1=0.08,
                                  fillcolor="tan", line_color=None, opacity=1, layer="above")
                
                    # Radial lines (mirror endpoints if LHB & not normalised)
                    angles = np.linspace(0, 2*np.pi, 9)[:-1]
                    for angle in angles:
                        x_end = np.cos(angle)
                        y_end = np.sin(angle)
                        if (not normalize_to_rhb) and is_lhb:
                            x_end = -x_end
                        fig.add_trace(go.Scatter(x=[0, x_end], y=[0, y_end],
                                                 mode='lines', line=dict(color='white', width=1), opacity=0.25, showlegend=False))
                
                    # Plot dismissal points (use x_vals,y_vals filtered earlier)
                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='markers',
                        marker=dict(color='red', size=12, line=dict(color='black', width=1.5)),
                        customdata=customdata,
                        hovertemplate=(
                            "<br>".join([f"<b>{col}:</b> %{{customdata[{i}]}}" for i, col in enumerate(customdata_cols)]) +
                            "<extra></extra>"
                        ),
                        name='Caught Dismissal Locations'
                    ))
                
                    # Layout & equal aspect scaling so mirror is obvious
                    axis_range = 1.2
                    fig.update_layout(
                        xaxis=dict(range=[-axis_range, axis_range], showgrid=False, zeroline=False, visible=False),
                        yaxis=dict(range=[-axis_range, axis_range], showgrid=False, zeroline=False, visible=False),
                        showlegend=True,
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                        width=800,
                        height=800,
                        margin=dict(l=0, r=0, t=0, b=0)
                    )
                    try:
                        fig.update_yaxes(scaleanchor="x", scaleratio=1)
                    except Exception:
                        pass
                
                    # Finally display the plotly figure
                    st.plotly_chart(fig, use_container_width=True)
                # import streamlit as st
                # import pandas as pd
                # import numpy as np
                # import matplotlib.pyplot as plt
                # import matplotlib
                # import plotly.graph_objects as go
                # from matplotlib.figure import Figure as MplFigure
                
                # # -------------------------
                # # draw_wagon_if_available
                # # -------------------------
                # def draw_wagon_if_available(df_wagon, batter_name):
                #     import matplotlib.pyplot as plt
                #     import streamlit as st
                
                #     # Detect LHB
                #     is_lhb = False
                #     if 'bat_hand' in df_wagon.columns and not df_wagon.empty:
                #         try:
                #             is_lhb = str(df_wagon['bat_hand'].dropna().iloc[0]).upper().startswith('L')
                #         except:
                #             pass
                
                #     if 'draw_cricket_field_with_run_totals_requested' in globals() and callable(draw_cricket_field_with_run_totals_requested):
                #         try:
                #             # 1️⃣ Let the function draw the wagon (matplotlib)
                #             draw_cricket_field_with_run_totals_requested(df_wagon, batter_name)
                
                #             # 2️⃣ Grab current figure + axes
                #             fig = plt.gcf()
                #             axes = fig.get_axes()
                
                #             # 3️⃣ MIRROR HERE (THIS IS THE FIX)
                #             if is_lhb:
                #                 for ax in axes:
                #                     ax.invert_xaxis()   # ✅ TRUE mirror image
                
                #             # 4️⃣ Display safely
                #             safe_fn = globals().get('safe_st_pyplot', None)
                #             if callable(safe_fn):
                #                 safe_fn(fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
                #             else:
                #                 st.pyplot(fig)
                
                #         except Exception as e:
                #             st.error(f"Wagon drawing function failed: {e}")
                #     else:
                #         st.warning("Wagon chart function not found.")

                
                
                # # -------------------------
                # # draw_caught_dismissals_wagon (Plotly)
                # # -------------------------
                # def draw_caught_dismissals_wagon(df_wagon, batter_name):
                #     """
                #     Plotly wagon chart for caught dismissals. Mirrors vertically (left-right) when batter is LHB.
                #     """
                #     # Defensive checks
                #     if not isinstance(df_wagon, pd.DataFrame):
                #         st.warning("draw_caught_dismissals_wagon expects a DataFrame")
                #         return
                
                #     if 'dismissal' not in df_wagon.columns:
                #         st.warning("No 'dismissal' column present — cannot filter caught dismissals.")
                #         return
                
                #     caught_df = df_wagon[df_wagon['dismissal'].astype(str).str.lower().str.contains('caught', na=False)].copy()
                #     if caught_df.empty:
                #         st.info(f"No caught dismissals for {batter_name}.")
                #         return
                
                #     if 'wagonX' not in caught_df.columns or 'wagonY' not in caught_df.columns:
                #         st.warning("Missing 'wagonX' or 'wagonY' columns.")
                #         return
                
                #     # Numeric conversion and drop invalid
                #     caught_df['wagonX'] = pd.to_numeric(caught_df['wagonX'], errors='coerce')
                #     caught_df['wagonY'] = pd.to_numeric(caught_df['wagonY'], errors='coerce')
                #     caught_df = caught_df.dropna(subset=['wagonX', 'wagonY']).copy()
                #     if caught_df.empty:
                #         st.info("No valid wagon coordinates for caught dismissals.")
                #         return
                
                #     # Normalize coords to [-1,1] (match your earlier scaling)
                #     center_x = 184.0
                #     center_y = 184.0
                #     radius = 184.0
                #     caught_df['x_plot'] = (caught_df['wagonX'].astype(float) - center_x) / radius
                #     caught_df['y_plot'] = - (caught_df['wagonY'].astype(float) - center_y) / radius  # flip Y so positive is up
                
                #     # Detect LHB if any
                #     batting_style_val = None
                #     if 'bat_hand' in df_wagon.columns and not df_wagon.empty:
                #         try:
                #             batting_style_val = df_wagon['bat_hand'].dropna().iloc[0]
                #         except Exception:
                #             batting_style_val = None
                #     is_lhb = False
                #     if batting_style_val is not None:
                #         try:
                #             is_lhb = str(batting_style_val).strip().upper().startswith('L')
                #         except Exception:
                #             is_lhb = False
                
                #     # Filter to inside circle
                #     caught_df['distance'] = np.sqrt(caught_df['x_plot']**2 + caught_df['y_plot']**2)
                #     caught_df = caught_df[caught_df['distance'] <= 1].copy()
                #     if caught_df.empty:
                #         st.info(f"No caught dismissals inside the field for {batter_name}.")
                #         return
                
                #     # Choose hover columns
                #     hover_candidates = ['bowler', 'bowl_style', 'line', 'length', 'shot', 'dismissal']
                #     customdata_cols = [c for c in hover_candidates if c in caught_df.columns]
                #     # Build customdata rows
                #     customdata = []
                #     for _, row in caught_df.iterrows():
                #         customdata.append([("" if pd.isna(row.get(c, "")) else str(row.get(c, ""))) for c in customdata_cols])
                
                #     # Transform x values for plotting according to LHB
                #     x_vals = caught_df['x_plot'].values
                #     y_vals = caught_df['y_plot'].values
                #     if is_lhb:
                #         x_vals = -x_vals
                
                #     # Build the figure
                #     fig = go.Figure()
                
                #     # Circles (outer + inner) using add_shape but ensuring flipped coordinates if LHB
                #     def add_circle_shape(fig_obj, x0_raw, y0, x1_raw, y1, **kwargs):
                #         if is_lhb:
                #             tx0 = -x1_raw
                #             tx1 = -x0_raw
                #         else:
                #             tx0 = x0_raw
                #             tx1 = x1_raw
                #         # ensure ordering
                #         x0_, x1_ = min(tx0, tx1), max(tx0, tx1)
                #         fig_obj.add_shape(type="circle", xref="x", yref="y", x0=x0_, y0=y0, x1=x1_, y1=y1, **kwargs)
                
                #     add_circle_shape(fig, -1, -1, 1, 1, fillcolor="#228B22", line_color="black", opacity=1, layer="below")
                #     add_circle_shape(fig, -0.5, -0.5, 0.5, 0.5, fillcolor="#66bb6a", line_color="white", opacity=1, layer="below")
                
                #     # Pitch rectangle (tan) - mirrored if LHB
                #     pitch_x0, pitch_x1 = (-0.04, 0.04)
                #     if is_lhb:
                #         pitch_x0, pitch_x1 = (-pitch_x1, -pitch_x0)
                #     fig.add_shape(type="rect", x0=pitch_x0, y0=-0.08, x1=pitch_x1, y1=0.08,
                #                   fillcolor="tan", line_color=None, opacity=1, layer="above")
                
                #     # Radial lines (mirror endpoints)
                #     angles = np.linspace(0, 2*np.pi, 9)[:-1]
                #     for angle in angles:
                #         x_end = np.cos(angle)
                #         y_end = np.sin(angle)
                #         if is_lhb:
                #             x_end = -x_end
                #         fig.add_trace(go.Scatter(x=[0, x_end], y=[0, y_end],
                #                                  mode='lines', line=dict(color='white', width=1), opacity=0.25, showlegend=False))
                
                #     # Plot dismissal points
                #     fig.add_trace(go.Scatter(
                #         x=x_vals,
                #         y=y_vals,
                #         mode='markers',
                #         marker=dict(color='red', size=12, line=dict(color='black', width=1.5)),
                #         customdata=customdata,
                #         hovertemplate=(
                #             "<br>".join([f"<b>{col}:</b> %{{customdata[{i}]}}" for i, col in enumerate(customdata_cols)]) +
                #             "<extra></extra>"
                #         ),
                #         name='Caught Dismissal Locations'
                #     ))
                
                #     # Layout & equal aspect scaling so mirror is obvious
                #     axis_range = 1.2
                #     fig.update_layout(
                #         xaxis=dict(range=[-axis_range, axis_range], showgrid=False, zeroline=False, visible=False),
                #         yaxis=dict(range=[-axis_range, axis_range], showgrid=False, zeroline=False, visible=False),
                #         showlegend=True,
                #         legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                #         width=800,
                #         height=800,
                #         margin=dict(l=0, r=0, t=0, b=0)
                #     )
                #     try:
                #         fig.update_yaxes(scaleanchor="x", scaleratio=1)
                #     except Exception:
                #         pass
                
                #     # Finally display the plotly figure
                #     st.plotly_chart(fig, use_container_width=True)


                
                # def draw_caught_dismissals_wagon(df_wagon, batter_name):
                #     # Filter caught dismissals
                #     caught_df = df_wagon[
                #         df_wagon['dismissal'].astype(str).str.lower().str.contains('caught', na=False)
                #     ].copy()
                
                #     if caught_df.empty:
                #         st.info(f"No caught dismissals for {batter_name}.")
                #         return
                
                #     # Extract coordinates and details for hover
                #     if 'wagonX' not in caught_df.columns or 'wagonY' not in caught_df.columns:
                #         st.warning("Missing 'wagonX' or 'wagonY' columns.")
                #         return
                
                #     # Required columns for hover (adjust if names differ)
                #     required_cols = ['wagonX', 'wagonY', 'bowler', 'bowl_style', 'line', 'length', 'shot']  # 'shot' assuming it's the column for shot played
                #     missing_cols = [col for col in required_cols if col not in caught_df.columns]
                #     if missing_cols:
                #         st.warning(f"Missing columns for hover: {missing_cols}. Hover will show available info only.")
                
                #     # Scale coordinates to normalized -1 to 1
                #     center_x = 184
                #     center_y = 184
                #     radius = 184
                #     caught_df['x_plot'] = (caught_df['wagonX'].astype(float) - center_x) / radius
                #     caught_df['y_plot'] = - (caught_df['wagonY'].astype(float) - center_y) / radius  # flip Y (positive down)
                
                #     # LHB flip X if needed
                #     batting_style_val = df_wagon['bat_hand'].iloc[0] if 'bat_hand' in df_wagon.columns and not df_wagon.empty else 'R'
                #     is_lhb = str(batting_style_val).upper().startswith('L')
                #     if is_lhb:
                #         caught_df['x_plot'] = -caught_df['x_plot']
                
                #     # Filter only points inside the circle (using circle formula)
                #     caught_df['distance'] = np.sqrt(caught_df['x_plot']**2 + caught_df['y_plot']**2)
                #     caught_df = caught_df[caught_df['distance'] <= 1]  # radius 1 in normalized space
                #     if caught_df.empty:
                #         st.info(f"No caught dismissals inside the field for {batter_name}.")
                #         return
                
                #     # Create interactive Plotly figure
                #     fig = go.Figure()
                
                #     # Outer field: dark green circle
                #     fig.add_shape(type="circle", xref="x", yref="y",
                #                   x0=-1, y0=-1, x1=1, y1=1,
                #                   fillcolor="#228B22", line_color="black", opacity=1, layer="below")
                
                #     # Inner circle: lighter green
                #     fig.add_shape(type="circle", xref="x", yref="y",
                #                   x0=-0.5, y0=-0.5, x1=0.5, y1=0.5,
                #                   fillcolor="#66bb6a", line_color="white", opacity=1, layer="below")
                
                #     # Pitch rectangle: tan
                #     fig.add_shape(type="rect", x0=-0.04, y0=-0.08, x1=0.04, y1=0.08,
                #                   fillcolor="tan", line_color=None, opacity=1, layer="above")
                
                #     # Radial lines (white, alpha 0.25)
                #     angles = np.linspace(0, 2*np.pi, 9)[:-1]
                #     for angle in angles:
                #         x_end = np.cos(angle)
                #         y_end = np.sin(angle)
                #         fig.add_trace(go.Scatter(x=[0, x_end], y=[0, y_end],
                #                                  mode='lines', line=dict(color='white', width=1, dash='solid'),
                #                                  opacity=0.25, showlegend=False))
                
                #     # Plot red dots with hover data
                #     hover_template = (
                #         "<b>Bowler:</b> %{customdata[0]}<br>"
                #         "<b>Bowler Style:</b> %{customdata[1]}<br>"
                #         "<b>Line:</b> %{customdata[2]}<br>"
                #         "<b>Length:</b> %{customdata[3]}<br>"
                #         "<b>Shot Played:</b> %{customdata[4]}<extra></extra>"
                #     )
                
                #     fig.add_trace(go.Scatter(
                #         x=caught_df['x_plot'],
                #         y=caught_df['y_plot'],
                #         mode='markers',
                #         marker=dict(color='red', size=12, line=dict(color='black', width=1.5)),
                #         customdata=caught_df[['bowler', 'bowl_style', 'line', 'length', 'shot']].values,  # Hover data
                #         hovertemplate=hover_template,
                #         name='Caught Dismissal Locations'
                #     ))
                
                #     # Layout
                #     fig.update_layout(
                #         xaxis=dict(range=[-1.2, 1.2], showgrid=False, zeroline=False, visible=False),
                #         yaxis=dict(range=[-1.2, 1.2], showgrid=False, zeroline=False, visible=False),
                #         showlegend=True,
                #         legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                #         width=800,
                #         height=800,
                #         margin=dict(l=0, r=0, t=0, b=0)
                #     )
                
                #     # Display in Streamlit
                #     st.plotly_chart(fig, use_container_width=True)
                # ---------- Caught Dismissals Wagon — exact same field style + red dots ----------
                # def draw_caught_dismissals_wagon(df_wagon, batter_name):
                #     # Filter caught dismissals
                #     caught_df = df_wagon[
                #         df_wagon['dismissal'].astype(str).str.lower().str.contains('caught', na=False)
                #     ].copy()
            
                #     if caught_df.empty:
                #         st.info(f"No caught dismissals for {batter_name}.")
                #         return
            
                #     # Extract coordinates
                #     if 'wagonX' not in caught_df.columns or 'wagonY' not in caught_df.columns:
                #         st.warning("Missing 'wagonX' or 'wagonY' columns.")
                #         return
            
                #     x_coords = caught_df['wagonX'].astype(float)
                #     y_coords = caught_df['wagonY'].astype(float)
            
                #     # Create figure with EXACT same style
                #     fig, ax = plt.subplots(figsize=(10, 10))
                #     ax.set_aspect('equal')
                #     ax.axis('off')
            
                #     # Outer field: #228B22 (dark green)
                #     ax.add_patch(Circle((0, 0), 1, fill=True, color='#228B22', alpha=1))
                #     ax.add_patch(Circle((0, 0), 1, fill=False, color='black', linewidth=3))
            
                #     # Inner circle: #66bb6a (lighter green)
                #     ax.add_patch(Circle((0, 0), 0.5, fill=True, color='#66bb6a'))
                #     ax.add_patch(Circle((0, 0), 0.5, fill=False, color='white', linewidth=1))
            
                #     # Pitch rectangle: tan
                #     pitch_rect = Rectangle((-0.04, -0.08), 0.08, 0.16, color='tan', alpha=1, zorder=8)
                #     ax.add_patch(pitch_rect)
            
                #     # Radial lines
                #     angles = np.linspace(0, 2*np.pi, 9)[:-1]
                #     for angle in angles:
                #         x = math.cos(angle)
                #         y = math.sin(angle)
                #         ax.plot([0, x], [0, y], color='white', alpha=0.25, linewidth=1)
            
                #     # Scale coordinates to normalized -1 to 1
                #     center_x = 184
                #     center_y = 184
                #     radius = 184
                #     x_plot = (x_coords - center_x) / radius
                #     y_plot = - (y_coords - center_y) / radius  # flip Y (positive down)
            
                #     # LHB flip X if needed
                #     batting_style_val = df_wagon['bat_hand'].iloc[0] if 'bat_hand' in df_wagon.columns and not df_wagon.empty else 'R'
                #     is_lhb = str(batting_style_val).upper().startswith('L')
                #     if is_lhb:
                #         x_plot = -x_plot
            
                #     # Plot red dots
                #     ax.scatter(
                #         x_plot, y_plot,
                #         color='red', s=80, alpha=0.9, edgecolor='black', linewidth=1.5,
                #         marker='o', zorder=20
                #     )
            
                #     # Legend
                #     ax.scatter([], [], color='red', s=150, label='Caught Dismissal Locations')
                #     ax.legend(loc='upper right', fontsize=10, frameon=True)
            
                #     # Limits
                #     ax.set_xlim(-1.2, 1.2)
                #     ax.set_ylim(-1.2, 1.2)
                #     ax.set_aspect('equal')
            
                #     # Display
                #     safe_fn = globals().get('safe_st_pyplot', None)
                #     try:
                #         if callable(safe_fn):
                #             safe_fn(fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
                #         else:
                #             st.pyplot(fig)
                #     except Exception:
                #         st.pyplot(fig)
                #     finally:
                #         plt.close(fig)
            
                # ---------- When user selects a kind ----------
# ============================================================================
                # USAGE - HOW TO CALL THE FUNCTIONS
                # ============================================================================
                
                # ---------- When user selects a kind ----------
                # When bowl_kind is chosen
                # When bowl_kind is chosen
                if chosen_kind and chosen_kind != '-- none --':
                    def filter_by_kind(df, col='bowl_kind', kind=chosen_kind):
                        if col not in df.columns:
                            return df.iloc[0:0]
                        mask = df[col].astype(str).str.lower().str.contains(str(kind).lower(), na=False)
                        if not mask.any():
                            norm_kind = _norm_key(kind)
                            mask = df[col].apply(lambda x: _norm_key(x) == norm_kind)
                        return df[mask].copy()
                
                    sel_pf = filter_by_kind(pf)
                    sel_bdf = filter_by_kind(bdf)
                
                    df_use = sel_pf if not sel_pf.empty else sel_bdf
                    if df_use.empty:
                        st.info(f"No deliveries found for bowler kind '{chosen_kind}'.")
                    else:
                        st.markdown(f"### Detailed view — Bowler Kind: {chosen_kind}")
                        draw_wagon_if_available(df_use, player_selected)
                
                        st.markdown(f"#### {player_selected}'s Caught Dismissals")
                        draw_caught_dismissals_wagon(df_use, player_selected)
                
                        # Updated call: no extra params needed
                        display_pitchmaps_from_df(df_use, f"vs Bowler Kind: {chosen_kind}")
                
                # When bowl_style is chosen
                if chosen_style and chosen_style != '-- none --':
                    def filter_by_style(df, col='bowl_style', style=chosen_style):
                        if col not in df.columns:
                            return df.iloc[0:0]
                        mask = df[col].astype(str).str.lower().str.contains(str(style).lower(), na=False)
                        if not mask.any():
                            norm_style = _norm_key(style)
                            mask = df[col].apply(lambda x: _norm_key(x) == norm_style)
                        return df[mask].copy()
                
                    sel_pf = filter_by_style(pf)
                    sel_bdf = filter_by_style(bdf)
                
                    df_use = sel_pf if not sel_pf.empty else sel_bdf
                    if df_use.empty:
                        st.info(f"No deliveries found for bowler style '{chosen_style}'.")
                    else:
                        st.markdown(f"### Detailed view — Bowler Style: {chosen_style}")
                        draw_wagon_if_available(df_use, player_selected)
                
                        st.markdown(f"#### {player_selected}'s Caught Dismissals")
                        draw_caught_dismissals_wagon(df_use, player_selected)
                
                        # Updated call: no extra params needed
                        display_pitchmaps_from_df(df_use, f"vs Bowler Style: {chosen_style}")
                # if chosen_kind and chosen_kind != '-- none --':
                #     def filter_by_kind(df, col='bowl_kind', kind=chosen_kind):
                #         if col not in df.columns:
                #             return df.iloc[0:0]
                #         mask = df[col].astype(str).str.lower().str.contains(str(kind).lower(), na=False)
                #         if not mask.any():
                #             norm_kind = _norm_key(kind)
                #             mask = df[col].apply(lambda x: _norm_key(x) == norm_kind)
                #         return df[mask].copy()
                
                #     sel_pf = filter_by_kind(pf)
                #     sel_bdf = filter_by_kind(bdf)
                
                #     df_use = sel_pf if not sel_pf.empty else sel_bdf
                #     if df_use.empty:
                #         st.info(f"No deliveries found for bowler kind '{chosen_kind}'.")
                #     else:
                #         st.markdown(f"### Detailed view — Bowler Kind: {chosen_kind}")
                #         draw_wagon_if_available(df_use, player_selected)
                #         st.markdown(f"#### {player_selected}'s Caught Dismissals")
                #         draw_caught_dismissals_wagon(df_use, player_selected)
                #         display_pitchmaps_from_df(df_use, f"vs Bowler Kind: {chosen_kind}", 
                #                                  chosen_kind=chosen_kind, chosen_style=None)
                
                # # When bowl_style is chosen
                # if chosen_style and chosen_style != '-- none --':
                #     def filter_by_style(df, col='bowl_style', style=chosen_style):
                #         if col not in df.columns:
                #             return df.iloc[0:0]
                #         mask = df[col].astype(str).str.lower().str.contains(str(style).lower(), na=False)
                #         if not mask.any():
                #             norm_style = _norm_key(style)
                #             mask = df[col].apply(lambda x: _norm_key(x) == norm_style)
                #         return df[mask].copy()
                
                #     sel_pf = filter_by_style(pf)
                #     sel_bdf = filter_by_style(bdf)
                
                #     df_use = sel_pf if not sel_pf.empty else sel_bdf
                #     if df_use.empty:
                #         st.info(f"No deliveries found for bowler style '{chosen_style}'.")
                #     else:
                #         st.markdown(f"### Detailed view — Bowler Style: {chosen_style}")
                #         draw_wagon_if_available(df_use, player_selected)
                #         st.markdown(f"#### {player_selected}'s Caught Dismissals")
                #         draw_caught_dismissals_wagon(df_use, player_selected)
                #         display_pitchmaps_from_df(df_use, f"vs Bowler Style: {chosen_style}", 
                #                                  chosen_kind=None, chosen_style=chosen_style)
            
                   
        # The rest of the code (wagon wheels, pitchmaps, shot productivity, etc.) will now use the phase-filtered pf/bdf automatically
       
        # ... (continue with the rest of your original code from here, like st.markdown("<div style='font-weight:800; font-size:16px; margin-top:8px;'> Wagon wheels — Pace & Spin</div>", unsafe_allow_html=True) and so on)
            
            st.markdown("<div style='font-weight:800; font-size:16px; margin-top:8px;'> Wagon wheels — Pace & Spin</div>", unsafe_allow_html=True)
            if COL_BOWL_KIND in pf.columns:
                pf_pace = pf[pf[COL_BOWL_KIND].str.contains('pace', na=False)].copy()
                pf_spin = pf[pf[COL_BOWL_KIND].str.contains('spin', na=False)].copy()
            else:
                pf_pace = pf.iloc[0:0].copy()
                pf_spin = pf.iloc[0:0].copy()
            
            c1, c2 = st.columns([1,1], gap="large")
            with c1:
                st.markdown(f"<div style='font-size:14px; font-weight:800;'> {player_selected} — vs Pace (Wagon)</div>", unsafe_allow_html=True)
                fig_p = draw_wagon(pf_pace, f"{player_selected} — vs Pace", is_lhb)
                display_figure_fixed_height_html(fig_p, height_px=HEIGHT_WAGON_PX, margin_px=0)
            with c2:
                st.markdown(f"<div style='font-size:14px; font-weight:800;'> {player_selected} — vs Spin (Wagon)</div>", unsafe_allow_html=True)
                fig_s = draw_wagon(pf_spin, f"{player_selected} — vs Spin", is_lhb)
                display_figure_fixed_height_html(fig_s, height_px=HEIGHT_WAGON_PX, margin_px=0)
            
            # -------------------------------------------------------------------------
            # Pitchmaps — Boundaries, Dismissals, and Dot Balls (Pace vs Spin)
            # use improved readable annotation style (same as bowling)
            # -------------------------------------------------------------------------
            st.markdown("<div style='font-size:16px; font-weight:800; margin-top:6px;'> Pitchmaps — Boundaries, Dismissals & Dot %</div>", unsafe_allow_html=True)
            
            # small consistent LINE/LENGTH maps used across app
            LINE_MAP = {
                'WIDE_OUTSIDE_OFFSTUMP': 0,
                'OUTSIDE_OFFSTUMP': 1,
                'ON_THE_STUMPS': 2,
                'DOWN_LEG': 3,
                'WIDE_DOWN_LEG': 4
            }
            LENGTH_MAP = {
                'SHORT': 0,
                'SHORT_OF_A_GOOD_LENGTH': 1,
                'GOOD_LENGTH': 2,
                'FULL': 3,
                'YORKER': 4,
                'FULL_TOSS': 4
            }
            
            # plotting helper (same readable style as bowling)
            import matplotlib.patheffects as mpatheffects
            def plot_grid_with_readable_labels(grid, title, cmap='Oranges', mirror=False, fmt='int', vmax=None):
                disp = np.fliplr(grid) if mirror else grid.copy()
                xticks_base = ['Wide Out Off','Outside Off','On Stumps','Down Leg','Wide Down Leg']
                xticks = list(reversed(xticks_base)) if mirror else xticks_base
                real_vmax = float(np.nanmax(disp)) if (not np.all(np.isnan(disp)) and np.nanmax(disp) > 0) else 1.0
                vmax_use = float(vmax) if (vmax is not None and vmax > 0) else real_vmax
                
                fig, ax = plt.subplots(figsize=(6,9), dpi=150)
                im = ax.imshow(disp, origin='lower', cmap=cmap, vmin=0, vmax=vmax_use)
                ax.set_xticks(range(5)); ax.set_yticks(range(5))
                ax.set_xticklabels(xticks, rotation=40, ha='right')
                ax.set_yticklabels(['Short','Back of Length','Good','Full','Yorker'])
                for i in range(5):
                    for j in range(5):
                        val = disp[i,j]
                        if fmt == 'pct':
                            lab = f"{val:.2f}%" if (not np.isnan(val)) else "0.00%"
                        elif fmt == 'float':
                            lab = f"{val:.2f}"
                        else:
                            try:
                                lab = f"{int(val)}"
                            except:
                                lab = f"{val}"
                        try:
                            intensity = float(val) / float(vmax_use) if vmax_use > 0 else 0.0
                        except:
                            intensity = 0.0
                        txt_color = 'white' if intensity > 0.55 else 'black'
                        txt = ax.text(j, i, lab, ha='center', va='center', color=txt_color, fontsize=12, fontweight='bold')
                        txt.set_path_effects([mpatheffects.Stroke(linewidth=2, foreground='white' if txt_color=='black' else 'black'),
                                              mpatheffects.Normal()])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                plt.title(title, pad=6, fontsize=12)
                plt.tight_layout(pad=0)
                fig.subplots_adjust(top=0.99, bottom=0.01, left=0.06, right=0.99)
                return fig
            
            # helper builders
            def build_boundaries_grid_local(df_local):
                grid = np.zeros((5,5), dtype=int)
                if df_local.shape[0] == 0: return grid
                if COL_LINE not in df_local.columns or COL_LENGTH not in df_local.columns or COL_RUNS not in df_local.columns:
                    return grid
                plot_df = df_local[[COL_LINE, COL_LENGTH, COL_RUNS]].dropna(subset=[COL_LINE, COL_LENGTH])
                for _, r in plot_df.iterrows():
                    li = LINE_MAP.get(r[COL_LINE], None)
                    le = LENGTH_MAP.get(r[COL_LENGTH], None)
                    if li is None or le is None: continue
                    try:
                        runs_here = int(r[COL_RUNS])
                    except:
                        runs_here = 0
                    if runs_here in (4,6):
                        grid[le, li] += 1
                return grid
            
            def build_dismissals_grid_local(df_local):
                grid = np.zeros((5,5), dtype=int)
                if df_local.shape[0] == 0: return grid
                if COL_LINE not in df_local.columns or COL_LENGTH not in df_local.columns:
                    return grid
                # create is_wkt in local frame similar to your rules
                df_local['dismissal_clean'] = df_local.get('dismissal', "").astype(str).str.lower().str.strip().replace({'nan':'','none':''})
                df_local['out_flag'] = pd.to_numeric(df_local.get('out',0), errors='coerce').fillna(0).astype(int)
                special_runout_types = set(['run out','runout','retired','retired not out','retired out','obstructing the field'])
                df_local['is_wkt'] = df_local.apply(
                    lambda r: 1 if (int(r.get('out_flag',0)) == 1 and str(r.get('dismissal_clean','')).strip() not in special_runout_types and str(r.get('dismissal_clean','')).strip() != '') else 0,
                    axis=1
                )
                plot_df = df_local[[COL_LINE, COL_LENGTH, 'is_wkt']].dropna(subset=[COL_LINE, COL_LENGTH])
                for _, r in plot_df.iterrows():
                    li = LINE_MAP.get(r[COL_LINE], None)
                    le = LENGTH_MAP.get(r[COL_LENGTH], None)
                    if li is None or le is None: continue
                    if int(r.get('is_wkt', 0)) == 1:
                        grid[le, li] += 1
                return grid
            
            def build_dot_grid_local(df_local):
                grid = np.zeros((5,5), dtype=int)
                if df_local.shape[0] == 0: return grid
                if COL_LINE not in df_local.columns or COL_LENGTH not in df_local.columns or COL_RUNS not in df_local.columns:
                    return grid
                nob_col = 'noball' if 'noball' in df_local.columns else None
                wide_col = 'wide' if 'wide' in df_local.columns else None
                pl = [COL_LINE, COL_LENGTH, COL_RUNS]
                if nob_col: pl.append(nob_col)
                if wide_col: pl.append(wide_col)
                plot_df = df_local[pl].dropna(subset=[COL_LINE, COL_LENGTH])
                for _, r in plot_df.iterrows():
                    if nob_col and int(r.get(nob_col,0)) != 0: continue
                    if wide_col and int(r.get(wide_col,0)) != 0: continue
                    li = LINE_MAP.get(r[COL_LINE], None)
                    le = LENGTH_MAP.get(r[COL_LENGTH], None)
                    if li is None or le is None: continue
                    try:
                        runs_here = int(r[COL_RUNS])
                    except:
                        runs_here = 0
                    if runs_here == 0:
                        grid[le, li] += 1
                return grid
            
            # build grids for pace/spin
            grid_pace_bound = build_boundaries_grid_local(pf_pace)
            grid_spin_bound = build_boundaries_grid_local(pf_spin)
            grid_pace_wkt = build_dismissals_grid_local(pf_pace)
            grid_spin_wkt = build_dismissals_grid_local(pf_spin)
            grid_pace_dot = build_dot_grid_local(pf_pace)
            grid_spin_dot = build_dot_grid_local(pf_spin)
            
            # determine sensible vmax values so annotation contrast uses consistent scale
            vmax_bound = max(np.max(grid_pace_bound), np.max(grid_spin_bound), 1)
            vmax_wkt = max(np.max(grid_pace_wkt), np.max(grid_spin_wkt), 1)
            vmax_dot = max(np.max(grid_pace_dot), np.max(grid_spin_dot), 1)
            
            # display Boundaries row
            c1, c2 = st.columns([1,1], gap="large")
            with c1:
                st.markdown(f"<div style='font-weight:800;'> Boundaries — Pace</div>", unsafe_allow_html=True)
                fig_b1 = plot_grid_with_readable_labels(grid_pace_bound, f"{player_selected} — Boundaries vs Pace", cmap='Oranges', mirror=is_lhb, fmt='int', vmax=vmax_bound)
                display_figure_fixed_height_html(fig_b1, height_px=HEIGHT_PITCHMAP_PX, margin_px=0)
            with c2:
                st.markdown(f"<div style='font-weight:800;'> Boundaries — Spin</div>", unsafe_allow_html=True)
                fig_b2 = plot_grid_with_readable_labels(grid_spin_bound, f"{player_selected} — Boundaries vs Spin", cmap='Oranges', mirror=is_lhb, fmt='int', vmax=vmax_bound)
                display_figure_fixed_height_html(fig_b2, height_px=HEIGHT_PITCHMAP_PX, margin_px=0)
            
            # display Dismissals row
            c3, c4 = st.columns([1,1], gap="large")
            with c3:
                st.markdown(f"<div style='font-weight:800;'> Dismissals — Pace</div>", unsafe_allow_html=True)
                fig_w1 = plot_grid_with_readable_labels(grid_pace_wkt, f"{player_selected} — Dismissals vs Pace", cmap='Reds', mirror=is_lhb, fmt='int', vmax=vmax_wkt)
                display_figure_fixed_height_html(fig_w1, height_px=HEIGHT_PITCHMAP_PX, margin_px=0)
            with c4:
                st.markdown(f"<div style='font-weight:800;'> Dismissals — Spin</div>", unsafe_allow_html=True)
                fig_w2 = plot_grid_with_readable_labels(grid_spin_wkt, f"{player_selected} — Dismissals vs Spin", cmap='Reds', mirror=is_lhb, fmt='int', vmax=vmax_wkt)
                display_figure_fixed_height_html(fig_w2, height_px=HEIGHT_PITCHMAP_PX, margin_px=0)
            
            # display Dot Balls row (as counts)
            c5, c6 = st.columns([1,1], gap="large")
            with c5:
                st.markdown(f"<div style='font-weight:800;'> Dot Balls — Pace</div>", unsafe_allow_html=True)
                fig_d1 = plot_grid_with_readable_labels(grid_pace_dot, f"{player_selected} — Dot Balls vs Pace", cmap='Blues', mirror=is_lhb, fmt='int', vmax=vmax_dot)
                display_figure_fixed_height_html(fig_d1, height_px=HEIGHT_PITCHMAP_PX, margin_px=0)
            with c6:
                st.markdown(f"<div style='font-weight:800;'> Dot Balls — Spin</div>", unsafe_allow_html=True)
                fig_d2 = plot_grid_with_readable_labels(grid_spin_dot, f"{player_selected} — Dot Balls vs Spin", cmap='Blues', mirror=is_lhb, fmt='int', vmax=vmax_dot)
                display_figure_fixed_height_html(fig_d2, height_px=HEIGHT_PITCHMAP_PX, margin_px=0)
            
            # -------------------- Shot productivity & control for selected batter (pf) --------------------
            # ---------- Shot productivity + SR + Dismissals + BallsPerDismissal (for selected batter only) ----------
            import plotly.express as px
            import pandas as pd
            import numpy as np
            
            # Defensive checks
            if 'pf' not in globals():
                st.error("Player frame `pf` not found. Filter your DataFrame for the selected batsman into `pf` first.")
                st.stop()
            
            # Ensure required columns exist
            if 'batruns' not in pf.columns:
                st.error("Column 'batruns' is required but not found in player frame `pf`.")
                st.stop()
            if 'shot' not in pf.columns:
                st.error("Column 'shot' is required but not found in player frame `pf`.")
                st.stop()
            
            # Build working df for this batter (keep rows that have shot info)
            df_local = pf.dropna(subset=['shot']).copy()
            if df_local.empty:
                st.info("No deliveries with 'shot' recorded for this batsman.")
            else:
                # Ensure numeric batruns
                df_local['batruns'] = pd.to_numeric(df_local['batruns'], errors='coerce').fillna(0).astype(int)
            
                # Compute dismissal flag (is_wkt) if not already present
                if 'is_wkt' not in df_local.columns:
                    # compute out_flag and dismissal_clean locally
                    df_local['out_flag'] = pd.to_numeric(df_local.get('out', 0), errors='coerce').fillna(0).astype(int)
                    df_local['dismissal_clean'] = df_local.get('dismissal', "").astype(str).str.lower().str.strip().replace({'nan':'','none':''})
                    special_runout_types = set(['run out','runout','retired','retired not out','retired out','obstructing the field'])
                    df_local['is_wkt'] = df_local.apply(
                        lambda r: 1 if (int(r.get('out_flag',0)) == 1 and str(r.get('dismissal_clean','')).strip() not in special_runout_types and str(r.get('dismissal_clean','')).strip() != '') else 0,
                        axis=1
                    )
                else:
                    df_local['is_wkt'] = pd.to_numeric(df_local['is_wkt'], errors='coerce').fillna(0).astype(int)
            
                # Total runs (batruns) by this batter (used for percentage)
                total_runs = df_local['batruns'].sum()
            
                # Group by shot to compute runs, balls, dismissals
                shot_grp = df_local.groupby('shot').agg(
                    runs_by_shot = ('batruns', 'sum'),
                    balls = ('shot', 'size'),
                    dismissals = ('is_wkt', 'sum')
                ).reset_index()
            
                # Compute derived metrics
                shot_grp['% of Runs'] = shot_grp['runs_by_shot'].apply(lambda r: (r / total_runs * 100.0) if total_runs > 0 else 0.0)
                shot_grp['SR'] = shot_grp.apply(lambda r: (r['runs_by_shot'] / r['balls'] * 100.0) if r['balls']>0 else np.nan, axis=1)
                # Balls per dismissal: if dismissals == 0 -> NaN (we'll show '-' later)
                shot_grp['BallsPerDismissal'] = shot_grp.apply(lambda r: (r['balls'] / r['dismissals']) if r['dismissals']>0 else np.nan, axis=1)
            
                # Round nicely for display
                shot_grp['% of Runs'] = shot_grp['% of Runs'].round(2)
                shot_grp['SR'] = shot_grp['SR'].round(2)
                shot_grp['BallsPerDismissal'] = shot_grp['BallsPerDismissal'].round(2)
            
                # Sort for plotting (so the biggest % of runs appear at top)
                productive_shot_df = shot_grp.sort_values('% of Runs', ascending=True)
            
                # -------------------------
                # Control % (if control column exists)
                # -------------------------
                control_df = None
                if 'control' in df_local.columns:
                    # normalize control (robust coercion to 0/1)
                    def _to_control_num(x):
                        if pd.isna(x):
                            return 0
                        try:
                            n = float(x)
                            return 1 if int(n) != 0 else 0
                        except:
                            s = str(x).strip().lower()
                            if s in ('1','true','t','y','yes','controlled','c','ok'):
                                return 1
                            return 0
                    df_local['control_num'] = df_local['control'].apply(_to_control_num).astype(int)
                    control_grp = df_local.groupby('shot').agg(
                        total_shots = ('control_num','size'),
                        controlled_shots = ('control_num','sum')
                    ).reset_index()
                    control_grp['Control Percentage'] = control_grp.apply(
                        lambda r: (r['controlled_shots'] / r['total_shots'] * 100.0) if r['total_shots']>0 else 0.0,
                        axis=1
                    )
                    control_grp['Control Percentage'] = control_grp['Control Percentage'].round(2)
                    control_df = control_grp.sort_values('Control Percentage', ascending=True)
            
                # -------------------------
                # Plotting: left = Productive Shots (% of runs) with hover showing SR, Dismissals & BallsPerDismissal
                # -------------------------
                col1, col2 = st.columns(2)
            
                with col1:
                    st.markdown("### Most Productive Shots")
                    if productive_shot_df.empty:
                        st.info("No shot data to plot.")
                    else:
                        fig1 = px.bar(
                            productive_shot_df,
                            x='% of Runs',
                            y='shot',
                            orientation='h',
                            color='% of Runs',
                            labels={'shot': 'Shot Type', '% of Runs': '% of Runs'},
                            height=600,
                            hover_data={
                                'runs_by_shot': True,
                                'balls': True,
                                'SR': True,
                                'dismissals': True,
                                'BallsPerDismissal': True,
                                '% of Runs': ':.2f'
                            }
                        )
                        fig1.update_layout(
                            margin=dict(l=180, r=40, t=40, b=40),
                            xaxis_title='% of Runs',
                            yaxis_title=None,
                        )
                        # show percentage inside bars with 2 decimals
                        fig1.update_traces(texttemplate='%{x:.2f}%', textposition='inside', hovertemplate=None)
                        fig1.update_yaxes(categoryorder='total ascending')
                        st.plotly_chart(fig1, use_container_width=True)
            
                # -------------------------
                # Plotting: right = Control % by shot (if available)
                # -------------------------
                with col2:
                    st.markdown("### Control Percentage by Shot")
                    if control_df is None:
                        st.info("No `control` column available for this batter; skipping Control % chart.")
                    else:
                        fig2 = px.bar(
                            control_df,
                            x='Control Percentage',
                            y='shot',
                            orientation='h',
                            color='Control Percentage',
                            labels={'shot': 'Shot Type', 'Control Percentage': '% Controlled'},
                            height=520,
                            hover_data={'total_shots': True, 'controlled_shots': True, 'Control Percentage': ':.2f'}
                        )
                        fig2.update_layout(
                            margin=dict(l=160, r=30, t=40, b=40),
                            xaxis_title='Control Percentage',
                            yaxis_title=None,
                        )
                        fig2.update_traces(texttemplate='%{x:.2f}%', textposition='inside', hovertemplate=None)
                        fig2.update_yaxes(categoryorder='total ascending')
                        st.plotly_chart(fig2, use_container_width=True)
                
                # ---------- Underlying numbers (rounded) ----------
                st.markdown("#### Underlying numbers")
                prod_show = productive_shot_df[['shot', 'runs_by_shot', 'balls', 'SR', 'dismissals', 'BallsPerDismissal', '% of Runs']].copy()
                prod_show = prod_show.rename(columns={
                    'shot': 'Shot',
                    'runs_by_shot': 'Runs (batruns)',
                    'balls': 'Balls',
                    'SR': 'SR (%)',
                    'dismissals': 'Dismissals',
                    'BallsPerDismissal': 'Balls per Dismissal',
                    '% of Runs': '% of Runs'
                })
                # Replace NaN BallsPerDismissal with '-' for display
                prod_show['Balls per Dismissal'] = prod_show['Balls per Dismissal'].apply(lambda x: '-' if pd.isna(x) else (round(x, 2) if not isinstance(x, str) else x)).astype(str)  # Convert entire column to string
                prod_show['SR (%)'] = prod_show['SR (%)'].apply(lambda x: '-' if pd.isna(x) else round(x,2))
                prod_show['% of Runs'] = prod_show['% of Runs'].round(2)
                prod_show = prod_show.sort_values('% of Runs', ascending=False).reset_index(drop=True)
                
                st.dataframe(prod_show, use_container_width=True)
                
                # If control table exists, show it too
                if control_df is not None:
                    ctrl_show = control_df[['shot','total_shots','controlled_shots','Control Percentage']].copy()
                    ctrl_show = ctrl_show.rename(columns={
                        'shot': 'Shot',
                        'total_shots': 'Total Shots',
                        'controlled_shots': 'Controlled Shots',
                        'Control Percentage': '% Controlled'
                    })
                    st.dataframe(ctrl_show, use_container_width=True)
            # ---------- end shot productivity snippet ----------
            # -------------------- 4 pitchmaps: SR and Control % (Pace vs Spin) --------------------
            # ---------------- Improved pitchmaps: nicer colors + show '-' for empty cells ----------------
            # ---------------- Light-colour pitchmaps (SR & Control) ----------------
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.patheffects as mpatheffects
            from matplotlib.colors import LinearSegmentedColormap
            import pandas as pd
            
            # Defensive existence / fallbacks
            if 'pf_pace' not in globals() or 'pf_spin' not in globals():
                if 'pf' in globals() and 'COL_BOWL_KIND' in globals():
                    pf_pace = pf[pf[COL_BOWL_KIND].astype(str).str.contains('pace', na=False)].copy()
                    pf_spin = pf[pf[COL_BOWL_KIND].astype(str).str.contains('spin', na=False)].copy()
                else:
                    pf_pace = pd.DataFrame()
                    pf_spin = pd.DataFrame()
            
            # column names / maps (use your app's names if different)
            COL_LINE = globals().get('COL_LINE', 'line')
            COL_LENGTH = globals().get('COL_LENGTH', 'length')
            COL_RUNS = globals().get('COL_RUNS', 'batruns')
            NOBALL_COL = 'noball'
            WIDE_COL = 'wide'
            CONTROL_COL = 'control'
            
            LINE_MAP = globals().get('LINE_MAP', {
                'WIDE_OUTSIDE_OFFSTUMP': 0, 'OUTSIDE_OFFSTUMP': 1, 'ON_THE_STUMPS': 2, 'DOWN_LEG': 3, 'WIDE_DOWN_LEG': 4
            })
            LENGTH_MAP = globals().get('LENGTH_MAP', {
                'SHORT': 0, 'SHORT_OF_A_GOOD_LENGTH': 1, 'GOOD_LENGTH': 2, 'FULL': 3, 'YORKER': 4, 'FULL_TOSS': 4
            })
            
            def _is_legal_row(r):
                try:
                    if NOBALL_COL in r.index and int(r.get(NOBALL_COL, 0)) != 0:
                        return False
                except Exception:
                    pass
                try:
                    if WIDE_COL in r.index and int(r.get(WIDE_COL, 0)) != 0:
                        return False
                except Exception:
                    pass
                return True
            
            def build_sr_and_counts(df_local):
                sr_grid = np.full((5,5), np.nan, dtype=float)
                runs_grid = np.zeros((5,5), dtype=float)
                balls_grid = np.zeros((5,5), dtype=int)
                if df_local is None or df_local.shape[0] == 0:
                    return sr_grid, balls_grid
                if any(c not in df_local.columns for c in [COL_LINE, COL_LENGTH, COL_RUNS]):
                    return sr_grid, balls_grid
                
                cols = [COL_LINE, COL_LENGTH, COL_RUNS]
                if NOBALL_COL in df_local.columns: cols.append(NOBALL_COL)
                if WIDE_COL in df_local.columns: cols.append(WIDE_COL)
                plot_df = df_local[cols].copy()
                
                for _, r in plot_df.iterrows():
                    if pd.isna(r[COL_LINE]) or pd.isna(r[COL_LENGTH]):
                        continue
                    if not _is_legal_row(r):
                        continue
                    li = LINE_MAP.get(r[COL_LINE], None)
                    le = LENGTH_MAP.get(r[COL_LENGTH], None)
                    if li is None or le is None:
                        continue
                    try:
                        runs_here = float(r.get(COL_RUNS, 0) or 0)
                    except:
                        runs_here = 0.0
                    runs_grid[le, li] += runs_here
                    balls_grid[le, li] += 1
                
                mask = balls_grid > 0
                sr_grid[mask] = (runs_grid[mask] / balls_grid[mask]) * 100.0
                sr_grid[mask] = np.round(sr_grid[mask], 2)
                return sr_grid, balls_grid
            
            def _to_control_num(x):
                if pd.isna(x):
                    return 0
                try:
                    n = float(x)
                    if np.isfinite(n):
                        return 1 if int(n) != 0 else 0
                except:
                    pass
                s = str(x).strip().lower()
                if s in ('1','true','t','y','yes','controlled','c','ok'):
                    return 1
                if s in ('0','false','f','n','no','uncontrolled','u'):
                    return 0
                return 1 if s != '' else 0
            
            def build_control_and_counts(df_local):
                ctrl_grid = np.full((5,5), np.nan, dtype=float)
                total_grid = np.zeros((5,5), dtype=int)
                controlled_grid = np.zeros((5,5), dtype=int)
                if df_local is None or df_local.shape[0] == 0:
                    return ctrl_grid, total_grid
                if any(c not in df_local.columns for c in [COL_LINE, COL_LENGTH]):
                    return ctrl_grid, total_grid
                
                cols = [COL_LINE, COL_LENGTH]
                if CONTROL_COL in df_local.columns: cols.append(CONTROL_COL)
                if NOBALL_COL in df_local.columns: cols.append(NOBALL_COL)
                if WIDE_COL in df_local.columns: cols.append(WIDE_COL)
                plot_df = df_local[cols].copy()
                
                if CONTROL_COL in plot_df.columns:
                    plot_df[CONTROL_COL] = plot_df[CONTROL_COL].apply(_to_control_num).astype(int)
                
                for _, r in plot_df.iterrows():
                    if pd.isna(r[COL_LINE]) or pd.isna(r[COL_LENGTH]):
                        continue
                    if not _is_legal_row(r):
                        continue
                    li = LINE_MAP.get(r[COL_LINE], None)
                    le = LENGTH_MAP.get(r[COL_LENGTH], None)
                    if li is None or le is None:
                        continue
                    total_grid[le, li] += 1
                    if CONTROL_COL in plot_df.columns and int(r.get(CONTROL_COL, 0)) == 1:
                        controlled_grid[le, li] += 1
                
                mask = total_grid > 0
                ctrl_grid[mask] = (controlled_grid[mask] / total_grid[mask]) * 100.0
                ctrl_grid[mask] = np.round(ctrl_grid[mask], 2)
                return ctrl_grid, total_grid
            
            # Create soft/light colormaps
            def make_light_cmap(name, colors):
                return LinearSegmentedColormap.from_list(name, colors)
            
            # light skin / pink for SR, light blue for Control
            cmap_sr_light = make_light_cmap('sr_light', ['#fff3e6', '#ffd6b3', '#ffc2a8']) # soft skin / peach tones
            cmap_ctrl_light = make_light_cmap('ctrl_light', ['#f0f8ff', '#d9efff', '#bfe6ff']) # very light blues
            
            def plot_grid_light(grid, counts_grid, title, cmap, mirror=False, fmt='float', vmax=None):
                disp = np.fliplr(grid) if mirror else grid.copy()
                counts_disp = np.fliplr(counts_grid) if mirror else counts_grid.copy()
                
                # set NaNs as light gray
                cmap = cmap.copy()
                cmap.set_bad('#f7f7f7')
                
                finite_vals = disp[np.isfinite(disp)]
                real_vmax = float(np.nanmax(finite_vals)) if finite_vals.size>0 else (vmax if vmax is not None else 1.0)
                vmax_use = float(vmax) if (vmax is not None and vmax>0) else (real_vmax if real_vmax>0 else 1.0)
                
                fig, ax = plt.subplots(figsize=(6,9), dpi=150)
                im = ax.imshow(disp, origin='lower', cmap=cmap, vmin=0, vmax=vmax_use)
                ax.set_xticks(range(5)); ax.set_yticks(range(5))
                xticks_base = ['Wide Out Off','Outside Off','On Stumps','Down Leg','Wide Down Leg']
                xticks = list(reversed(xticks_base)) if mirror else xticks_base
                ax.set_xticklabels(xticks, rotation=36, ha='right')
                ax.set_yticklabels(['Short','Back of Length','Good','Full','Yorker'])
                
                for i in range(5):
                    for j in range(5):
                        val = disp[i,j]
                        cnt = int(counts_disp[i,j]) if not np.isnan(counts_disp[i,j]) else 0
                        if cnt == 0 or not np.isfinite(val):
                            lab = '-' # no deliveries
                            txt_color = 'black'
                        else:
                            if fmt == 'pct':
                                lab = f"{val:.2f}%"
                            elif fmt == 'int':
                                lab = f"{int(val)}"
                            else:
                                lab = f"{val:.2f}"
                            try:
                                intensity = float(val) / float(vmax_use) if vmax_use > 0 else 0.0
                            except:
                                intensity = 0.0
                            txt_color = 'black' if intensity < 0.6 else 'white'
                        txt = ax.text(j, i, lab, ha='center', va='center', color=txt_color, fontsize=12, fontweight='bold')
                        stroke_col = 'white' if txt_color=='black' else 'black'
                        txt.set_path_effects([mpatheffects.Stroke(linewidth=2, foreground=stroke_col), mpatheffects.Normal()])
                
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cbar.ax.tick_params(labelsize=9)
                plt.title(title, pad=6, fontsize=12)
                plt.tight_layout(pad=0)
                fig.subplots_adjust(top=0.95, bottom=0.02, left=0.06, right=0.98)
                return fig
            
            # Build grids
            sr_pace, counts_pace = build_sr_and_counts(pf_pace)
            sr_spin, counts_spin = build_sr_and_counts(pf_spin)
            ctrl_pace, total_pace = build_control_and_counts(pf_pace)
            ctrl_spin, total_spin = build_control_and_counts(pf_spin)
            
            # vmax choices
            vmax_sr = max(np.nanmax(sr_pace[np.isfinite(sr_pace)]) if np.any(np.isfinite(sr_pace)) else 0,
                          np.nanmax(sr_spin[np.isfinite(sr_spin)]) if np.any(np.isfinite(sr_spin)) else 0, 10.0)
            vmax_ctrl = 100.0
            
            # Display maps
            c1, c2 = st.columns([1,1], gap="large")
            with c1:
                st.markdown("<div style='font-weight:800;'>Strike Rate — Pace</div>", unsafe_allow_html=True)
                fig_sr_pace = plot_grid_light(sr_pace, counts_pace, f"{player_selected} — SR% vs Pace", cmap_sr_light, mirror=is_lhb, fmt='float', vmax=vmax_sr)
                display_figure_fixed_height_html(fig_sr_pace, height_px=HEIGHT_PITCHMAP_PX, margin_px=0)
            with c2:
                st.markdown("<div style='font-weight:800;'>Strike Rate — Spin</div>", unsafe_allow_html=True)
                fig_sr_spin = plot_grid_light(sr_spin, counts_spin, f"{player_selected} — SR% vs Spin", cmap_sr_light, mirror=is_lhb, fmt='float', vmax=vmax_sr)
                display_figure_fixed_height_html(fig_sr_spin, height_px=HEIGHT_PITCHMAP_PX, margin_px=0)
            
            c3, c4 = st.columns([1,1], gap="large")
            with c3:
                st.markdown("<div style='font-weight:800;'> Control % — Pace</div>", unsafe_allow_html=True)
                fig_ctrl_pace = plot_grid_light(ctrl_pace, total_pace, f"{player_selected} — Control% vs Pace", cmap_ctrl_light, mirror=is_lhb, fmt='pct', vmax=vmax_ctrl)
                display_figure_fixed_height_html(fig_ctrl_pace, height_px=HEIGHT_PITCHMAP_PX, margin_px=0)
            with c4:
                st.markdown("<div style='font-weight:800;'> Control % — Spin</div>", unsafe_allow_html=True)
                fig_ctrl_spin = plot_grid_light(ctrl_spin, total_spin, f"{player_selected} — Control% vs Spin", cmap_ctrl_light, mirror=is_lhb, fmt='pct', vmax=vmax_ctrl)
                display_figure_fixed_height_html(fig_ctrl_spin, height_px=HEIGHT_PITCHMAP_PX, margin_px=0)
            # ---------------- end light-colour pitchmaps ----------------
# ---------------- end improved pitchmaps snippet ----------------
        
    # -------------------- end pitchmaps snippet --------------------

    else:
        st.markdown(f"<div style='font-size:20px; font-weight:800; color:#111;'> Bowling — {player_selected}</div>", unsafe_allow_html=True)
        
        # ============================================================================
        # BOWLER ANALYSIS - WAGON WHEEL FUNCTIONS (MATCHING BATTER ANALYSIS STYLE)
        # ============================================================================
        
        def draw_bowler_wagon_if_available(df_wagon, bowler_name, normalize_to_rhb=True):
            """
            Wrapper that calls draw_cricket_field_with_run_totals_requested for BOWLER analysis.
            Shows runs conceded by the bowler in different zones.
            - normalize_to_rhb: True => request RHB-normalised output (legacy behaviour).
                                False => request true handedness visualization (LHB will appear mirrored).
            """
            import matplotlib.pyplot as plt
            import streamlit as st
            import inspect
            from matplotlib.figure import Figure as MplFigure
        
            # Defensive check
            if not isinstance(df_wagon, pd.DataFrame) or df_wagon.empty:
                st.warning("No wagon data available to draw.")
                return
        
            # Decide handedness (for UI messages / debugging)
            batting_style_val = None
            if 'bat_hand' in df_wagon.columns:
                try:
                    batting_style_val = df_wagon['bat_hand'].dropna().iloc[0]
                except Exception:
                    batting_style_val = None
            is_lhb = isinstance(batting_style_val, str) and batting_style_val.strip().upper().startswith('L')
        
            # Check function signature
            draw_fn = globals().get('draw_cricket_field_with_run_totals_requested', None)
            if draw_fn is None or not callable(draw_fn):
                st.warning("Wagon chart function not found; please ensure `draw_cricket_field_with_run_totals_requested` is defined earlier.")
                return
        
            try:
                sig = inspect.signature(draw_fn)
                if 'normalize_to_rhb' in sig.parameters:
                    # call with the explicit flag (preferred)
                    fig = draw_fn(df_wagon, bowler_name, normalize_to_rhb=normalize_to_rhb)
                else:
                    # older signature: call without flag (maintain legacy behaviour)
                    fig = draw_fn(df_wagon, bowler_name)
        
                # If the function returned a Matplotlib fig — display it
                if isinstance(fig, MplFigure):
                    safe_fn = globals().get('safe_st_pyplot', None)
                    if callable(safe_fn):
                        try:
                            safe_fn(fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
                        except Exception:
                            st.pyplot(fig)
                    else:
                        st.pyplot(fig)
                    return
        
                # If function returned None, it may have drawn to current fig; capture that
                if fig is None:
                    mpl_fig = plt.gcf()
                    # If figure has axes and content, display it
                    if isinstance(mpl_fig, MplFigure) and len(mpl_fig.axes) > 0:
                        safe_fn = globals().get('safe_st_pyplot', None)
                        if callable(safe_fn):
                            try:
                                safe_fn(mpl_fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
                            except Exception:
                                st.pyplot(mpl_fig)
                        else:
                            st.pyplot(mpl_fig)
                    return
        
                # If function returned a Plotly figure (rare), display it
                if isinstance(fig, go.Figure):
                    try:
                        fig.update_yaxes(scaleanchor="x", scaleratio=1)
                    except Exception:
                        pass
                    st.plotly_chart(fig, use_container_width=True)
                    return
        
                # Unknown return — just state it
                st.warning("Wagon draw function executed but returned an unexpected type; nothing displayed.")
            except Exception as e:
                st.error(f"Wagon drawing function raised: {e}")
        
            def draw_bowler_caught_dismissals_wagon(df_wagon, bowler_name, normalize_to_rhb=True):
                """
                Plotly wagon chart for caught dismissals FOR BOWLER.
                Clean, single-render, no duplication.
                """
                import plotly.graph_objects as go
                import numpy as np
                import pandas as pd
                import streamlit as st
            
                # ---------------- Defensive checks ----------------
                if not isinstance(df_wagon, pd.DataFrame) or df_wagon.empty:
                    return
            
                if not {'dismissal', 'bowl', 'wagonX', 'wagonY'}.issubset(df_wagon.columns):
                    st.warning("Required columns missing for caught dismissal wagon.")
                    return
            
                # ---------------- Filter caught dismissals ----------------
                caught_df = df_wagon[
                    (df_wagon['bowl'] == bowler_name) &
                    (df_wagon['dismissal'].astype(str).str.lower().str.contains('caught', na=False))
                ].copy()
            
                if caught_df.empty:
                    st.info(f"No caught dismissals for {bowler_name}.")
                    return
            
                caught_df['wagonX'] = pd.to_numeric(caught_df['wagonX'], errors='coerce')
                caught_df['wagonY'] = pd.to_numeric(caught_df['wagonY'], errors='coerce')
                caught_df = caught_df.dropna(subset=['wagonX', 'wagonY'])
                if caught_df.empty:
                    return
            
                # ---------------- Normalize coordinates ----------------
                center, radius = 184.0, 184.0
                x = (caught_df['wagonX'] - center) / radius
                y = - (caught_df['wagonY'] - center) / radius
            
                # Detect LHB
                is_lhb = False
                if 'bat_hand' in caught_df.columns:
                    try:
                        is_lhb = caught_df['bat_hand'].dropna().iloc[0].upper().startswith('L')
                    except Exception:
                        pass
            
                if (not normalize_to_rhb) and is_lhb:
                    x = -x
            
                mask = np.sqrt(x**2 + y**2) <= 1
                x, y = x[mask], y[mask]
                caught_df = caught_df.loc[mask]
            
                if caught_df.empty:
                    return
            
                # ---------------- Hover data ----------------
                hover_cols = [c for c in ['bat', 'bowl', 'line', 'length', 'shot', 'dismissal'] if c in caught_df.columns]
                customdata = caught_df[hover_cols].astype(str).values
            
                hovertemplate = (
                    "<br>".join(
                        [f"<b>{col}:</b> %{{customdata[{i}]}}" for i, col in enumerate(hover_cols)]
                    ) + "<extra></extra>"
                )
            
                # ---------------- Build figure ----------------
                fig = go.Figure()
            
                def add_circle(x0, y0, x1, y1, **kw):
                    if (not normalize_to_rhb) and is_lhb:
                        x0, x1 = -x1, -x0
                    fig.add_shape(type="circle", x0=x0, y0=y0, x1=x1, y1=y1, **kw)
            
                # Field
                add_circle(-1, -1, 1, 1, fillcolor="#228B22", line_color="black", layer="below")
                add_circle(-0.5, -0.5, 0.5, 0.5, fillcolor="#66bb6a", line_color="white", layer="below")
            
                # Pitch
                px0, px1 = (-0.04, 0.04)
                if (not normalize_to_rhb) and is_lhb:
                    px0, px1 = -px1, -px0
                fig.add_shape(type="rect", x0=px0, y0=-0.08, x1=px1, y1=0.08,
                              fillcolor="tan", line_color=None)
            
                # Radials
                for ang in np.linspace(0, 2*np.pi, 8, endpoint=False):
                    xe, ye = np.cos(ang), np.sin(ang)
                    if (not normalize_to_rhb) and is_lhb:
                        xe = -xe
                    fig.add_trace(go.Scatter(
                        x=[0, xe], y=[0, ye],
                        mode="lines", line=dict(color="white", width=1),
                        opacity=0.25, showlegend=False
                    ))
            
                # Points (ONCE)
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode="markers",
                    marker=dict(color="red", size=12, line=dict(color="black", width=1.5)),
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                    name="Caught Dismissals"
                ))
            
                fig.update_layout(
                    xaxis=dict(range=[-1.2, 1.2], visible=False),
                    yaxis=dict(range=[-1.2, 1.2], visible=False, scaleanchor="x"),
                    height=700,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=True
                )
            
                st.plotly_chart(fig, use_container_width=True)

        
            # Circles and pitch: if normalize_to_rhb True, use canonical RHB shapes; if False and LHB, flip x coords for shapes
            def add_circle_shape(fig_obj, x0_raw, y0, x1_raw, y1, **kwargs):
                if (not normalize_to_rhb) and is_lhb:
                    tx0 = -x1_raw
                    tx1 = -x0_raw
                else:
                    tx0 = x0_raw
                    tx1 = x1_raw
                x0_, x1_ = min(tx0, tx1), max(tx0, tx1)
                fig_obj.add_shape(type="circle", xref="x", yref="y", x0=x0_, y0=y0, x1=x1_, y1=y1, **kwargs)
        
            add_circle_shape(fig, -1, -1, 1, 1, fillcolor="#228B22", line_color="black", opacity=1, layer="below")
            add_circle_shape(fig, -0.5, -0.5, 0.5, 0.5, fillcolor="#66bb6a", line_color="white", opacity=1, layer="below")
        
            # Pitch rectangle (tan)
            pitch_x0, pitch_x1 = (-0.04, 0.04)
            if (not normalize_to_rhb) and is_lhb:
                pitch_x0, pitch_x1 = (-pitch_x1, -pitch_x0)
            fig.add_shape(type="rect", x0=pitch_x0, y0=-0.08, x1=pitch_x1, y1=0.08,
                          fillcolor="tan", line_color=None, opacity=1, layer="above")
        
            # Radial lines (mirror endpoints if LHB & not normalised)
            angles = np.linspace(0, 2*np.pi, 9)[:-1]
            for angle in angles:
                x_end = np.cos(angle)
                y_end = np.sin(angle)
                if (not normalize_to_rhb) and is_lhb:
                    x_end = -x_end
                fig.add_trace(go.Scatter(x=[0, x_end], y=[0, y_end],
                                         mode='lines', line=dict(color='white', width=1), opacity=0.25, showlegend=False))
        
            # Plot dismissal points (use x_vals,y_vals filtered earlier)
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                marker=dict(color='red', size=12, line=dict(color='black', width=1.5)),
                customdata=customdata,
                hovertemplate=(
                    "<br>".join([f"<b>{col}:</b> %{{customdata[{i}]}}" for i, col in enumerate(customdata_cols)]) +
                    "<extra></extra>"
                ),
                name='Caught Dismissal Locations'
            ))
        
            # Layout & equal aspect scaling so mirror is obvious
            axis_range = 1.2
            fig.update_layout(
                xaxis=dict(range=[-axis_range, axis_range], showgrid=False, zeroline=False, visible=False),
                yaxis=dict(range=[-axis_range, axis_range], showgrid=False, zeroline=False, visible=False),
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                width=800,
                height=800,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            try:
                fig.update_yaxes(scaleanchor="x", scaleratio=1)
            except Exception:
                pass
        
            # Finally display the plotly figure
            st.plotly_chart(fig, use_container_width=True)
        
            # Circles and pitch: if normalize_to_rhb True, use canonical RHB shapes; if False and LHB, flip x coords for shapes
            def add_circle_shape(fig_obj, x0_raw, y0, x1_raw, y1, **kwargs):
                if (not normalize_to_rhb) and is_lhb:
                    tx0 = -x1_raw
                    tx1 = -x0_raw
                else:
                    tx0 = x0_raw
                    tx1 = x1_raw
                x0_, x1_ = min(tx0, tx1), max(tx0, tx1)
                fig_obj.add_shape(type="circle", xref="x", yref="y", x0=x0_, y0=y0, x1=x1_, y1=y1, **kwargs)
        
            add_circle_shape(fig, -1, -1, 1, 1, fillcolor="#228B22", line_color="black", opacity=1, layer="below")
            add_circle_shape(fig, -0.5, -0.5, 0.5, 0.5, fillcolor="#66bb6a", line_color="white", opacity=1, layer="below")
        
            # Pitch rectangle (tan)
            pitch_x0, pitch_x1 = (-0.04, 0.04)
            if (not normalize_to_rhb) and is_lhb:
                pitch_x0, pitch_x1 = (-pitch_x1, -pitch_x0)
            fig.add_shape(type="rect", x0=pitch_x0, y0=-0.08, x1=pitch_x1, y1=0.08,
                          fillcolor="tan", line_color=None, opacity=1, layer="above")
        
            # Radial lines (mirror endpoints if LHB & not normalised)
            angles = np.linspace(0, 2*np.pi, 9)[:-1]
            for angle in angles:
                x_end = np.cos(angle)
                y_end = np.sin(angle)
                if (not normalize_to_rhb) and is_lhb:
                    x_end = -x_end
                fig.add_trace(go.Scatter(x=[0, x_end], y=[0, y_end],
                                         mode='lines', line=dict(color='white', width=1), opacity=0.25, showlegend=False))
        
            # Plot dismissal points (use x_vals,y_vals filtered earlier)
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                marker=dict(color='red', size=12, line=dict(color='black', width=1.5)),
                customdata=customdata,
                hovertemplate=(
                    "<br>".join([f"<b>{col}:</b> %{{customdata[{i}]}}" for i, col in enumerate(customdata_cols)]) +
                    "<extra></extra>"
                ),
                name='Caught Dismissal Locations'
            ))
        
            # Layout & equal aspect scaling so mirror is obvious
            axis_range = 1.2
            fig.update_layout(
                xaxis=dict(range=[-axis_range, axis_range], showgrid=False, zeroline=False, visible=False),
                yaxis=dict(range=[-axis_range, axis_range], showgrid=False, zeroline=False, visible=False),
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                width=800,
                height=800,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            try:
                fig.update_yaxes(scaleanchor="x", scaleratio=1)
            except Exception:
                pass
        
            # Finally display the plotly figure
            st.plotly_chart(fig, use_container_width=True)
        
            # Circles and pitch: if normalize_to_rhb True, use canonical RHB shapes; if False and LHB, flip x coords for shapes
            def add_circle_shape(fig_obj, x0_raw, y0, x1_raw, y1, **kwargs):
                if (not normalize_to_rhb) and is_lhb:
                    tx0 = -x1_raw
                    tx1 = -x0_raw
                else:
                    tx0 = x0_raw
                    tx1 = x1_raw
                x0_, x1_ = min(tx0, tx1), max(tx0, tx1)
                fig_obj.add_shape(type="circle", xref="x", yref="y", x0=x0_, y0=y0, x1=x1_, y1=y1, **kwargs)
        
            add_circle_shape(fig, -1, -1, 1, 1, fillcolor="#228B22", line_color="black", opacity=1, layer="below")
            add_circle_shape(fig, -0.5, -0.5, 0.5, 0.5, fillcolor="#66bb6a", line_color="white", opacity=1, layer="below")
        
            # Pitch rectangle (tan)
            pitch_x0, pitch_x1 = (-0.04, 0.04)
            if (not normalize_to_rhb) and is_lhb:
                pitch_x0, pitch_x1 = (-pitch_x1, -pitch_x0)
            fig.add_shape(type="rect", x0=pitch_x0, y0=-0.08, x1=pitch_x1, y1=0.08,
                          fillcolor="tan", line_color=None, opacity=1, layer="above")
        
            # Radial lines (mirror endpoints if LHB & not normalised)
            angles = np.linspace(0, 2*np.pi, 9)[:-1]
            for angle in angles:
                x_end = np.cos(angle)
                y_end = np.sin(angle)
                if (not normalize_to_rhb) and is_lhb:
                    x_end = -x_end
                fig.add_trace(go.Scatter(x=[0, x_end], y=[0, y_end],
                                         mode='lines', line=dict(color='white', width=1), opacity=0.25, showlegend=False))
        
            # Plot dismissal points (use x_vals,y_vals filtered earlier)
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                marker=dict(color='red', size=12, line=dict(color='black', width=1.5)),
                customdata=customdata,
                hovertemplate=(
                    "<br>".join([f"<b>{col}:</b> %{{customdata[{i}]}}" for i, col in enumerate(customdata_cols)]) +
                    "<extra></extra>"
                ),
                name='Caught Dismissal Locations'
            ))
        
            # Layout & equal aspect scaling so mirror is obvious
            axis_range = 1.2
            fig.update_layout(
                xaxis=dict(range=[-axis_range, axis_range], showgrid=False, zeroline=False, visible=False),
                yaxis=dict(range=[-axis_range, axis_range], showgrid=False, zeroline=False, visible=False),
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                width=800,
                height=800,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            try:
                fig.update_yaxes(scaleanchor="x", scaleratio=1)
            except Exception:
                pass
        
            # Finally display the plotly figure
            st.plotly_chart(fig, use_container_width=True)
        
            # Circles and pitch: if normalize_to_rhb True, use canonical RHB shapes; if False and LHB, flip x coords for shapes
            def add_circle_shape(fig_obj, x0_raw, y0, x1_raw, y1, **kwargs):
                if (not normalize_to_rhb) and is_lhb:
                    tx0 = -x1_raw
                    tx1 = -x0_raw
                else:
                    tx0 = x0_raw
                    tx1 = x1_raw
                x0_, x1_ = min(tx0, tx1), max(tx0, tx1)
                fig_obj.add_shape(type="circle", xref="x", yref="y", x0=x0_, y0=y0, x1=x1_, y1=y1, **kwargs)
        
            add_circle_shape(fig, -1, -1, 1, 1, fillcolor="#228B22", line_color="black", opacity=1, layer="below")
            add_circle_shape(fig, -0.5, -0.5, 0.5, 0.5, fillcolor="#66bb6a", line_color="white", opacity=1, layer="below")
        
            # Pitch rectangle (tan)
            pitch_x0, pitch_x1 = (-0.04, 0.04)
            if (not normalize_to_rhb) and is_lhb:
                pitch_x0, pitch_x1 = (-pitch_x1, -pitch_x0)
            fig.add_shape(type="rect", x0=pitch_x0, y0=-0.08, x1=pitch_x1, y1=0.08,
                          fillcolor="tan", line_color=None, opacity=1, layer="above")
        
            # Radial lines (mirror endpoints if LHB & not normalised)
            angles = np.linspace(0, 2*np.pi, 9)[:-1]
            for angle in angles:
                x_end = np.cos(angle)
                y_end = np.sin(angle)
                if (not normalize_to_rhb) and is_lhb:
                    x_end = -x_end
                fig.add_trace(go.Scatter(x=[0, x_end], y=[0, y_end],
                                         mode='lines', line=dict(color='white', width=1), opacity=0.25, showlegend=False))
        
            # Plot dismissal points (use x_vals,y_vals filtered earlier)
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                marker=dict(color='red', size=12, line=dict(color='black', width=1.5)),
                customdata=customdata,
                hovertemplate=(
                    "<br>".join([f"<b>{col}:</b> %{customdata[{i}]}" for i, col in enumerate(customdata_cols)]) +
                    "<extra></extra>"
                ),
                name='Caught Dismissal Locations'
            ))
        
            # Layout & equal aspect scaling so mirror is obvious
            axis_range = 1.2
            fig.update_layout(
                xaxis=dict(range=[-axis_range, axis_range], showgrid=False, zeroline=False, visible=False),
                yaxis=dict(range=[-axis_range, axis_range], showgrid=False, zeroline=False, visible=False),
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                width=800,
                height=800,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            try:
                fig.update_yaxes(scaleanchor="x", scaleratio=1)
            except Exception:
                pass
        
            # Finally display the plotly figure
            st.plotly_chart(fig, use_container_width=True)
        
        # require bdf in globals
        if 'bdf' not in globals():
            bdf = as_dataframe(df) # Use df if bdf not defined
        
        # For bowler, adapt runs_col to bowler runs
        runs_col = safe_get_col(bdf, ['bowlruns', 'score', 'runs'])
        if runs_col is None:
            st.error("No runs column found for bowling.")
            st.stop()
        
        # coerce runs col
        bdf[runs_col] = pd.to_numeric(bdf.get(runs_col, 0), errors='coerce').fillna(0).astype(int)
        pf[runs_col] = pd.to_numeric(pf.get(runs_col, 0), errors='coerce').fillna(0).astype(int)
        
        # For bowling, group by bat_hand or other
        if COL_BAT_HAND in pf.columns:
            pf[COL_BAT_HAND] = pf[COL_BAT_HAND].astype(str).str.lower().fillna('unknown')
            kinds = sorted(pf[COL_BAT_HAND].dropna().unique().tolist())
            group_col = COL_BAT_HAND
            group_label = 'Batter Hand'
        else:
            kinds = []
            group_col = None
        
        rows = []
        if kinds:
            for k in kinds:
                g = pf[pf[group_col] == k]
                m = compute_bowling_metrics(g, run_col=runs_col)
                m['group'] = k
                rows.append(m)
        else:
            m = compute_bowling_metrics(pf, run_col=runs_col)
            m['group'] = 'unknown'
            rows.append(m)
        bk_df = pd.DataFrame(rows).set_index('group')
        bk_df.index.name = group_label if group_label else 'Group'
        
        st.markdown("<div style='font-weight:700; font-size:15px;'> Performance by batter hand </div>", unsafe_allow_html=True)
        st.dataframe(bk_df, use_container_width=True)
        
        # ---------- Bowling view (rearranged) ----------
        bf = bdf[bdf[COL_BOWL] == player_selected].copy()
        if bf.empty:
            st.info("No bowling rows for this bowler.")
            st.stop()
        
        # choose runs column for bowler frame
        if 'bowlruns' in bf.columns:
            bf_runs_col = 'bowlruns'
            bf[bf_runs_col] = pd.to_numeric(bf[bf_runs_col], errors='coerce').fillna(0).astype(int)
        elif 'score' in bf.columns:
            bf_runs_col = 'score'
            bf[bf_runs_col] = pd.to_numeric(bf[bf_runs_col], errors='coerce').fillna(0).astype(int)
        else:
            bf_runs_col = COL_RUNS
            bf[bf_runs_col] = pd.to_numeric(bf[bf_runs_col], errors='coerce').fillna(0).astype(int)
        
        # Ensure dismissal / out / is_wkt in bowler frame
        if COL_DISMISSAL in bf.columns:
            bf['dismissal_clean'] = bf[COL_DISMISSAL].astype(str).str.lower().str.strip().replace({'nan':'','none':''})
        else:
            bf['dismissal_clean'] = ''
        if COL_OUT in bf.columns:
            bf['out_flag'] = pd.to_numeric(bf[COL_OUT], errors='coerce').fillna(0).astype(int)
        else:
            bf['out_flag'] = 0
        bf['is_wkt'] = bf.apply(lambda r: 1 if is_bowler_wicket(r.get('out_flag',0), r.get('dismissal_clean','')) else 0, axis=1)
        
        # split by batter handedness
        if COL_BAT_HAND in bf.columns:
            bf_lhb = bf[bf[COL_BAT_HAND].astype(str).str.upper().str.startswith('L')].copy()
            bf_rhb = bf[bf[COL_BAT_HAND].astype(str).str.upper().str.startswith('R')].copy()
        else:
            bf_lhb = bf.iloc[0:0].copy()
            bf_rhb = bf.iloc[0:0].copy()
        
        # Collect unique values for batter hand exploration
        def unique_vals_union(col):
            vals = []
            for df in (pf, bdf):
                if col in df.columns:
                    vals.extend(df[col].dropna().astype(str).str.strip().tolist())
            vals = sorted({v for v in vals if str(v).strip() != ''})
            return vals
        
        # ---- CLEAN batter hand options ----
        raw_hands = unique_vals_union('bat_hand')
        
        # Normalize to only LHB / RHB
        clean_hands = []
        for h in raw_hands:
            if str(h).upper().startswith('L'):
                clean_hands.append('LHB')
            elif str(h).upper().startswith('R'):
                clean_hands.append('RHB')
        
        clean_hands = sorted(set(clean_hands))
        chosen_hand = st.selectbox("Batter Hand", options=clean_hands)
        
        # ---------- robust map-lookup helpers ----------
        def _norm_key(s):
            if s is None:
                return ''
            return str(s).strip().upper().replace(' ', '_').replace('-', '_')
        
        def get_map_index(map_obj, raw_val):
            if raw_val is None:
                return None
            sval = str(raw_val).strip()
            if sval == '' or sval.lower() in ('nan', 'none'):
                return None
        
            if sval in map_obj:
                return int(map_obj[sval])
            s_norm = _norm_key(sval)
            for k in map_obj:
                try:
                    if isinstance(k, str) and _norm_key(k) == s_norm:
                        return int(map_obj[k])
                except Exception:
                    continue
            for k in map_obj:
                try:
                    if isinstance(k, str) and (k.lower() in sval.lower() or sval.lower() in k.lower()):
                        return int(map_obj[k])
                except Exception:
                    continue
            return None
        
        # ---------- grids builder (adapted for bowling, but same as batting since metrics are batter performance vs bowler) ----------
        def build_pitch_grids(df_in, line_col_name='line', length_col_name='length', runs_col_candidates=('bowlruns', 'score'),
                              control_col='control', dismissal_col='dismissal'):
            if 'length_map' in globals() and isinstance(length_map, dict) and len(length_map) > 0:
                try:
                    max_idx = max(int(v) for v in length_map.values())
                    n_rows = max(5, max_idx + 1)
                except Exception:
                    n_rows = 5
            else:
                n_rows = 5
                st.warning("length_map not found; defaulting to 5 rows.")
        
            length_vals = df_in.get(length_col_name, pd.Series()).dropna().astype(str).str.lower().unique()
            if any('full toss' in val for val in length_vals):
                n_rows = max(n_rows, 6)
        
            n_cols = 5
        
            count = np.zeros((n_rows, n_cols), dtype=int)
            bounds = np.zeros((n_rows, n_cols), dtype=int)
            dots = np.zeros((n_rows, n_cols), dtype=int)
            runs = np.zeros((n_rows, n_cols), dtype=float)
            wkt = np.zeros((n_rows, n_cols), dtype=int)
            ctrl_not = np.zeros((n_rows, n_cols), dtype=int)
        
            # choose runs column present (for bowling, prefer bowlruns)
            runs_col = None
            for c in runs_col_candidates:
                if c in df_in.columns:
                    runs_col = c
                    break
            if runs_col is None:
                runs_col = None # will use 0
        
            wkt_tokens = {'caught', 'bowled', 'stumped', 'lbw','leg before wicket','hit wicket'}
            dismissal_series = df_in[dismissal_col].fillna('').astype(str).str.lower()
            for _, row in df_in.iterrows():
                li = get_map_index(line_map, row.get(line_col_name, None)) if 'line_map' in globals() else None
                le = get_map_index(length_map, row.get(length_col_name, None)) if 'length_map' in globals() else None
                if li is None or le is None:
                    continue
                if not (0 <= le < n_rows and 0 <= li < n_cols):
                    continue
                count[le, li] += 1
                rv = 0
                if runs_col:
                    try:
                        rv = int(row.get(runs_col, 0) or 0)
                    except:
                        rv = 0
                runs[le, li] += rv
                if rv >= 4:
                    bounds[le, li] += 1
                if rv == 0:
                    dots[le, li] += 1
                dval = str(row.get(dismissal_col, '') or '').lower()
                if any(tok in dval for tok in wkt_tokens):
                    wkt[le, li] += 1
                cval = row.get(control_col, None)
                if cval is not None:
                    if isinstance(cval, str) and 'not' in cval.lower():
                        ctrl_not[le, li] += 1
                    elif isinstance(cval, (int, float)) and float(cval) == 0:
                        ctrl_not[le, li] += 1
        
            # compute Econ (runs/6 balls) and control %
            econ = np.full(count.shape, np.nan)
            ctrl_pct = np.full(count.shape, np.nan)
            for i in range(n_rows):
                for j in range(n_cols):
                    if count[i, j] > 0:
                        econ[i, j] = (runs[i, j] * 6.0 / count[i, j])
                        ctrl_pct[i, j] = (ctrl_not[i, j] / count[i, j]) * 100.0
            
            return {
                'count': count, 'bounds': bounds, 'dots': dots,
                'runs': runs, 'econ': econ, 'ctrl_pct': ctrl_pct, 'wkt': wkt, 'n_rows': n_rows, 'n_cols': n_cols
            }
        
        # ---------- display utility (adapted for bowling metrics) ----------
        def display_pitchmaps_from_df(df_src, title_prefix):
            if df_src is None or df_src.empty:
                st.info(f"No deliveries to show for {title_prefix}")
                return
        
            grids = build_pitch_grids(df_src)
        
            # detect LHB presence among deliveries
            bh_col_name = globals().get('bat_hand_col', 'bat_hand')
            is_lhb = False
            if bh_col_name in df_src.columns:
                hands = df_src[bh_col_name].dropna().astype(str).str.strip().unique()
                if any(h.upper().startswith('L') for h in hands):
                    is_lhb = True
        
            def maybe_flip(arr):
                return np.fliplr(arr) if is_lhb else arr.copy()
        
            count = maybe_flip(grids['count'])
            bounds = maybe_flip(grids['bounds'])
            dots = maybe_flip(grids['dots'])
            econ = maybe_flip(grids['econ'])
            ctrl = maybe_flip(grids['ctrl_pct'])
            wkt = maybe_flip(grids['wkt'])
            runs = maybe_flip(grids['runs'])
        
            total = count.sum() if count.sum() > 0 else 1.0
            perc = count.astype(float) / total * 100.0
        
            # xticks order: for RHB left->right, for LHB reversed (we flipped arrays already,
            # so choose labels accordingly)
            xticks_base = ['Wide Out Off', 'Outside Off', 'On Stumps', 'Down Leg', 'Wide Down Leg']
            xticks = xticks_base[::-1] if is_lhb else xticks_base
        
            # ytick labels depends on n_rows and whether FULL_TOSS exists
            n_rows = grids['n_rows']
            # prefer to show Full Toss on top if n_rows == 6
            if n_rows >= 6:
                yticklabels = ['Short', 'Back of Length', 'Good', 'Full', 'Yorker', 'Full Toss'][:n_rows]
            else:
                yticklabels = ['Short', 'Back of Length', 'Good', 'Full', 'Yorker'][:n_rows]
        
            fig, axes = plt.subplots(3, 2, figsize=(14, 18))
            plt.suptitle(f"{player_selected} — {title_prefix}", fontsize=16, weight='bold')
        
            plot_list = [
                (perc, '% of balls (heat)', 'Blues'),
                (bounds, 'Boundaries conceded (count)', 'OrRd'),
                (dots, 'Dot balls (count)', 'Blues'),
                (econ, 'Econ (runs/6 balls)', 'Reds'),
                (ctrl, 'False Shot % (induced)', 'PuBu'),
                (runs, 'Runs conceded (sum)', 'Reds')
            ]
        
            for ax_idx, (ax, (arr, ttl, cmap)) in enumerate(zip(axes.flat, plot_list)):
                safe_arr = np.nan_to_num(arr.astype(float), nan=0.0)
                # autoscale vmax by 95th percentile to reduce outlier effect
                flat = safe_arr.flatten()
                if np.all(flat == 0):
                    vmin, vmax = 0, 1
                else:
                    vmin = float(np.nanmin(flat))
                    vmax = float(np.nanpercentile(flat, 95))
                    if vmax <= vmin:
                        vmax = vmin + 1.0
        
                im = ax.imshow(safe_arr, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(ttl)
                ax.set_xticks(range(grids['n_cols'])); ax.set_yticks(range(grids['n_rows']))
                ax.set_xticklabels(xticks, rotation=45, ha='right')
                ax.set_yticklabels(yticklabels)
        
                # black minor-grid borders for cells
                ax.set_xticks(np.arange(-0.5, grids['n_cols'], 1), minor=True)
                ax.set_yticks(np.arange(-0.5, grids['n_rows'], 1), minor=True)
                ax.grid(which='minor', color='black', linewidth=0.6, alpha=0.95)
                ax.tick_params(which='minor', bottom=False, left=False)
        
                # annotate N W for wicket cells (e.g., '2 W') ONLY in the first plot (% of balls)
                if ax_idx == 0:
                    for i in range(grids['n_rows']):
                        for j in range(grids['n_cols']):
                            w_count = int(wkt[i, j])
                            if w_count > 0:
                                w_text = f"{w_count} W" if w_count > 1 else 'W'
                                ax.text(j, i, w_text, ha='center', va='center', fontsize=14, color='gold', weight='bold',
                                        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
        
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
            # display via safe_st_pyplot if available
            safe_fn = globals().get('safe_st_pyplot', None)
            try:
                if callable(safe_fn):
                    safe_fn(fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
                else:
                    st.pyplot(fig)
            except Exception:
                st.pyplot(fig)
            finally:
                plt.close(fig)
        
        # ---------- When user selects a batter hand ----------
        if chosen_hand:
            def filter_by_hand(df, col='bat_hand', hand=chosen_hand):
                if col not in df.columns:
                    return df.iloc[0:0]
                mask = df[col].astype(str).str.lower().str.contains(str(hand).lower(), na=False)
                if not mask.any():
                    norm_hand = _norm_key(hand)
                    mask = df[col].apply(lambda x: _norm_key(x) == norm_hand)
                return df[mask].copy()
        
            sel_pf = filter_by_hand(pf)
            sel_bdf = filter_by_hand(bdf)
        
            df_use = sel_pf if not sel_pf.empty else sel_bdf
            if df_use.empty:
                st.info(f"No deliveries found for batter hand '{chosen_hand}'.")
            else:
                st.markdown(f"### Detailed view — Batter Hand: {chosen_hand}")
        
                draw_bowler_wagon_if_available(df_use, player_selected)
        
                st.markdown(f"#### {player_selected}'s Caught Dismissals")
                draw_bowler_caught_dismissals_wagon(df_use, player_selected)
        
                display_pitchmaps_from_df(df_use, f"vs Batter Hand: {chosen_hand}")

        
#     else:
#         st.markdown(f"<div style='font-size:20px; font-weight:800; color:#111;'> Bowling — {player_selected}</div>", unsafe_allow_html=True)
#         # ============================================================================
#         # BOWLER ANALYSIS - WAGON WHEEL FUNCTIONS (MATCHING BATTER ANALYSIS STYLE)
#         # ============================================================================
        
#         def draw_bowler_wagon_if_available(df_wagon, bowler_name, normalize_to_rhb=True):
#             """
#             Wrapper that calls draw_cricket_field_with_run_totals_requested for BOWLER analysis.
#             Shows runs conceded by the bowler in different zones.
#             - normalize_to_rhb: True => request RHB-normalised output (legacy behaviour).
#                                 False => request true handedness visualization (LHB will appear mirrored).
#             """
#             import matplotlib.pyplot as plt
#             import streamlit as st
#             import inspect
#             from matplotlib.figure import Figure as MplFigure
        
#             # Defensive check
#             if not isinstance(df_wagon, pd.DataFrame) or df_wagon.empty:
#                 st.warning("No wagon data available to draw.")
#                 return
        
#             # Decide handedness (for UI messages / debugging)
#             batting_style_val = None
#             if 'bat_hand' in df_wagon.columns:
#                 try:
#                     batting_style_val = df_wagon['bat_hand'].dropna().iloc[0]
#                 except Exception:
#                     batting_style_val = None
#             is_lhb = isinstance(batting_style_val, str) and batting_style_val.strip().upper().startswith('L')
        
#             # Check function signature
#             draw_fn = globals().get('draw_cricket_field_with_run_totals_requested', None)
#             if draw_fn is None or not callable(draw_fn):
#                 st.warning("Wagon chart function not found; please ensure `draw_cricket_field_with_run_totals_requested` is defined earlier.")
#                 return
        
#             try:
#                 sig = inspect.signature(draw_fn)
#                 if 'normalize_to_rhb' in sig.parameters:
#                     # call with the explicit flag (preferred)
#                     fig = draw_fn(df_wagon, bowler_name, normalize_to_rhb=normalize_to_rhb)
#                 else:
#                     # older signature: call without flag (maintain legacy behaviour)
#                     fig = draw_fn(df_wagon, bowler_name)
        
#                 # If the function returned a Matplotlib fig — display it
#                 if isinstance(fig, MplFigure):
#                     safe_fn = globals().get('safe_st_pyplot', None)
#                     if callable(safe_fn):
#                         try:
#                             safe_fn(fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
#                         except Exception:
#                             st.pyplot(fig)
#                     else:
#                         st.pyplot(fig)
#                     return
        
#                 # If function returned None, it may have drawn to current fig; capture that
#                 if fig is None:
#                     mpl_fig = plt.gcf()
#                     # If figure has axes and content, display it
#                     if isinstance(mpl_fig, MplFigure) and len(mpl_fig.axes) > 0:
#                         safe_fn = globals().get('safe_st_pyplot', None)
#                         if callable(safe_fn):
#                             try:
#                                 safe_fn(mpl_fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
#                             except Exception:
#                                 st.pyplot(mpl_fig)
#                         else:
#                             st.pyplot(mpl_fig)
#                         return
        
#                 # If function returned a Plotly figure (rare), display it
#                 if isinstance(fig, go.Figure):
#                     try:
#                         fig.update_yaxes(scaleanchor="x", scaleratio=1)
#                     except Exception:
#                         pass
#                     st.plotly_chart(fig, use_container_width=True)
#                     return
        
#                 # Unknown return — just state it
#                 st.warning("Wagon draw function executed but returned an unexpected type; nothing displayed.")
#             except Exception as e:
#                 st.error(f"Wagon drawing function raised: {e}")
        
        
#         def draw_bowler_caught_dismissals_wagon(df_wagon, bowler_name, normalize_to_rhb=True):
#             """
#             Plotly wagon chart for caught dismissals FOR BOWLER.
#             EXACTLY matches the batter analysis style and behavior.
#             - normalize_to_rhb=True: show points in RHB frame (no flipping)
#             - normalize_to_rhb=False: show true handedness; if batter is LHB, flip x coordinates to mirror
#             """
#             import plotly.graph_objects as go
#             import numpy as np
#             import pandas as pd
#             import streamlit as st
            
#             # Defensive checks
#             if not isinstance(df_wagon, pd.DataFrame):
#                 st.warning("draw_bowler_caught_dismissals_wagon expects a DataFrame")
#                 return
        
#             if 'dismissal' not in df_wagon.columns:
#                 st.warning("No 'dismissal' column present — cannot filter caught dismissals.")
#                 return
            
#             # Filter for THIS BOWLER's caught dismissals
#             if 'bowl' not in df_wagon.columns:
#                 st.warning("No 'bowl' column found - cannot filter by bowler.")
#                 return
        
#             caught_df = df_wagon[
#                 (df_wagon['bowl'] == bowler_name) &
#                 (df_wagon['dismissal'].astype(str).str.lower().str.contains('caught', na=False))
#             ].copy()
            
#             if caught_df.empty:
#                 st.info(f"No caught dismissals for {bowler_name}.")
#                 return
        
#             if 'wagonX' not in caught_df.columns or 'wagonY' not in caught_df.columns:
#                 st.warning("Missing 'wagonX' or 'wagonY' columns.")
#                 return
        
#             # Numeric conversion and drop invalid
#             caught_df['wagonX'] = pd.to_numeric(caught_df['wagonX'], errors='coerce')
#             caught_df['wagonY'] = pd.to_numeric(caught_df['wagonY'], errors='coerce')
#             caught_df = caught_df.dropna(subset=['wagonX', 'wagonY']).copy()
#             if caught_df.empty:
#                 st.info("No valid wagon coordinates for caught dismissals.")
#                 return
        
#             # Normalize coords to [-1,1] (match your scaling)
#             center_x = 184.0
#             center_y = 184.0
#             radius = 184.0
#             caught_df['x_plot'] = (caught_df['wagonX'].astype(float) - center_x) / radius
#             caught_df['y_plot'] = - (caught_df['wagonY'].astype(float) - center_y) / radius  # flip Y so positive is up
        
#             # Detect LHB if any
#             batting_style_val = None
#             if 'bat_hand' in df_wagon.columns and not df_wagon.empty:
#                 try:
#                     batting_style_val = df_wagon['bat_hand'].dropna().iloc[0]
#                 except Exception:
#                     batting_style_val = None
#             is_lhb = False
#             if batting_style_val is not None:
#                 try:
#                     is_lhb = str(batting_style_val).strip().upper().startswith('L')
#                 except Exception:
#                     is_lhb = False
        
#             # Apply display rule:
#             # - If normalize_to_rhb True => do NOT flip (everyone in RHB frame)
#             # - If normalize_to_rhb False and batter is LHB => flip x for true mirror
#             x_vals = caught_df['x_plot'].values
#             y_vals = caught_df['y_plot'].values
#             if (not normalize_to_rhb) and is_lhb:
#                 x_vals = -x_vals
        
#             # Filter to inside circle
#             caught_df['distance'] = np.sqrt(x_vals**2 + y_vals**2)
#             caught_df = caught_df[caught_df['distance'] <= 1].copy()
#             if caught_df.empty:
#                 st.info(f"No caught dismissals inside the field for {bowler_name}.")
#                 return
        
#             # Choose hover columns (BOWLER VERSION - show batter info)
#             hover_candidates = ['bat', 'bowl', 'bowl_style', 'bowl_kind', 'line', 'length', 'shot', 'dismissal']
#             customdata_cols = [c for c in hover_candidates if c in caught_df.columns]
#             # Build customdata rows
#             customdata = []
#             for _, row in caught_df.iterrows():
#                 customdata.append([("" if pd.isna(row.get(c, "")) else str(row.get(c, ""))) for c in customdata_cols])
        
#             # Build the figure
#             fig = go.Figure()
        
#             # Circles and pitch: if normalize_to_rhb True, use canonical RHB shapes; if False and LHB, flip x coords for shapes
#             def add_circle_shape(fig_obj, x0_raw, y0, x1_raw, y1, **kwargs):
#                 if (not normalize_to_rhb) and is_lhb:
#                     tx0 = -x1_raw
#                     tx1 = -x0_raw
#                 else:
#                     tx0 = x0_raw
#                     tx1 = x1_raw
#                 x0_, x1_ = min(tx0, tx1), max(tx0, tx1)
#                 fig_obj.add_shape(type="circle", xref="x", yref="y", x0=x0_, y0=y0, x1=x1_, y1=y1, **kwargs)
        
#             add_circle_shape(fig, -1, -1, 1, 1, fillcolor="#228B22", line_color="black", opacity=1, layer="below")
#             add_circle_shape(fig, -0.5, -0.5, 0.5, 0.5, fillcolor="#66bb6a", line_color="white", opacity=1, layer="below")
        
#             # Pitch rectangle (tan)
#             pitch_x0, pitch_x1 = (-0.04, 0.04)
#             if (not normalize_to_rhb) and is_lhb:
#                 pitch_x0, pitch_x1 = (-pitch_x1, -pitch_x0)
#             fig.add_shape(type="rect", x0=pitch_x0, y0=-0.08, x1=pitch_x1, y1=0.08,
#                           fillcolor="tan", line_color=None, opacity=1, layer="above")
        
#             # Radial lines (mirror endpoints if LHB & not normalised)
#             angles = np.linspace(0, 2*np.pi, 9)[:-1]
#             for angle in angles:
#                 x_end = np.cos(angle)
#                 y_end = np.sin(angle)
#                 if (not normalize_to_rhb) and is_lhb:
#                     x_end = -x_end
#                 fig.add_trace(go.Scatter(x=[0, x_end], y=[0, y_end],
#                                          mode='lines', line=dict(color='white', width=1), opacity=0.25, showlegend=False))
        
#             # Plot dismissal points (use x_vals,y_vals filtered earlier)
#             fig.add_trace(go.Scatter(
#                 x=x_vals,
#                 y=y_vals,
#                 mode='markers',
#                 marker=dict(color='red', size=12, line=dict(color='black', width=1.5)),
#                 customdata=customdata,
#                 hovertemplate=(
#                     "<br>".join([f"<b>{col}:</b> %{{customdata[{i}]}}" for i, col in enumerate(customdata_cols)]) +
#                     "<extra></extra>"
#                 ),
#                 name='Caught Dismissal Locations'
#             ))
        
#             # Layout & equal aspect scaling so mirror is obvious
#             axis_range = 1.2
#             fig.update_layout(
#                 xaxis=dict(range=[-axis_range, axis_range], showgrid=False, zeroline=False, visible=False),
#                 yaxis=dict(range=[-axis_range, axis_range], showgrid=False, zeroline=False, visible=False),
#                 showlegend=True,
#                 legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
#                 width=800,
#                 height=800,
#                 margin=dict(l=0, r=0, t=0, b=0)
#             )
#             try:
#                 fig.update_yaxes(scaleanchor="x", scaleratio=1)
#             except Exception:
#                 pass
        
#             # Finally display the plotly figure
#             st.plotly_chart(fig, use_container_width=True)
#         # ============================================================================
# # BOWLER ANALYSIS - INTERACTIVE CAUGHT DISMISSALS WAGON WHEEL (COMPLETE)
# # ============================================================================

#         def draw_bowler_caught_dismissals_wagon_interactive(df_wagon, bowler_name):
#             """
#             Interactive wagon wheel showing caught dismissals FOR A BOWLER with hover info.
#             Only shows catches within the boundary circle.
#             Completely independent - uses 'bowl' column instead of 'bat'.
#             """
#             import plotly.graph_objects as go
#             import numpy as np
#             import pandas as pd
#             import streamlit as st
            
#             # Filter caught dismissals FOR THIS BOWLER
#             if 'bowl' not in df_wagon.columns:
#                 st.warning("No 'bowl' column found - cannot filter by bowler.")
#                 return
            
#             # Filter for this specific bowler's deliveries that resulted in caught dismissals
#             caught_df = df_wagon[
#                 (df_wagon['bowl'] == bowler_name) &
#                 (df_wagon['dismissal'].astype(str).str.lower().str.contains('caught', na=False))
#             ].copy()
            
#             if caught_df.empty:
#                 st.info(f"No caught dismissals for bowler {bowler_name}.")
#                 return
            
#             # Check for required columns
#             if 'wagonX' not in caught_df.columns or 'wagonY' not in caught_df.columns:
#                 st.warning("Missing 'wagonX' or 'wagonY' columns.")
#                 return
            
#             # Scale coordinates to normalized -1 to 1
#             center_x = 184
#             center_y = 184
#             radius = 184
            
#             caught_df['wagonX'] = pd.to_numeric(caught_df['wagonX'], errors='coerce')
#             caught_df['wagonY'] = pd.to_numeric(caught_df['wagonY'], errors='coerce')
#             caught_df = caught_df.dropna(subset=['wagonX', 'wagonY'])
            
#             if caught_df.empty:
#                 st.info(f"No valid coordinates for caught dismissals.")
#                 return
            
#             caught_df['x_plot'] = (caught_df['wagonX'].astype(float) - center_x) / radius
#             caught_df['y_plot'] = -(caught_df['wagonY'].astype(float) - center_y) / radius
            
#             # Detect batter handedness (to flip or not)
#             batting_style_val = caught_df['bat_hand'].iloc[0] if 'bat_hand' in caught_df.columns and not caught_df.empty else 'R'
#             is_lhb = str(batting_style_val).upper().startswith('L')
#             if is_lhb:
#                 caught_df['x_plot'] = -caught_df['x_plot']
            
#             # FILTER: Only keep dots within the boundary circle (radius = 1)
#             caught_df['distance'] = np.sqrt(caught_df['x_plot']**2 + caught_df['y_plot']**2)
#             caught_df = caught_df[caught_df['distance'] <= 1.0].copy()
            
#             if caught_df.empty:
#                 st.info(f"No caught dismissals within the boundary for bowler {bowler_name}.")
#                 return
            
#             # Prepare hover text with available info (BOWLER-SPECIFIC COLUMNS)
#             hover_texts = []
#             for _, row in caught_df.iterrows():
#                 parts = []
                
#                 # Show BATTER name (who got out)
#                 if 'bat' in caught_df.columns:
#                     parts.append(f"<b>Batter:</b> {row['bat']}")
                
#                 # Show bowler (should be same as bowler_name)
#                 if 'bowl' in caught_df.columns:
#                     parts.append(f"<b>Bowler:</b> {row['bowl']}")
                
#                 if 'bowl_style' in caught_df.columns:
#                     parts.append(f"<b>Bowler Style:</b> {row['bowl_style']}")
                
#                 if 'bowl_kind' in caught_df.columns:
#                     parts.append(f"<b>Bowler Kind:</b> {row['bowl_kind']}")
                
#                 if 'line' in caught_df.columns:
#                     parts.append(f"<b>Line:</b> {row['line']}")
                
#                 if 'length' in caught_df.columns:
#                     parts.append(f"<b>Length:</b> {row['length']}")
                
#                 if 'shot' in caught_df.columns:
#                     parts.append(f"<b>Shot:</b> {row['shot']}")
                
#                 if 'score' in caught_df.columns or 'runs' in caught_df.columns:
#                     runs = row.get('score', row.get('runs', '-'))
#                     parts.append(f"<b>Runs:</b> {runs}")
                
#                 # Add dismissal info
#                 if 'dismissal' in caught_df.columns:
#                     parts.append(f"<b>Dismissal:</b> {row['dismissal']}")
                
#                 hover_texts.append("<br>".join(parts))
            
#             # Create Plotly figure
#             fig = go.Figure()
            
#             # Outer boundary circle (dark green)
#             theta_boundary = np.linspace(0, 2*np.pi, 100)
#             fig.add_trace(go.Scatter(
#                 x=np.cos(theta_boundary),
#                 y=np.sin(theta_boundary),
#                 mode='lines',
#                 line=dict(color='black', width=3),
#                 fill='toself',
#                 fillcolor='rgba(34, 139, 34, 0.3)',  # #228B22 with transparency
#                 hoverinfo='skip',
#                 showlegend=False
#             ))
            
#             # Inner circle (lighter green)
#             theta_inner = np.linspace(0, 2*np.pi, 100)
#             fig.add_trace(go.Scatter(
#                 x=0.5 * np.cos(theta_inner),
#                 y=0.5 * np.sin(theta_inner),
#                 mode='lines',
#                 line=dict(color='white', width=1),
#                 fill='toself',
#                 fillcolor='rgba(102, 187, 106, 0.4)',  # #66bb6a
#                 hoverinfo='skip',
#                 showlegend=False
#             ))
            
#             # Pitch rectangle
#             pitch_x = [-0.04, 0.04, 0.04, -0.04, -0.04]
#             pitch_y = [-0.08, -0.08, 0.08, 0.08, -0.08]
#             fig.add_trace(go.Scatter(
#                 x=pitch_x,
#                 y=pitch_y,
#                 mode='lines',
#                 fill='toself',
#                 fillcolor='tan',
#                 line=dict(color='tan', width=1),
#                 hoverinfo='skip',
#                 showlegend=False
#             ))
            
#             # Radial lines
#             angles = np.linspace(0, 2*np.pi, 9)[:-1]
#             for angle in angles:
#                 fig.add_trace(go.Scatter(
#                     x=[0, np.cos(angle)],
#                     y=[0, np.sin(angle)],
#                     mode='lines',
#                     line=dict(color='rgba(255, 255, 255, 0.25)', width=1),
#                     hoverinfo='skip',
#                     showlegend=False
#                 ))
            
#             # Plot caught dismissal dots with hover info
#             fig.add_trace(go.Scatter(
#                 x=caught_df['x_plot'],
#                 y=caught_df['y_plot'],
#                 mode='markers',
#                 marker=dict(
#                     size=12,
#                     color='red',
#                     opacity=0.9,
#                     line=dict(color='black', width=1.5)
#                 ),
#                 text=hover_texts,
#                 hovertemplate='%{text}<extra></extra>',
#                 name='Caught Dismissals',
#                 showlegend=True
#             ))
            
#             # Layout
#             fig.update_layout(
#                 title=dict(
#                     text=f"{bowler_name} - Caught Dismissals (Hover for details)",
#                     font=dict(size=16, color='black')
#                 ),
#                 width=700,
#                 height=700,
#                 xaxis=dict(
#                     range=[-1.2, 1.2],
#                     showgrid=False,
#                     zeroline=False,
#                     showticklabels=False,
#                     scaleanchor='y',
#                     scaleratio=1
#                 ),
#                 yaxis=dict(
#                     range=[-1.2, 1.2],
#                     showgrid=False,
#                     zeroline=False,
#                     showticklabels=False
#                 ),
#                 plot_bgcolor='white',
#                 hovermode='closest',
#                 showlegend=True,
#                 legend=dict(
#                     x=0.02,
#                     y=0.98,
#                     bgcolor='rgba(255, 255, 255, 0.8)',
#                     bordercolor='black',
#                     borderwidth=1
#                 )
#             )
            
#             st.plotly_chart(fig, use_container_width=True)
            
#             # Optional: Show summary table
#             st.markdown(f"**Total caught dismissals within boundary:** {len(caught_df)}")
            
#             # Show detailed table with dismissal info
#             if not caught_df.empty:
#                 display_cols = []
#                 for col in ['bat', 'bowl', 'bowl_style', 'bowl_kind', 'line', 'length', 'shot', 'score', 'dismissal']:
#                     if col in caught_df.columns:
#                         display_cols.append(col)
                
#                 if display_cols:
#                     with st.expander("📊 View Dismissal Details Table"):
#                         st.dataframe(
#                             caught_df[display_cols].reset_index(drop=True),
#                             use_container_width=True
#                         )



        
#         def draw_bowler_wagon_if_available(df_wagon, bowler_name, normalize_to_rhb=True):
#             """
#             Wrapper for drawing bowler wagon wheel (runs conceded by zone).
#             Uses the same cricket field function but filtered for bowler deliveries.
            
#             Args:
#                 df_wagon: DataFrame with bowler's deliveries
#                 bowler_name: Name of the bowler
#                 normalize_to_rhb: If True, show in RHB frame; if False, mirror for LHB batters
#             """
#             import matplotlib.pyplot as plt
#             import streamlit as st
#             import inspect
            
#             # Defensive check
#             if not isinstance(df_wagon, pd.DataFrame) or df_wagon.empty:
#                 st.warning("No wagon data available to draw.")
#                 return
            
#             # Detect batter handedness (since we're showing where bowler conceded runs)
#             batting_style_val = None
#             if 'bat_hand' in df_wagon.columns:
#                 try:
#                     # For bowler analysis, we care about the batter's hand
#                     batting_style_val = df_wagon['bat_hand'].dropna().iloc[0]
#                 except Exception:
#                     batting_style_val = None
#             is_lhb = isinstance(batting_style_val, str) and batting_style_val.strip().upper().startswith('L')
            
#             # Check function signature
#             draw_fn = globals().get('draw_cricket_field_with_run_totals_requested', None)
#             if draw_fn is None or not callable(draw_fn):
#                 st.warning("Wagon chart function not found; please ensure `draw_cricket_field_with_run_totals_requested` is defined earlier.")
#                 return
            
#             try:
#                 sig = inspect.signature(draw_fn)
#                 if 'normalize_to_rhb' in sig.parameters:
#                     # Call with the explicit flag (preferred)
#                     # Note: For bowler, we pass bowler_name but the function will use bat/batter columns for wagon zones
#                     fig = draw_fn(df_wagon, bowler_name, normalize_to_rhb=normalize_to_rhb)
#                 else:
#                     # Older signature: call without flag
#                     fig = draw_fn(df_wagon, bowler_name)
                
#                 # Display the figure
#                 if isinstance(fig, plt.Figure):
#                     safe_fn = globals().get('safe_st_pyplot', None)
#                     if callable(safe_fn):
#                         try:
#                             safe_fn(fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
#                         except Exception:
#                             st.pyplot(fig)
#                     else:
#                         st.pyplot(fig)
#                     return
                
#                 # If function returned None, capture current figure
#                 if fig is None:
#                     mpl_fig = plt.gcf()
#                     if isinstance(mpl_fig, plt.Figure) and len(mpl_fig.axes) > 0:
#                         safe_fn = globals().get('safe_st_pyplot', None)
#                         if callable(safe_fn):
#                             try:
#                                 safe_fn(mpl_fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
#                             except Exception:
#                                 st.pyplot(mpl_fig)
#                         else:
#                             st.pyplot(mpl_fig)
#                         return
                
#                 # If Plotly figure
#                 if isinstance(fig, go.Figure):
#                     try:
#                         fig.update_yaxes(scaleanchor="x", scaleratio=1)
#                     except Exception:
#                         pass
#                     st.plotly_chart(fig, use_container_width=True)
#                     return
                
#                 st.warning("Wagon draw function executed but returned an unexpected type; nothing displayed.")
#             except Exception as e:
#                 st.error(f"Wagon drawing function raised: {e}")
        
        
#         def draw_bowler_caught_dismissals_wagon(df_wagon, bowler_name, normalize_to_rhb=True):
#             """
#             Plotly wagon chart for caught dismissals taken by a bowler.
#             Shows where the bowler gets caught dismissals on the field.
            
#             Args:
#                 df_wagon: DataFrame with bowler's deliveries
#                 bowler_name: Name of the bowler
#                 normalize_to_rhb: If True, show in RHB frame; if False, mirror for LHB batters
#             """
#             # Defensive checks
#             if not isinstance(df_wagon, pd.DataFrame):
#                 st.warning("draw_bowler_caught_dismissals_wagon expects a DataFrame")
#                 return
            
#             if 'dismissal' not in df_wagon.columns:
#                 st.warning("No 'dismissal' column present — cannot filter caught dismissals.")
#                 return
            
#             # Filter for caught dismissals by this bowler
#             caught_df = df_wagon[
#                 df_wagon['dismissal'].astype(str).str.lower().str.contains('caught', na=False)
#             ].copy()
            
#             if caught_df.empty:
#                 st.info(f"No caught dismissals for {bowler_name}.")
#                 return
            
#             if 'wagonX' not in caught_df.columns or 'wagonY' not in caught_df.columns:
#                 st.warning("Missing 'wagonX' or 'wagonY' columns.")
#                 return
            
#             # Numeric conversion and drop invalid
#             caught_df['wagonX'] = pd.to_numeric(caught_df['wagonX'], errors='coerce')
#             caught_df['wagonY'] = pd.to_numeric(caught_df['wagonY'], errors='coerce')
#             caught_df = caught_df.dropna(subset=['wagonX', 'wagonY']).copy()
            
#             if caught_df.empty:
#                 st.info("No valid wagon coordinates for caught dismissals.")
#                 return
            
#             # Normalize coords to [-1,1]
#             center_x = 184.0
#             center_y = 184.0
#             radius = 184.0
#             caught_df['x_plot'] = (caught_df['wagonX'].astype(float) - center_x) / radius
#             caught_df['y_plot'] = -(caught_df['wagonY'].astype(float) - center_y) / radius  # flip Y
            
#             # Detect batter handedness (LHB vs RHB)
#             batting_style_val = None
#             if 'bat_hand' in df_wagon.columns and not df_wagon.empty:
#                 try:
#                     batting_style_val = df_wagon['bat_hand'].dropna().iloc[0]
#                 except Exception:
#                     batting_style_val = None
            
#             is_lhb = False
#             if batting_style_val is not None:
#                 try:
#                     is_lhb = str(batting_style_val).strip().upper().startswith('L')
#                 except Exception:
#                     is_lhb = False
            
#             # Apply display rule for handedness
#             x_vals = caught_df['x_plot'].values
#             y_vals = caught_df['y_plot'].values
#             if (not normalize_to_rhb) and is_lhb:
#                 x_vals = -x_vals
            
#             # Filter to inside circle
#             caught_df['distance'] = np.sqrt(x_vals**2 + y_vals**2)
#             caught_df = caught_df[caught_df['distance'] <= 1].copy()
            
#             if caught_df.empty:
#                 st.info(f"No caught dismissals inside the field for {bowler_name}.")
#                 return
            
#             # Prepare hover data - for bowler, show batter info
#             hover_candidates = ['bat', 'bowl_kind', 'line', 'length', 'shot', 'dismissal']
#             customdata_cols = [c for c in hover_candidates if c in caught_df.columns]
            
#             customdata = []
#             for _, row in caught_df.iterrows():
#                 customdata.append([
#                     ("" if pd.isna(row.get(c, "")) else str(row.get(c, ""))) 
#                     for c in customdata_cols
#                 ])
            
#             # Build the figure
#             fig = go.Figure()
            
#             # Add field shapes
#             def add_circle_shape(fig_obj, x0_raw, y0, x1_raw, y1, **kwargs):
#                 if (not normalize_to_rhb) and is_lhb:
#                     tx0 = -x1_raw
#                     tx1 = -x0_raw
#                 else:
#                     tx0 = x0_raw
#                     tx1 = x1_raw
#                 x0_, x1_ = min(tx0, tx1), max(tx0, tx1)
#                 fig_obj.add_shape(type="circle", xref="x", yref="y", x0=x0_, y0=y0, x1=x1_, y1=y1, **kwargs)
            
#             # Outer boundary
#             add_circle_shape(fig, -1, -1, 1, 1, fillcolor="#228B22", line_color="black", opacity=1, layer="below")
#             # Inner circle
#             add_circle_shape(fig, -0.5, -0.5, 0.5, 0.5, fillcolor="#66bb6a", line_color="white", opacity=1, layer="below")
            
#             # Pitch rectangle
#             pitch_x0, pitch_x1 = (-0.04, 0.04)
#             if (not normalize_to_rhb) and is_lhb:
#                 pitch_x0, pitch_x1 = (-pitch_x1, -pitch_x0)
#             fig.add_shape(type="rect", x0=pitch_x0, y0=-0.08, x1=pitch_x1, y1=0.08,
#                           fillcolor="tan", line_color=None, opacity=1, layer="above")
            
#             # Radial lines
#             angles = np.linspace(0, 2*np.pi, 9)[:-1]
#             for angle in angles:
#                 x_end = np.cos(angle)
#                 y_end = np.sin(angle)
#                 if (not normalize_to_rhb) and is_lhb:
#                     x_end = -x_end
#                 fig.add_trace(go.Scatter(
#                     x=[0, x_end], y=[0, y_end],
#                     mode='lines', line=dict(color='white', width=1), 
#                     opacity=0.25, showlegend=False
#                 ))
            
#             # Plot dismissal points
#             hovertemplate_parts = [f"<b>{col}:</b> %{{customdata[{i}]}}" for i, col in enumerate(customdata_cols)]
            
#             fig.add_trace(go.Scatter(
#                 x=x_vals,
#                 y=y_vals,
#                 mode='markers',
#                 marker=dict(color='red', size=12, line=dict(color='black', width=1.5)),
#                 customdata=customdata,
#                 hovertemplate="<br>".join(hovertemplate_parts) + "<extra></extra>",
#                 name='Caught Dismissal Locations'
#             ))
            
#             # Layout
#             axis_range = 1.2
#             fig.update_layout(
#                 title=f"{bowler_name} - Caught Dismissals (Hover for details)",
#                 xaxis=dict(range=[-axis_range, axis_range], showgrid=False, zeroline=False, visible=False),
#                 yaxis=dict(range=[-axis_range, axis_range], showgrid=False, zeroline=False, visible=False),
#                 showlegend=True,
#                 legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
#                 width=800,
#                 height=800,
#                 margin=dict(l=0, r=0, t=40, b=0)
#             )
            
#             try:
#                 fig.update_yaxes(scaleanchor="x", scaleratio=1)
#             except Exception:
#                 pass
            
#             # Display
#             st.plotly_chart(fig, use_container_width=True)

#         # ----------------------------
# # Bowler-specific helpers & pipeline
#         # ----------------------------
#         import plotly.graph_objects as go
        
#         def _pick_col(df, candidates):
#             """Return first candidate column name present in df, else None."""
#             for c in candidates:
#                 if c in df.columns:
#                     return c
#             return None
        
#         def get_bowler_deliveries(df_all, bowler_name):
#             """
#             Return dataframe rows for deliveries by the given bowler_name.
#             Accepts multiple possible column names for the bowler column.
#             """
#             if df_all is None or getattr(df_all, "empty", True):
#                 return pd.DataFrame()
#             bowl_col = _pick_col(df_all, [COL_BOWL, 'bowl', 'bowler'])
#             if bowl_col is None:
#                 return pd.DataFrame()
#             mask = df_all[bowl_col].astype(str) == str(bowler_name)
#             return df_all.loc[mask].copy()
        
#         def filter_by_batter_hand(df_local, hand):
#             """Filter df_local by batter hand (case-insensitive substring). Returns copy."""
#             if df_local is None or df_local.empty:
#                 return df_local
#             if 'bat_hand' not in df_local.columns:
#                 return df_local.iloc[0:0].copy()
#             hand_lo = str(hand).lower()
#             mask = df_local['bat_hand'].astype(str).str.lower().str.contains(hand_lo, na=False)
#             if not mask.any():
#                 # fallback: exact normalized match
#                 norm = _norm_key(hand)
#                 mask = df_local['bat_hand'].apply(lambda x: _norm_key(x) == norm)
#             return df_local.loc[mask].copy()
        
#         def _local_draw_caught_dismissals_plotly(caught_df, title_text):
#             """Local plotly fallback for caught dismissals (expects columns for coords)."""
#             if caught_df is None or caught_df.empty:
#                 st.info("No caught dismissals to plot (local fallback).")
#                 return
        
#             # find coordinate candidates
#             wx = _pick_col(caught_df, ['wagonX','wagon_x','wagon_x_coord','x_coord','x'])
#             wy = _pick_col(caught_df, ['wagonY','wagon_y','wagon_y_coord','y_coord','y'])
#             if wx is None or wy is None:
#                 st.warning("No wagon coordinate columns present; cannot draw caught dismissal locations.")
#                 return
        
#             center_x, center_y, radius = 184.0, 184.0, 184.0
#             caught_df = caught_df.copy()
#             caught_df['x_plot'] = (pd.to_numeric(caught_df[wx], errors='coerce') - center_x) / radius
#             caught_df['y_plot'] = - (pd.to_numeric(caught_df[wy], errors='coerce') - center_y) / radius
#             caught_df = caught_df.dropna(subset=['x_plot','y_plot'])
#             if caught_df.empty:
#                 st.info("No valid coordinates for caught dismissals (local fallback).")
#                 return
        
#             fig = go.Figure()
#             fig.add_shape(type="circle", x0=-1, y0=-1, x1=1, y1=1, fillcolor="#228B22", line_color="black")
#             fig.add_shape(type="circle", x0=-0.5, y0=-0.5, x1=0.5, y1=0.5, fillcolor="#66bb6a", line_color="white")
#             fig.add_shape(type="rect", x0=-0.04, y0=-0.08, x1=0.04, y1=0.08, fillcolor="tan", line_color=None)
        
#             for a in np.linspace(0, 2*np.pi, 9)[:-1]:
#                 fig.add_trace(go.Scatter(x=[0, np.cos(a)], y=[0, np.sin(a)], mode='lines',
#                                          line=dict(color='white', width=1), opacity=0.25, showlegend=False))
        
#             hover_cols = []
#             for c in ('bat','batter','batsman'):
#                 if c in caught_df.columns:
#                     hover_cols.append(c); break
#             if 'dismissal' in caught_df.columns:
#                 hover_cols.append('dismissal')
#             if 'fielder' in caught_df.columns:
#                 hover_cols.append('fielder')
        
#             customdata = []
#             for _, r in caught_df.iterrows():
#                 rowvals = []
#                 for col in hover_cols:
#                     rowvals.append("" if pd.isna(r.get(col,'')) else str(r.get(col,'')))
#                 customdata.append(rowvals)
        
#             fig.add_trace(go.Scatter(
#                 x=caught_df['x_plot'],
#                 y=caught_df['y_plot'],
#                 mode='markers',
#                 marker=dict(color='red', size=10, line=dict(color='black', width=1.5)),
#                 customdata=customdata,
#                 hovertemplate=("".join([f"<b>{col}:</b> %{{customdata[{i}]}}<br>" for i,col in enumerate(hover_cols)]) + "<extra></extra>"),
#                 name='Caught dismissals'
#             ))
        
#             fig.update_layout(xaxis=dict(range=[-1.2,1.2], visible=False),
#                               yaxis=dict(range=[-1.2,1.2], visible=False, scaleanchor="x"),
#                               title=title_text, width=700, height=700, margin=dict(l=0,r=0,t=40,b=0))
#             st.plotly_chart(fig, use_container_width=True)
        
        
#         def draw_bowler_caughts_wrapper(df_bowler, bowler_name, normalize_to_rhb=True):
#             """
#             Try to call existing draw_caught_dismissals_wagon (preferred).
#             If it doesn't exist, or fails, use local fallback.
#             """
#             if df_bowler is None or df_bowler.empty:
#                 st.info("No deliveries available for caught-dismissal view.")
#                 return
        
#             # prefer the user's function if present
#             user_fn = globals().get('draw_caught_dismissals_wagon', None)
#             if callable(user_fn):
#                 try:
#                     # try calling with normalize flag if supported
#                     import inspect
#                     sig = inspect.signature(user_fn)
#                     if 'normalize_to_rhb' in sig.parameters:
#                         user_fn(df_bowler, bowler_name, normalize_to_rhb=normalize_to_rhb)
#                     else:
#                         user_fn(df_bowler, bowler_name)
#                     return
#                 except Exception as e:
#                     # swallow and fallback to local drawing
#                     st.warning(f"User draw_caught_dismissals_wagon raised: {e} — using local fallback.")
#             # local fallback: need only caught rows
#             caught_df = df_bowler[df_bowler.get(COL_DISMISSAL, df_bowler.get('dismissal', '')).astype(str).str.lower().str.contains('caught', na=False)].copy()
#             _local_draw_caught_dismissals_plotly(caught_df, title_text=f"Caught dismissals — {bowler_name}")
        
        
#         def _call_wagon_wrapper(df_bowler, bowler_name, normalize_to_rhb=True):
#             """Call user wagon function robustly (supports older signature)."""
#             user_wagon = globals().get('draw_wagon_if_available', None)
#             if callable(user_wagon):
#                 try:
#                     import inspect
#                     sig = inspect.signature(user_wagon)
#                     if 'normalize_to_rhb' in sig.parameters:
#                         user_wagon(df_bowler, bowler_name, normalize_to_rhb=normalize_to_rhb)
#                     else:
#                         # older: no normalize param
#                         user_wagon(df_bowler, bowler_name)
#                     return
#                 except Exception as e:
#                     st.warning(f"User draw_wagon_if_available raised: {e} — wagon not drawn.")
#                     return
#             # no wagon function found
#             st.warning("Wagon drawing function not found (draw_wagon_if_available).")
        
#         def show_bowler_detail_from_all(df_all, player_selected, chosen_hand=None, normalize_to_rhb=True, title_suffix=None):
#             """
#             Main pipeline for Bowler detail view (use this where you previously did wagon->caught->pitchmap).
#             - df_all: full dataframe (bdf/pf context)
#             - player_selected: bowler name
#             - chosen_hand: optional 'LHB'/'RHB' filter for batter hand
#             - normalize_to_rhb: whether to keep everyone in RHB frame (True) or show true LHB mirror (False)
#             """
#             if df_all is None or getattr(df_all, "empty", True):
#                 st.info("No data available.")
#                 return
        
#             # 1) get bowler deliveries
#             bf = get_bowler_deliveries(df_all, player_selected)
#             if bf is None or bf.empty:
#                 # fallback: try global bdf if available
#                 if 'bdf' in globals():
#                     bf = get_bowler_deliveries(globals()['bdf'], player_selected)
#             if bf is None or bf.empty:
#                 st.info("No bowling rows for this bowler.")
#                 return
        
#             # 2) optionally filter by batter hand
#             if chosen_hand:
#                 df_filtered = filter_by_batter_hand(bf, chosen_hand)
#                 if df_filtered is None or df_filtered.empty:
#                     st.info(f"No deliveries found for batter hand '{chosen_hand}'.")
#                     return
#             else:
#                 df_filtered = bf
        
#             # 3) draw wagon (robust wrapper)
#             st.markdown(f"### {player_selected} — Wagon wheel")
#             _call_wagon_wrapper(df_filtered, player_selected, normalize_to_rhb=normalize_to_rhb)
        
#             # 4) draw caught dismissals
#             st.markdown(f"### {player_selected}'s Caught Dismissals")
#             draw_bowler_caughts_wrapper(df_filtered, player_selected, normalize_to_rhb=normalize_to_rhb)
        
#             # 5) pitchmaps (use your existing function)
#             suffix = title_suffix if title_suffix is not None else f"vs {player_selected}"
#             display_pitchmaps_from_df(df_filtered, suffix)
        
        
#         # ----------------------------
#         # Example execution lines to use in your chosen_hand / chosen_kind / chosen_style blocks:
#         # Use the same df selection you already compute (or pass the full df)
#         # ----------------------------
        
#         # When user selects batter hand:
#         # (Replace current wagon + caught + pitchmaps calls with this single call)
#         # show_bowler_detail_from_all(
#         #     df_all=bdf if 'bdf' in globals() else df,   # prefer your bdf if available
#         #     player_selected=player_selected,
#         #     chosen_hand=chosen_hand,                    # e.g. 'LHB' or 'RHB' from your selectbox
#         #     normalize_to_rhb=True,                      # set False if you want true LHB mirror
#         #     title_suffix=f"vs Batter Hand: {chosen_hand}"
#         # )
        
#         # When user selects bowler kind/style, call similarly passing the df_use you already built:
#         # show_bowler_detail_from_all(df_all=df_use, player_selected=player_selected, chosen_hand=None,
#         #                            normalize_to_rhb=True, title_suffix=f"vs Bowler Kind: {chosen_kind}")

#         # require bdf in globals
#         if 'bdf' not in globals():
#             bdf = as_dataframe(df) # Use df if bdf not defined
    
#         # For bowler, adapt runs_col to bowler runs
#         runs_col = safe_get_col(bdf, ['bowlruns', 'score', 'runs'])
#         if runs_col is None:
#             st.error("No runs column found for bowling.")
#             st.stop()
    
#         # coerce runs col
#         bdf[runs_col] = pd.to_numeric(bdf.get(runs_col, 0), errors='coerce').fillna(0).astype(int)
#         pf[runs_col] = pd.to_numeric(pf.get(runs_col, 0), errors='coerce').fillna(0).astype(int)
    
#         # For bowling, group by bat_hand or other
#         if COL_BAT_HAND in pf.columns:
#             pf[COL_BAT_HAND] = pf[COL_BAT_HAND].astype(str).str.lower().fillna('unknown')
#             kinds = sorted(pf[COL_BAT_HAND].dropna().unique().tolist())
#             group_col = COL_BAT_HAND
#             group_label = 'Batter Hand'
#         else:
#             kinds = []
#             group_col = None
    
#         rows = []
#         if kinds:
#             for k in kinds:
#                 g = pf[pf[group_col] == k]
#                 m = compute_bowling_metrics(g, run_col=runs_col)
#                 m['group'] = k
#                 rows.append(m)
#         else:
#             m = compute_bowling_metrics(pf, run_col=runs_col)
#             m['group'] = 'unknown'
#             rows.append(m)
#         bk_df = pd.DataFrame(rows).set_index('group')
#         bk_df.index.name = group_label if group_label else 'Group'
    
#         st.markdown("<div style='font-weight:700; font-size:15px;'> Performance by batter hand </div>", unsafe_allow_html=True)
#         st.dataframe(bk_df, use_container_width=True)
    
#         # ---------- Bowling view (rearranged) ----------
#         bf = bdf[bdf[COL_BOWL] == player_selected].copy()
#         if bf.empty:
#             st.info("No bowling rows for this bowler.")
#             st.stop()
    
#         # choose runs column for bowler frame
#         if 'bowlruns' in bf.columns:
#             bf_runs_col = 'bowlruns'
#             bf[bf_runs_col] = pd.to_numeric(bf[bf_runs_col], errors='coerce').fillna(0).astype(int)
#         elif 'score' in bf.columns:
#             bf_runs_col = 'score'
#             bf[bf_runs_col] = pd.to_numeric(bf[bf_runs_col], errors='coerce').fillna(0).astype(int)
#         else:
#             bf_runs_col = COL_RUNS
#             bf[bf_runs_col] = pd.to_numeric(bf[bf_runs_col], errors='coerce').fillna(0).astype(int)
    
#         # Ensure dismissal / out / is_wkt in bowler frame
#         if COL_DISMISSAL in bf.columns:
#             bf['dismissal_clean'] = bf[COL_DISMISSAL].astype(str).str.lower().str.strip().replace({'nan':'','none':''})
#         else:
#             bf['dismissal_clean'] = ''
#         if COL_OUT in bf.columns:
#             bf['out_flag'] = pd.to_numeric(bf[COL_OUT], errors='coerce').fillna(0).astype(int)
#         else:
#             bf['out_flag'] = 0
#         bf['is_wkt'] = bf.apply(lambda r: 1 if is_bowler_wicket(r.get('out_flag',0), r.get('dismissal_clean','')) else 0, axis=1)
    
#         # split by batter handedness
#         if COL_BAT_HAND in bf.columns:
#             bf_lhb = bf[bf[COL_BAT_HAND].astype(str).str.upper().str.startswith('L')].copy()
#             bf_rhb = bf[bf[COL_BAT_HAND].astype(str).str.upper().str.startswith('R')].copy()
#         else:
#             bf_lhb = bf.iloc[0:0].copy()
#             bf_rhb = bf.iloc[0:0].copy()
    
#         # Collect unique values for batter hand exploration
#         def unique_vals_union(col):
#             vals = []
#             for df in (pf, bdf):
#                 if col in df.columns:
#                     vals.extend(df[col].dropna().astype(str).str.strip().tolist())
#             vals = sorted({v for v in vals if str(v).strip() != ''})
#             return vals
    
#         # ---- CLEAN batter hand options ----
#         raw_hands = unique_vals_union('bat_hand')
        
#         # Normalize to only LHB / RHB
#         clean_hands = []
#         for h in raw_hands:
#             if str(h).upper().startswith('L'):
#                 clean_hands.append('LHB')
#             elif str(h).upper().startswith('R'):
#                 clean_hands.append('RHB')
        
#         clean_hands = sorted(set(clean_hands))
#         chosen_hand = st.selectbox("Batter Hand", options=clean_hands)

#         def draw_bowler_caught_dismissals_wagon(df_in, bowler_name):
#             """
#             Plot caught dismissal locations for a bowler (wagon-style field).
#             Uses wagonX / wagonY like batter version.
#             """
#             if df_in.empty:
#                 st.info("No deliveries available.")
#                 return
        
#             # Filter caught dismissals credited to the bowler
#             if 'dismissal' not in df_in.columns or 'bowl' not in df_in.columns:
#                 st.warning("Required columns missing for caught dismissal plot.")
#                 return
        
#             caught_df = df_in[
#                 (df_in['bowl'] == bowler_name) &
#                 (df_in['dismissal'].astype(str).str.lower().str.contains('caught', na=False))
#             ].copy()
        
#             if caught_df.empty:
#                 st.info("No caught dismissals for this bowler.")
#                 return
        
#             # Require wagon coordinates
#             if 'wagonX' not in caught_df.columns or 'wagonY' not in caught_df.columns:
#                 st.warning("Missing wagonX / wagonY columns.")
#                 return
        
#             # Normalize wagon coordinates
#             center_x, center_y, radius = 184.0, 184.0, 184.0
#             caught_df['x'] = (pd.to_numeric(caught_df['wagonX'], errors='coerce') - center_x) / radius
#             caught_df['y'] = - (pd.to_numeric(caught_df['wagonY'], errors='coerce') - center_y) / radius
#             caught_df = caught_df.dropna(subset=['x', 'y'])
        
#             if caught_df.empty:
#                 st.info("No valid caught dismissal coordinates.")
#                 return
        
#             # Build plotly figure
#             fig = go.Figure()
        
#             # Field
#             fig.add_shape(type="circle", x0=-1, y0=-1, x1=1, y1=1,
#                           fillcolor="#228B22", line_color="black")
#             fig.add_shape(type="circle", x0=-0.5, y0=-0.5, x1=0.5, y1=0.5,
#                           fillcolor="#66bb6a", line_color="white")
        
#             # Pitch
#             fig.add_shape(type="rect", x0=-0.04, y0=-0.08, x1=0.04, y1=0.08,
#                           fillcolor="tan", line_color=None)
        
#             # Radials
#             for a in np.linspace(0, 2*np.pi, 9)[:-1]:
#                 fig.add_trace(go.Scatter(
#                     x=[0, np.cos(a)], y=[0, np.sin(a)],
#                     mode='lines', line=dict(color='white', width=1),
#                     opacity=0.25, showlegend=False
#                 ))
        
#             # Plot caught dismissal points
#             fig.add_trace(go.Scatter(
#                 x=caught_df['x'],
#                 y=caught_df['y'],
#                 mode='markers',
#                 marker=dict(color='red', size=12, line=dict(color='black', width=1.5)),
#                 hovertemplate=
#                     "<b>Batter:</b> %{customdata[0]}<br>"
#                     "<b>Bowler:</b> %{customdata[1]}<br>"
#                     "<b>Shot:</b> %{customdata[2]}<br>"
#                     "<extra></extra>",
#                 customdata=caught_df[['bat', 'bowl', 'shot']].fillna('').values,
#                 name="Caught dismissals"
#             ))
        
#             fig.update_layout(
#                 title="Caught Dismissal Locations",
#                 xaxis=dict(range=[-1.2, 1.2], visible=False),
#                 yaxis=dict(range=[-1.2, 1.2], visible=False, scaleanchor="x"),
#                 width=700, height=700,
#                 margin=dict(l=0, r=0, t=40, b=0)
#             )
        
#             st.plotly_chart(fig, use_container_width=True)

#         # ---------- robust map-lookup helpers ----------
#         def _norm_key(s):
#             if s is None:
#                 return ''
#             return str(s).strip().upper().replace(' ', '_').replace('-', '_')
    
#         def get_map_index(map_obj, raw_val):
#             if raw_val is None:
#                 return None
#             sval = str(raw_val).strip()
#             if sval == '' or sval.lower() in ('nan', 'none'):
#                 return None
    
#             if sval in map_obj:
#                 return int(map_obj[sval])
#             s_norm = _norm_key(sval)
#             for k in map_obj:
#                 try:
#                     if isinstance(k, str) and _norm_key(k) == s_norm:
#                         return int(map_obj[k])
#                 except Exception:
#                     continue
#             for k in map_obj:
#                 try:
#                     if isinstance(k, str) and (k.lower() in sval.lower() or sval.lower() in k.lower()):
#                         return int(map_obj[k])
#                 except Exception:
#                     continue
#             return None
    
#         # ---------- grids builder (adapted for bowling, but same as batting since metrics are batter performance vs bowler) ----------
#         def build_pitch_grids(df_in, line_col_name='line', length_col_name='length', runs_col_candidates=('bowlruns', 'score'),
#                               control_col='control', dismissal_col='dismissal'):
#             if 'length_map' in globals() and isinstance(length_map, dict) and len(length_map) > 0:
#                 try:
#                     max_idx = max(int(v) for v in length_map.values())
#                     n_rows = max(5, max_idx + 1)
#                 except Exception:
#                     n_rows = 5
#             else:
#                 n_rows = 5
#                 st.warning("length_map not found; defaulting to 5 rows.")
    
#             length_vals = df_in.get(length_col_name, pd.Series()).dropna().astype(str).str.lower().unique()
#             if any('full toss' in val for val in length_vals):
#                 n_rows = max(n_rows, 6)
    
#             n_cols = 5
    
#             count = np.zeros((n_rows, n_cols), dtype=int)
#             bounds = np.zeros((n_rows, n_cols), dtype=int)
#             dots = np.zeros((n_rows, n_cols), dtype=int)
#             runs = np.zeros((n_rows, n_cols), dtype=float)
#             wkt = np.zeros((n_rows, n_cols), dtype=int)
#             ctrl_not = np.zeros((n_rows, n_cols), dtype=int)
    
#             # choose runs column present (for bowling, prefer bowlruns)
#             runs_col = None
#             for c in runs_col_candidates:
#                 if c in df_in.columns:
#                     runs_col = c
#                     break
#             if runs_col is None:
#                 runs_col = None # will use 0
    
#             wkt_tokens = {'caught', 'bowled', 'stumped', 'lbw','leg before wicket','hit wicket'}
#             dismissal_series = df_in[dismissal_col].fillna('').astype(str).str.lower()
#             for _, row in df_in.iterrows():
#                 li = get_map_index(line_map, row.get(line_col_name, None)) if 'line_map' in globals() else None
#                 le = get_map_index(length_map, row.get(length_col_name, None)) if 'length_map' in globals() else None
#                 if li is None or le is None:
#                     continue
#                 if not (0 <= le < n_rows and 0 <= li < n_cols):
#                     continue
#                 count[le, li] += 1
#                 rv = 0
#                 if runs_col:
#                     try:
#                         rv = int(row.get(runs_col, 0) or 0)
#                     except:
#                         rv = 0
#                 runs[le, li] += rv
#                 if rv >= 4:
#                     bounds[le, li] += 1
#                 if rv == 0:
#                     dots[le, li] += 1
#                 dval = str(row.get(dismissal_col, '') or '').lower()
#                 if any(tok in dval for tok in wkt_tokens):
#                     wkt[le, li] += 1
#                 cval = row.get(control_col, None)
#                 if cval is not None:
#                     if isinstance(cval, str) and 'not' in cval.lower():
#                         ctrl_not[le, li] += 1
#                     elif isinstance(cval, (int, float)) and float(cval) == 0:
#                         ctrl_not[le, li] += 1
    
#             # compute Econ (runs/6 balls) and control %
#             econ = np.full(count.shape, np.nan)
#             ctrl_pct = np.full(count.shape, np.nan)
#             for i in range(n_rows):
#                 for j in range(n_cols):
#                     if count[i, j] > 0:
#                         econ[i, j] = (runs[i, j] * 6.0 / count[i, j])
#                         ctrl_pct[i, j] = (ctrl_not[i, j] / count[i, j]) * 100.0
            
#             return {
#                 'count': count, 'bounds': bounds, 'dots': dots,
#                 'runs': runs, 'econ': econ, 'ctrl_pct': ctrl_pct, 'wkt': wkt, 'n_rows': n_rows, 'n_cols': n_cols
#             }
    
#         # ---------- display utility (adapted for bowling metrics) ----------
#         def display_pitchmaps_from_df(df_src, title_prefix):
#             if df_src is None or df_src.empty:
#                 st.info(f"No deliveries to show for {title_prefix}")
#                 return
    
#             grids = build_pitch_grids(df_src)
    
#             # detect LHB presence among deliveries
#             bh_col_name = globals().get('bat_hand_col', 'bat_hand')
#             is_lhb = False
#             if bh_col_name in df_src.columns:
#                 hands = df_src[bh_col_name].dropna().astype(str).str.strip().unique()
#                 if any(h.upper().startswith('L') for h in hands):
#                     is_lhb = True
    
#             def maybe_flip(arr):
#                 return np.fliplr(arr) if is_lhb else arr.copy()
    
#             count = maybe_flip(grids['count'])
#             bounds = maybe_flip(grids['bounds'])
#             dots = maybe_flip(grids['dots'])
#             econ = maybe_flip(grids['econ'])
#             ctrl = maybe_flip(grids['ctrl_pct'])
#             wkt = maybe_flip(grids['wkt'])
#             runs = maybe_flip(grids['runs'])
    
#             total = count.sum() if count.sum() > 0 else 1.0
#             perc = count.astype(float) / total * 100.0
    
#             # xticks order: for RHB left->right, for LHB reversed (we flipped arrays already,
#             # so choose labels accordingly)
#             xticks_base = ['Wide Out Off', 'Outside Off', 'On Stumps', 'Down Leg', 'Wide Down Leg']
#             xticks = xticks_base[::-1] if is_lhb else xticks_base
    
#             # ytick labels depends on n_rows and whether FULL_TOSS exists
#             n_rows = grids['n_rows']
#             # prefer to show Full Toss on top if n_rows == 6
#             if n_rows >= 6:
#                 yticklabels = ['Short', 'Back of Length', 'Good', 'Full', 'Yorker', 'Full Toss'][:n_rows]
#             else:
#                 yticklabels = ['Short', 'Back of Length', 'Good', 'Full', 'Yorker'][:n_rows]
    
#             fig, axes = plt.subplots(3, 2, figsize=(14, 18))
#             plt.suptitle(f"{player_selected} — {title_prefix}", fontsize=16, weight='bold')
    
#             plot_list = [
#                 (perc, '% of balls (heat)', 'Blues'),
#                 (bounds, 'Boundaries conceded (count)', 'OrRd'),
#                 (dots, 'Dot balls (count)', 'Blues'),
#                 (econ, 'Econ (runs/6 balls)', 'Reds'),
#                 (ctrl, 'False Shot % (induced)', 'PuBu'),
#                 (runs, 'Runs conceded (sum)', 'Reds')
#             ]
    
#             for ax_idx, (ax, (arr, ttl, cmap)) in enumerate(zip(axes.flat, plot_list)):
#                 safe_arr = np.nan_to_num(arr.astype(float), nan=0.0)
#                 # autoscale vmax by 95th percentile to reduce outlier effect
#                 flat = safe_arr.flatten()
#                 if np.all(flat == 0):
#                     vmin, vmax = 0, 1
#                 else:
#                     vmin = float(np.nanmin(flat))
#                     vmax = float(np.nanpercentile(flat, 95))
#                     if vmax <= vmin:
#                         vmax = vmin + 1.0
    
#                 im = ax.imshow(safe_arr, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
#                 ax.set_title(ttl)
#                 ax.set_xticks(range(grids['n_cols'])); ax.set_yticks(range(grids['n_rows']))
#                 ax.set_xticklabels(xticks, rotation=45, ha='right')
#                 ax.set_yticklabels(yticklabels)
    
#                 # black minor-grid borders for cells
#                 ax.set_xticks(np.arange(-0.5, grids['n_cols'], 1), minor=True)
#                 ax.set_yticks(np.arange(-0.5, grids['n_rows'], 1), minor=True)
#                 ax.grid(which='minor', color='black', linewidth=0.6, alpha=0.95)
#                 ax.tick_params(which='minor', bottom=False, left=False)
    
#                 # annotate N W for wicket cells (e.g., '2 W') ONLY in the first plot (% of balls)
#                 if ax_idx == 0:
#                     for i in range(grids['n_rows']):
#                         for j in range(grids['n_cols']):
#                             w_count = int(wkt[i, j])
#                             if w_count > 0:
#                                 w_text = f"{w_count} W" if w_count > 1 else 'W'
#                                 ax.text(j, i, w_text, ha='center', va='center', fontsize=14, color='gold', weight='bold',
#                                         bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
    
#                 fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    
#             plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
#             # display via safe_st_pyplot if available
#             safe_fn = globals().get('safe_st_pyplot', None)
#             try:
#                 if callable(safe_fn):
#                     safe_fn(fig, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
#                 else:
#                     st.pyplot(fig)
#             except Exception:
#                 st.pyplot(fig)
#             finally:
#                 plt.close(fig)
    
#         # ---------- attempt to draw wagon chart using your existing function ----------
#         def draw_wagon_if_available(df_wagon, batter_name):
#             if 'draw_cricket_field_with_run_totals_requested' in globals() and callable(globals()['draw_cricket_field_with_run_totals_requested']):
#                 try:
#                     fig_w = draw_cricket_field_with_run_totals_requested(df_wagon, batter_name)
#                     safe_fn = globals().get('safe_st_pyplot', None)
#                     if callable(safe_fn):
#                         safe_fn(fig_w, max_pixels=40_000_000, fallback_set_max=False, use_container_width=True)
#                     else:
#                         st.pyplot(fig_w)
#                 except Exception as e:
#                     st.error(f"Wagon drawing function exists but raised: {e}")
#             else:
#                 st.warning("Wagon chart function not found; please ensure `draw_cricket_field_with_run_totals_requested` is defined earlier.")
    
#         # ---------- When user selects a batter hand ----------
# # ============================================================================
# # EXECUTION CODE - Place this where you select batter hand in bowler analysis
# # ============================================================================

#         # When user selects a batter hand (already in your code):
#         if chosen_hand and chosen_hand != '-- none --':
#             def filter_by_hand(df, col='bat_hand', hand=chosen_hand):
#                 if col not in df.columns:
#                     return df.iloc[0:0]
#                 mask = df[col].astype(str).str.lower().str.contains(str(hand).lower(), na=False)
#                 if not mask.any():
#                     norm_hand = _norm_key(hand)
#                     mask = df[col].apply(lambda x: _norm_key(x) == norm_hand)
#                 return df[mask].copy()
        
#             sel_pf = filter_by_hand(pf)
#             sel_bdf = filter_by_hand(bdf)
        
#             df_use = sel_pf if not sel_pf.empty else sel_bdf
#             if df_use.empty:
#                 st.info(f"No deliveries found for batter hand '{chosen_hand}'.")
#             else:
#                 st.markdown(f"### Detailed view — Batter Hand: {chosen_hand}")
                
#                 # Draw wagon wheel for bowler (runs conceded)
#                 draw_bowler_wagon_if_available(df_use, player_selected)
        
#                 # Draw caught dismissals wagon for bowler
#                 st.markdown(f"#### {player_selected}'s Caught Dismissals")
#                 draw_bowler_caught_dismissals_wagon(df_use, player_selected)
        
#                 # Display pitch maps
#                 display_pitchmaps_from_df(df_use, f"vs Batter Hand: {chosen_hand}")
#                 # ---------- When user selects a batter hand ----------





# Sidebar UI
else:
    import io
    import pandas as pd
    import streamlit as st
    def title_case_df_values(df, exclude_cols=None):
      """
      Title-case (First Letter Caps) all string values in the dataframe,
      except columns listed in exclude_cols.
      """
      if exclude_cols is None:
          exclude_cols = []
  
      df_out = df.copy()
      for col in df_out.columns:
          if col in exclude_cols:
              continue
          if df_out[col].dtype == object:
              df_out[col] = (
                  df_out[col]
                  .astype(str)
                  .str.strip()
                  .str.title()
              )
      return df_out


    # ================= LOAD DATA =================
    icr_25 = pd.read_csv("Datasets/Batting ICR IPL 2025.csv")
    bicr_25 = pd.read_csv("Datasets/Bowling ICR IPL 2025.csv")

    icr_24 = pd.read_csv("Datasets/icr_2024 (1).csv")
    bicr_24 = pd.read_csv("Datasets/bicr_2024.csv")

    icr_23 = pd.read_csv("Datasets/icr_2023.csv")
    bicr_23 = pd.read_csv("Datasets/bicr_2023.csv")

    # ================= PREPARE PERCENTILE COLS =================
    icr_24["ICR percentile"] = icr_24["icr_percentile_final"]
    icr_23["ICR percentile"] = icr_23["icr_percentile_final"]

    bicr_24["BICR percentile"] = bicr_24["icr_percentile_final"]
    bicr_23["BICR percentile"] = bicr_23["icr_percentile_final"]

    # ================= SELECT SAME DATA AS BEFORE =================
    icr_25 = icr_25[["batter", "Team", "Role", "matches", "ICR percentile"]]
    bicr_25 = bicr_25[["Bowler", "Team", "Role", "Matches", "BICR percentile"]]

    icr_24 = icr_24[["player", "team_bat", "comp_group", "n1", "ICR percentile"]]
    icr_23 = icr_23[["player", "team_bat", "comp_group", "n1", "ICR percentile"]]

    bicr_24 = bicr_24[["player", "team_bowl", "comp_group", "n1", "BICR percentile"]]
    bicr_23 = bicr_23[["player", "team_bowl", "comp_group", "n1", "BICR percentile"]]

    # ================= HEADER HARMONISATION (KEY FIX) =================
    BAT_COL_RENAME = {
        "batter": "Batter",
        "player": "Batter",
        "Team": "Team",
        "team_bat": "Team",
        "Role": "Role",
        "comp_group": "Role",
        "matches": "Matches",
        "n1": "Matches",
        "ICR percentile": "ICR Percentile"
    }

    BOWL_COL_RENAME = {
        "Bowler": "Bowler",
        "player": "Bowler",
        "Team": "Team",
        "team_bowl": "Team",
        "Role": "Role",
        "comp_group": "Role",
        "Matches": "Matches",
        "n1": "Matches",
        "BICR percentile": "BICR Percentile"
    }

    icr_25.rename(columns=BAT_COL_RENAME, inplace=True)
    icr_24.rename(columns=BAT_COL_RENAME, inplace=True)
    icr_23.rename(columns=BAT_COL_RENAME, inplace=True)

    bicr_25.rename(columns=BOWL_COL_RENAME, inplace=True)
    bicr_24.rename(columns=BOWL_COL_RENAME, inplace=True)
    bicr_23.rename(columns=BOWL_COL_RENAME, inplace=True)

    # ================= SIDEBAR =================
    st.markdown("## Integrated Contextual Ratings")

    year_choice = st.selectbox("Select Year", ["2023", "2024", "2025"], index=2)
    board_choice = st.radio(
        "Leaderboard",
        ["Batting Leaderboard", "Bowling Leaderboard"],
        index=0
    )
    show_top_n = st.number_input(
        "Show Top N Rows (0 = All)",
        min_value=0,
        value=50,
        step=10
    )

    # ================= PICK DATA =================
    if board_choice.startswith("Bat"):
        df_map = {
            "2023": icr_23,
            "2024": icr_24,
            "2025": icr_25
        }
        rating_label = "ICR Percentile"
    else:
        df_map = {
            "2023": bicr_23,
            "2024": bicr_24,
            "2025": bicr_25
        }
        rating_label = "BICR Percentile"

    df_show = df_map.get(year_choice)

    # ================= DISPLAY =================
    st.markdown("---")
    st.markdown(f"### {board_choice} — {year_choice}")

    if df_show is None or df_show.empty:
        st.warning("No data available.")
    else:
        df_disp = df_show.copy()

        # Apply Title Case ONLY for 2023 & 2024 (values, not headers)
        if year_choice in ["2023", "2024"]:
            # Do NOT title-case numeric percentile column
            if board_choice.startswith("Bat"):
                exclude = ["ICR Percentile", "Matches"]
            else:
                exclude = ["BICR Percentile", "Matches"]
        
            df_disp = title_case_df_values(df_disp, exclude_cols=exclude)


        # sort by percentile
        df_disp[rating_label] = pd.to_numeric(
            df_disp[rating_label], errors="coerce"
        )
        df_disp = df_disp.sort_values(
            rating_label, ascending=False
        )

        st.markdown(
            f"**Source:** IPL {year_choice} — {rating_label}"
        )

        if show_top_n > 0:
            df_disp = df_disp.head(int(show_top_n))

        st.dataframe(
            df_disp.reset_index(drop=True),
            use_container_width=True
        )

        # ================= DOWNLOAD =================
        buffer = io.StringIO()
        df_disp.to_csv(buffer, index=False)
        buffer.seek(0)
        st.markdown("""
          <div style="
              background: linear-gradient(135deg, #0b1f3a, #0f2f55);
              border-left: 4px solid #22d3ee;
              padding: 14px 18px;
              border-radius: 10px;
              margin: 14px 0 18px 0;
              color: #e5e7eb;
              line-height: 1.5;
          ">
          
          <div style="font-size:15px; font-weight:600; margin-bottom:6px;">
          What is Integrated Contextual Rating (ICR)?
          </div>
          
          <div style="font-size:14px; color:#cbd5f5;">
          ICR quantifies a player’s <b>true T20 impact</b> by adjusting performance for role, venue, opposition, and match context —
          not just what appears on the scorecard.
          </div>
          
          <div style="font-size:14px; color:#cbd5f5; margin-top:6px;">
          The framework explains <b>70%+ of team win variance</b>, more than any commonly used T20 metric today.
          </div>
          
          <div style="margin-top:10px;">
          <a href="https://open.substack.com/pub/theunseengame/p/the-unseen-game-icr-metric"
             target="_blank"
             style="
                color:#22d3ee;
                font-weight:600;
                text-decoration: underline;
             ">
          Read the full ICR methodology & insights →
          </a>
          </div>
          
          </div>
          """, unsafe_allow_html=True)

        st.download_button(
            label="Download CSV",
            data=buffer.getvalue().encode("utf-8"),
            file_name=f"{board_choice.replace(' ', '_')}_{year_choice}.csv",
            mime="text/csv"
        )

# -------------------- end sidebar section --------------------

