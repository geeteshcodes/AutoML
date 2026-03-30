import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import io
import matplotlib.pyplot as plt
import seaborn as sns
from backend_automl import train_automl

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  ──  dark industrial theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:       #0d0f14;
    --surface:  #13161e;
    --border:   #1f2430;
    --accent:   #00ffc2;
    --accent2:  #ff6b35;
    --muted:    #4a5568;
    --text:     #e2e8f0;
    --text-dim: #718096;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 2px; }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

[data-testid="stAppViewContainer"] > .main { background: var(--bg); }
[data-testid="block-container"] { padding-top: 1.5rem; }

h1, h2, h3, h4, h5 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}

[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: var(--text-dim) !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.1em; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Space Mono', monospace !important; font-size: 1.6rem !important; }

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden;
}

.stButton > button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: transparent !important;
    color: var(--accent) !important;
    border: 1.5px solid var(--accent) !important;
    border-radius: 4px !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
    box-shadow: 0 0 20px rgba(0,255,194,0.3) !important;
}
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: var(--bg) !important;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
}

[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border);
    gap: 0;
}
[data-testid="stTabs"] button[role="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-dim) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 0 !important;
    transition: all 0.2s;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--surface) !important;
}

input, .stNumberInput input {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    border-radius: 4px !important;
}

[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
}

hr { border-color: var(--border) !important; }

.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
}
.stat-card .label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.4rem;
}
.stat-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent);
}
.stat-card .sub {
    font-size: 0.75rem;
    color: var(--text-dim);
    margin-top: 0.2rem;
}
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
}
.section-header .badge {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--bg);
    background: var(--accent);
    padding: 0.15rem 0.5rem;
    border-radius: 2px;
}
.model-chip {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    background: var(--border);
    color: var(--text);
    padding: 0.2rem 0.6rem;
    border-radius: 3px;
    margin: 0.15rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def apply_dark_mpl():
    plt.rcParams.update({
        "figure.facecolor": "#13161e",
        "axes.facecolor": "#13161e",
        "axes.edgecolor": "#1f2430",
        "axes.labelcolor": "#e2e8f0",
        "xtick.color": "#718096",
        "ytick.color": "#718096",
        "text.color": "#e2e8f0",
        "grid.color": "#1f2430",
        "font.family": "monospace",
    })

def stat_card(label, value, sub=""):
    return f"""<div class="stat-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {'<div class="sub">' + sub + '</div>' if sub else ''}
    </div>"""

def section_title(badge, title):
    st.markdown(f"""
    <div class="section-header">
        <span class="badge">{badge}</span>
        <span style="font-size:1.05rem;font-weight:700;letter-spacing:-0.01em">{title}</span>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, default in [
    ("model", None), ("df", None), ("target", None),
    ("model_filename", None), ("model_name", None),
    ("score", None), ("task", "classification"),
    ("train_time", None), ("pkl_bytes", None),
    ("dropped_cols", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 1.2rem">
        <div style="font-size:1.4rem;font-weight:800;letter-spacing:-0.03em;color:#00ffc2">⚡ AutoML</div>
        <div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;color:#4a5568;margin-top:0.2rem">Studio v2.0</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown('<div style="font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:#4a5568;margin-bottom:0.6rem">Pipeline Status</div>', unsafe_allow_html=True)

    if st.session_state.df is not None:
        df_s = st.session_state.df
        st.success("✓ Dataset loaded")
        st.caption(f"Rows: **{df_s.shape[0]:,}** · Cols: **{df_s.shape[1]}**")
    else:
        st.info("No dataset loaded")

    if st.session_state.model is not None:
        st.success("✓ Model trained")
        st.caption(f"**{st.session_state.model_name}**")
        st.caption(f"Score: **{st.session_state.score:.4f}**")

    if st.session_state.dropped_cols:
        st.warning(f"⚠ {len(st.session_state.dropped_cols)} col(s) excluded")

    st.divider()

    # EDA options (only when dataset loaded)
    if st.session_state.df is not None:
        st.markdown('<div style="font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:#4a5568;margin-bottom:0.6rem">EDA Options</div>', unsafe_allow_html=True)
        show_corr = st.checkbox("Correlation heatmap", value=True)
        show_dist = st.checkbox("Distributions", value=True)
        max_dist_cols = st.slider("Max dist. cols", 2, 12, 6)
    else:
        show_corr, show_dist, max_dist_cols = True, True, 6

    st.divider()
    st.markdown('<div style="font-size:0.62rem;color:#4a5568;text-align:center;padding:0.5rem 0">Scikit-learn · XGBoost · Streamlit</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("""
<h1 style="font-size:2.1rem;margin-bottom:0;letter-spacing:-0.04em">Automated ML Pipeline</h1>
<p style="color:#718096;font-size:0.85rem;margin-top:0.25rem;font-family:'Space Mono',monospace">
Upload → Explore → Train → Predict → Export
</p>""", unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_data, tab_eda, tab_engineer, tab_train, tab_predict = st.tabs([
    "01 · DATA", "02 · EXPLORE", "03 · ENGINEER", "04 · TRAIN", "05 · PREDICT"
])


# ══════════════════════════════════════════════
# TAB 1 · DATA
# ══════════════════════════════════════════════
with tab_data:
    section_title("STEP 1", "Upload your dataset")

    uploaded_file = st.file_uploader(
        "Drop a CSV file here, or click to browse",
        type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.model = None
        st.session_state.score = None

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(stat_card("Rows", f"{df.shape[0]:,}"), unsafe_allow_html=True)
        c2.markdown(stat_card("Columns", str(df.shape[1])), unsafe_allow_html=True)
        c3.markdown(stat_card("Numeric", str(df.select_dtypes(include="number").shape[1])), unsafe_allow_html=True)
        c4.markdown(stat_card("Missing", f"{df.isnull().sum().sum():,}", "total cells"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns([2, 1])
        with col_left:
            section_title("PREVIEW", "First 50 rows")
            st.dataframe(df.head(50), use_container_width=True, height=320)
        with col_right:
            section_title("SCHEMA", "Column info")
            schema_df = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.astype(str).values,
                "Nulls": df.isnull().sum().values,
                "Unique": df.nunique().values,
            })
            st.dataframe(schema_df, use_container_width=True, height=320, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_title("CONFIG", "Training configuration")

        cfg1, cfg2, cfg3 = st.columns(3)
        with cfg1:
            target = st.selectbox("🎯 Target column", df.columns)
            st.session_state.target = target
        with cfg2:
            task = st.radio("📋 Task type", ["classification", "regression"])
            st.session_state.task = task
        with cfg3:
            st.markdown("<br>", unsafe_allow_html=True)
            t_info = df[target]
            st.markdown(stat_card("Target info", f"{t_info.nunique()} unique", f"dtype: {t_info.dtype}"), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#4a5568">
            <div style="font-size:3rem;margin-bottom:1rem">📂</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.8rem;letter-spacing:0.1em;text-transform:uppercase">No file loaded yet</div>
            <div style="font-size:0.8rem;margin-top:0.5rem">Upload a CSV to get started</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 · EDA
# ══════════════════════════════════════════════
with tab_eda:
    if st.session_state.df is None:
        st.info("Upload a dataset first in the **DATA** tab.")
    else:
        df = st.session_state.df
        target = st.session_state.target
        apply_dark_mpl()

        section_title("STATS", "Descriptive statistics")
        st.dataframe(df.describe(include="all").T.round(3), use_container_width=True, height=250)

        st.markdown("<br>", unsafe_allow_html=True)

        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            section_title("MISSING", "Null value breakdown")
            miss_df = pd.DataFrame({
                "Column": missing.index,
                "Missing": missing.values,
                "Pct %": (missing.values / len(df) * 100).round(2),
            })
            st.dataframe(miss_df, use_container_width=True, hide_index=True)
        else:
            st.success("✓ No missing values detected")

        st.markdown("<br>", unsafe_allow_html=True)

        if show_dist:
            section_title("DISTRIBUTIONS", "Numeric feature distributions")
            num_cols = df.select_dtypes(include="number").columns.tolist()
            plot_cols = [c for c in num_cols if c != target][:max_dist_cols]
            if plot_cols:
                n = len(plot_cols)
                ncols = min(3, n)
                nrows = (n + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows))
                axes_flat = np.array(axes).flatten() if n > 1 else [axes]
                palette = ["#00ffc2", "#ff6b35", "#7c6af5", "#f7d842", "#ff4b8d", "#00b4d8"]
                for i, col in enumerate(plot_cols):
                    ax = axes_flat[i]
                    ax.hist(df[col].dropna(), bins=30, color=palette[i % len(palette)], alpha=0.85, edgecolor="none")
                    ax.set_title(col, fontsize=8, fontweight="bold", color="#e2e8f0")
                    ax.tick_params(labelsize=7)
                for j in range(len(plot_cols), len(axes_flat)):
                    axes_flat[j].set_visible(False)
                plt.tight_layout(pad=1.5)
                st.pyplot(fig, use_container_width=True)
                plt.close()

        st.markdown("<br>", unsafe_allow_html=True)

        if show_corr:
            section_title("CORRELATION", "Pearson correlation matrix")
            num_df = df.select_dtypes(include="number")
            if len(num_df.columns) >= 2:
                corr = num_df.corr()
                n = len(corr)
                fig, ax = plt.subplots(figsize=(min(n * 0.7 + 2, 14), min(n * 0.6 + 1.5, 12)))
                mask = np.zeros_like(corr, dtype=bool)
                mask[np.triu_indices_from(mask, k=1)] = True
                sns.heatmap(
                    corr, mask=mask, ax=ax,
                    cmap=sns.diverging_palette(145, 10, as_cmap=True),
                    center=0, linewidths=0.3, linecolor="#0d0f14",
                    annot=n <= 15, fmt=".2f", annot_kws={"size": 7},
                    cbar_kws={"shrink": 0.7},
                )
                ax.tick_params(labelsize=8, colors="#718096")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

        if target:
            st.markdown("<br>", unsafe_allow_html=True)
            section_title("TARGET", f"Distribution of '{target}'")
            fig, ax = plt.subplots(figsize=(8, 3.5))
            if df[target].dtype in ["object", "category"] or df[target].nunique() <= 20:
                counts = df[target].value_counts()
                ax.bar(counts.index.astype(str), counts.values, color="#00ffc2", alpha=0.85, edgecolor="none")
                ax.set_xlabel("Class", fontsize=8)
            else:
                ax.hist(df[target].dropna(), bins=40, color="#00ffc2", alpha=0.85, edgecolor="none")
                ax.set_xlabel(target, fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()



# ══════════════════════════════════════════════
# TAB 3 · ENGINEER
# ══════════════════════════════════════════════
with tab_engineer:
    if st.session_state.df is None:
        st.info("Upload a dataset first in the **DATA** tab.")
    else:
        df = st.session_state.df
        target = st.session_state.target or df.columns[-1]

        section_title("AUTO-DETECT", "ID-like columns (high cardinality)")

        st.markdown("""
        <div style="font-size:0.8rem;color:#718096;margin-bottom:1.2rem;font-family:'Space Mono',monospace">
        Columns where unique values ≥ 95% of rows are likely identifiers (IDs, names, hashes) that
        add no signal to the model and should be dropped.
        </div>""", unsafe_allow_html=True)

        # Auto-detect: unique ratio >= 0.95, excluding target
        feature_cols = [c for c in df.columns if c != target]
        unique_ratio = df[feature_cols].nunique() / len(df)
        auto_flagged = unique_ratio[unique_ratio >= 0.95].index.tolist()

        # Build summary table for all feature columns
        col_summary = pd.DataFrame({
            "Column": feature_cols,
            "Type": df[feature_cols].dtypes.astype(str).values,
            "Unique": df[feature_cols].nunique().values,
            "Unique %": (unique_ratio.values * 100).round(1),
            "Auto-flagged": ["⚠️ ID-like" if c in auto_flagged else "✓ OK" for c in feature_cols],
        })

        st.dataframe(col_summary, use_container_width=True, height=280, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_title("SELECT", "Choose columns to drop before training")

        # Pre-select auto-flagged, user can override
        cols_to_drop = st.multiselect(
            "Columns to exclude from training",
            options=feature_cols,
            default=[c for c in auto_flagged if c in feature_cols],
            help="Auto-flagged columns are pre-selected. You can add or remove any column.",
        )

        # Preview
        if cols_to_drop:
            st.markdown("<br>", unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            remaining = [c for c in feature_cols if c not in cols_to_drop]
            d1.markdown(stat_card("Dropping", str(len(cols_to_drop)), "columns"), unsafe_allow_html=True)
            d2.markdown(stat_card("Remaining", str(len(remaining)), "feature columns"), unsafe_allow_html=True)
            d3.markdown(stat_card("Target", str(target), "kept"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("👁  Preview training-ready dataset"):
                preview_df = df.drop(columns=cols_to_drop)
                st.dataframe(preview_df.head(20), use_container_width=True, height=260)
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            st.success("✓ No columns dropped — all features will be used for training.")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✅ Confirm & Save Column Selection", type="primary"):
            st.session_state.dropped_cols = cols_to_drop
            st.session_state.model = None  # reset model when features change
            if cols_to_drop:
                st.success(f"Saved! {len(cols_to_drop)} column(s) will be excluded from training: {', '.join(cols_to_drop)}")
            else:
                st.success("Saved! All feature columns will be used.")

        # Show current saved state
        if st.session_state.dropped_cols:
            st.markdown("<br>", unsafe_allow_html=True)
            chips = "".join(f'<span class="model-chip" style="background:rgba(255,107,53,0.15);color:#ff6b35;border:1px solid #ff6b35">{c}</span>' for c in st.session_state.dropped_cols)
            st.markdown(f"""
            <div style="font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:#4a5568;margin-bottom:0.4rem">Currently excluded</div>
            <div>{chips}</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4 · TRAIN
# ══════════════════════════════════════════════
with tab_train:
    if st.session_state.df is None:
        st.info("Upload a dataset first in the **DATA** tab.")
    else:
        df = st.session_state.df
        target = st.session_state.target or df.columns[-1]
        task = st.session_state.task or "classification"
        dropped = st.session_state.dropped_cols or []

        # Apply dropped columns
        train_df = df.drop(columns=dropped) if dropped else df

        section_title("READY", "Training configuration summary")

        s1, s2, s3, s4 = st.columns(4)
        s1.markdown(stat_card("Target", str(target)), unsafe_allow_html=True)
        s2.markdown(stat_card("Task", task.upper()), unsafe_allow_html=True)
        s3.markdown(stat_card("Features", str(train_df.shape[1] - 1)), unsafe_allow_html=True)
        s4.markdown(stat_card("Dropped", str(len(dropped)), "columns excluded"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("ℹ️  Which models will be tried?"):
            if task == "classification":
                models_list = ["LogisticRegression", "RandomForest", "SVC", "KNeighbors",
                               "GradientBoosting", "AdaBoost", "ExtraTrees", "SGD", "XGBoost"]
            else:
                models_list = ["LinearRegression", "Ridge", "Lasso", "RandomForest",
                               "GradientBoosting", "ExtraTrees", "KNeighbors", "XGBoost"]
            chips = "".join(f'<span class="model-chip">{m}</span>' for m in models_list)
            st.markdown(f'<div style="padding:0.5rem 0">{chips}</div>', unsafe_allow_html=True)
            st.caption("5-fold GridSearchCV · Automatic preprocessing · PCA on numeric features")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("⚡ Launch Training", type="primary"):
            with st.status("Training AutoML pipeline…", expanded=True) as status:
                st.write("🔍 Building preprocessing pipeline…")
                progress = st.progress(0)
                for i in range(1, 26):
                    time.sleep(0.03)
                    progress.progress(i * 2)
                st.write("🧠 Running GridSearchCV across all models…")
                start = time.time()
                model, score, model_name = train_automl(train_df, target, task)
                end = time.time()
                elapsed = round(end - start, 2)
                for i in range(50, 101):
                    time.sleep(0.01)
                    progress.progress(i)
                st.write("💾 Saving model artifact…")
                buf = io.BytesIO()
                joblib.dump(model, buf)
                buf.seek(0)
                st.session_state.model = model
                st.session_state.model_name = model_name
                st.session_state.score = score
                st.session_state.train_time = elapsed
                st.session_state.model_filename = f"model_{model_name}.pkl"
                st.session_state.pkl_bytes = buf.read()
                status.update(label="✅ Training complete!", state="complete", expanded=False)

        if st.session_state.model:
            st.markdown("<br>", unsafe_allow_html=True)
            section_title("RESULTS", "Best model found")

            r1, r2, r3, r4 = st.columns(4)
            metric_label = "Accuracy" if st.session_state.task == "classification" else "R² Score"
            r1.markdown(stat_card("Best Model", st.session_state.model_name or "—"), unsafe_allow_html=True)
            r2.markdown(stat_card(metric_label, f"{st.session_state.score:.4f}"), unsafe_allow_html=True)
            r3.markdown(stat_card("Train Time", f"{st.session_state.train_time}s"), unsafe_allow_html=True)
            r4.markdown(stat_card("CV Folds", "5"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Feature importances
            mdl = st.session_state.model.named_steps.get("model")
            if hasattr(mdl, "feature_importances_"):
                section_title("FEATURES", "Feature importances (post-preprocessing)")
                apply_dark_mpl()
                try:
                    importances = mdl.feature_importances_
                    n_imp = len(importances)
                    feat_names = [f"Feature {i+1}" for i in range(n_imp)]
                    imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
                    imp_df = imp_df.sort_values("Importance", ascending=True).tail(20)
                    fig, ax = plt.subplots(figsize=(8, max(3, len(imp_df) * 0.38)))
                    ax.barh(imp_df["Feature"], imp_df["Importance"], color="#00ffc2", alpha=0.85, edgecolor="none")
                    ax.set_xlabel("Importance", fontsize=8)
                    ax.tick_params(labelsize=7)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
                except Exception:
                    pass

            st.markdown("<br>", unsafe_allow_html=True)
            if st.session_state.pkl_bytes:
                st.download_button(
                    "⬇  Download trained model (.pkl)",
                    data=st.session_state.pkl_bytes,
                    file_name=st.session_state.model_filename,
                    mime="application/octet-stream",
                )


# ══════════════════════════════════════════════
# TAB 4 · PREDICT
# ══════════════════════════════════════════════
with tab_predict:
    if st.session_state.model is None:
        st.info("Train a model first in the **TRAIN** tab.")
    else:
        df = st.session_state.df
        target = st.session_state.target
        model = st.session_state.model
        dropped = st.session_state.dropped_cols or []
        predict_df = df.drop(columns=dropped) if dropped else df
        feature_cols = predict_df.drop(columns=[target]).columns.tolist()

        pred_tab1, pred_tab2 = st.tabs(["BATCH PREDICTION", "ROW PREDICTION"])

        with pred_tab1:
            section_title("BATCH", "Predict on entire dataset")
            st.markdown(f"""
            <div style="font-size:0.8rem;color:#718096;margin-bottom:1.2rem;font-family:'Space Mono',monospace">
            Model: <b style="color:#e2e8f0">{st.session_state.model_name}</b> ·
            <b style="color:#e2e8f0">{df.shape[0]:,}</b> rows ·
            <b style="color:#e2e8f0">{len(feature_cols)}</b> features
            </div>""", unsafe_allow_html=True)

            if st.button("▶ Run batch prediction", use_container_width=False):
                with st.spinner("Predicting…"):
                    preds = model.predict(predict_df[feature_cols])
                    result_df = predict_df.copy()
                    result_df["⚡ Prediction"] = preds

                st.success(f"✓ {len(preds):,} predictions generated")
                st.dataframe(result_df, use_container_width=True, height=380)

                # Prediction distribution
                apply_dark_mpl()
                fig, ax = plt.subplots(figsize=(8, 3))
                pred_series = pd.Series(preds)
                if pred_series.dtype in ["object", "category"] or pred_series.nunique() <= 20:
                    counts = pred_series.value_counts()
                    ax.bar(counts.index.astype(str), counts.values, color="#00ffc2", alpha=0.85, edgecolor="none")
                    ax.set_xlabel("Predicted class", fontsize=8)
                else:
                    ax.hist(preds, bins=40, color="#00ffc2", alpha=0.85, edgecolor="none")
                    ax.set_xlabel("Predicted value", fontsize=8)
                ax.set_title("Prediction distribution", fontsize=9, color="#e2e8f0")
                ax.set_ylabel("Count", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                st.download_button(
                    "⬇  Download predictions CSV",
                    result_df.to_csv(index=False).encode(),
                    "predictions.csv",
                    "text/csv",
                )

        with pred_tab2:
            section_title("MANUAL", "Predict for a single row")
            st.markdown('<div style="font-size:0.78rem;color:#718096;margin-bottom:1.2rem;font-family:\'Space Mono\',monospace">Fill in the fields below and hit Predict.</div>', unsafe_allow_html=True)

            input_data = {}
            num_feats = [f for f in feature_cols if pd.api.types.is_numeric_dtype(df[f])]
            cat_feats = [f for f in feature_cols if not pd.api.types.is_numeric_dtype(df[f])]

            if num_feats:
                st.markdown('<div style="font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;color:#4a5568;margin-bottom:0.5rem">Numeric features</div>', unsafe_allow_html=True)
                num_grid = st.columns(min(4, len(num_feats)))
                for idx, feat_name in enumerate(num_feats):
                    with num_grid[idx % len(num_grid)]:
                        med_val = float(df[feat_name].median())
                        input_data[feat_name] = st.number_input(
                            feat_name,
                            value=med_val,
                            key=f"num_input_{feat_name}",
                            help=f"Median: {med_val:.3f} | Range: [{df[feat_name].min():.3f}, {df[feat_name].max():.3f}]"
                        )

            if cat_feats:
                st.markdown('<br><div style="font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;color:#4a5568;margin-bottom:0.5rem">Categorical features</div>', unsafe_allow_html=True)
                cat_grid = st.columns(min(3, len(cat_feats)))
                for idx, feat_name in enumerate(cat_feats):
                    with cat_grid[idx % len(cat_grid)]:
                        options = df[feat_name].dropna().unique().tolist()
                        if options:
                            input_data[feat_name] = st.selectbox(
                                feat_name, options, key=f"cat_input_{feat_name}"
                            )
                        else:
                            input_data[feat_name] = st.text_input(
                                feat_name, key=f"txt_input_{feat_name}"
                            )

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⚡ Predict this row", type="primary", key="row_predict_btn"):
                try:
                    row_df = pd.DataFrame([input_data])
                    pred = model.predict(row_df[feature_cols])[0]

                    prob_str = ""
                    if hasattr(model, "predict_proba"):
                        try:
                            proba = model.predict_proba(row_df[feature_cols])[0]
                            classes = model.classes_
                            prob_str = " · ".join(f"{c}: {p:.1%}" for c, p in zip(classes, proba))
                        except Exception:
                            pass

                    st.markdown(f"""
                    <div class="stat-card" style="border-color:#00ffc2;margin-top:0.5rem">
                        <div class="label">Prediction result</div>
                        <div class="value" style="font-size:2rem">{pred}</div>
                        {'<div class="sub">' + prob_str + '</div>' if prob_str else ''}
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
