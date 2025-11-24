import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler

from src.services.pipeliine.source_data.preprocessing.cleaning_service import impute
from src.ui.common import end_tab_scroll, begin_tab_scroll, section_panel, load_active_feature_master_from_session
from src.utils.data_io_utils import save_processed, save_profile
from src.utils.log_utils import get_logger

LOGGER = get_logger("ui_cleaning")


def _num_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    num = df[cols].select_dtypes(include="number")
    if num.empty:
        return pd.DataFrame(columns=["column", "mean", "median", "std", "min", "q1", "q3", "max", "missing"])
    return pd.DataFrame({
        "column": num.columns,
        "mean": [num[c].mean() for c in num.columns],
        "median": [num[c].median() for c in num.columns],
        "std": [num[c].std(ddof=1) for c in num.columns],
        "min": [num[c].min() for c in num.columns],
        "q1": [num[c].quantile(0.25) for c in num.columns],
        "q3": [num[c].quantile(0.75) for c in num.columns],
        "max": [num[c].max() for c in num.columns],
        "missing": [num[c].isna().sum() for c in num.columns],
    })


def _cat_summary(df: pd.DataFrame, cols: list[str], top_n: int = 15) -> pd.DataFrame:
    rows = []
    for c in cols:
        vc = df[c].astype("string").fillna("Unknown").value_counts(dropna=False).head(top_n)
        rows.append(pd.DataFrame({"column": c, "value": vc.index, "count": vc.values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["column", "value", "count"])


def _normalize_series(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.upper().replace({"NAN": "UNKNOWN"}).fillna("UNKNOWN")


def _summary_for_imputation(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    num = df[cols].select_dtypes(include="number")
    if num.empty:
        return pd.DataFrame(columns=["column", "mean", "median", "missing"])
    out = pd.DataFrame({
        "column": num.columns,
        "mean": [num[c].mean() for c in num.columns],
        "median": [num[c].median() for c in num.columns],
        "missing": [num[c].isna().sum() for c in num.columns],
    })
    return out


def _log_step(step_name: str, payload: dict):
    st.session_state.steps.append({step_name: payload})


def _tab_impute():
    df = st.session_state.df.copy()
    num_cols = df.select_dtypes(include="number").columns.tolist()

    with st.form("impute_form"):
        strat = st.selectbox("Strategy", ["median", "mean"])
        cols_sel = st.multiselect("Columns (blank = all)", df.columns.tolist())
        tol_pct = st.slider("Stability tolerance (%)", 0, 20, 2, 1,
                            help="Allowed change in mean/median after imputation.")
        submitted = st.form_submit_button("Apply imputation")

    if submitted:
        # --- BEFORE
        cols_to_check = cols_sel or num_cols
        before = _summary_for_imputation(df, cols_to_check)

        # --- APPLY
        st.session_state.cleaned_df = impute(df, strategy=strat, columns=(cols_sel or None))

        # --- AFTER
        df_after = st.session_state.cleaned_df
        after = _summary_for_imputation(df_after, cols_to_check)

        # --- COMPARE
        comp = before.merge(after, on="column", suffixes=("_before", "_after"))
        # avoid divide-by-zero by using where/abs with small epsilon
        eps = 1e-12
        comp["mean_delta"] = comp["mean_after"] - comp["mean_before"]
        comp["median_delta"] = comp["median_after"] - comp["median_before"]
        comp["mean_delta_pct"] = (comp["mean_delta"] / comp["mean_before"].where(comp["mean_before"].abs() > eps,
                                                                                 eps)) * 100
        comp["median_delta_pct"] = (comp["median_delta"] / comp["median_before"].where(
            comp["median_before"].abs() > eps, eps)) * 100
        comp["missing_reduction"] = comp["missing_before"] - comp["missing_after"]

        tol = float(tol_pct)
        comp["mean_stable"] = comp["mean_delta_pct"].abs() <= tol
        comp["median_stable"] = comp["median_delta_pct"].abs() <= tol

        # --- UI OUTPUT
        st.success("Imputation applied.")
        with st.container(border=True):
            st.markdown("Post-imputation summary (before vs after)")
            st.caption("Green = within tolerance; Red = exceeded tolerance")
            st.dataframe(
                comp[[
                    "column",
                    "missing_before", "missing_after", "missing_reduction",
                    "mean_before", "mean_after", "mean_delta", "mean_delta_pct", "mean_stable",
                    "median_before", "median_after", "median_delta", "median_delta_pct", "median_stable",
                ]].round(6),
                use_container_width=True
            )

            violations = comp[(~comp["mean_stable"]) | (~comp["median_stable"])]
            if not violations.empty:
                st.caption("Columns exceeding tolerance")
                fig, ax = plt.subplots()
                ax.bar(violations["column"], violations["mean_delta_pct"].abs())
                ax.set_ylabel("|Mean Δ%|")
                ax.set_xticklabels(violations["column"], rotation=45, ha="right")
                st.pyplot(fig, clear_figure=True)

            # At-a-glance counters
            total_cols = len(comp)
            mean_ok = int(comp["mean_stable"].sum())
            median_ok = int(comp["median_stable"].sum())
            col1, col2, col3 = st.columns(3)
            col1.metric("Columns checked", total_cols)
            col2.metric("Mean stable", f"{mean_ok}/{total_cols}")
            col3.metric("Median stable", f"{median_ok}/{total_cols}")

        # --- Persist to steps/manifest so it’s traceable in Save
        report = {
            "strategy": strat,
            "columns": cols_sel or "ALL",
            "tolerance_pct": tol_pct,
            "summary": comp.to_dict(orient="records"),
        }

        _log_step("post_impute_check", report)


def _tab_outliers():
    df = st.session_state.cleaned_df.copy()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    st.subheader("Outlier Handling")

    method = st.selectbox("Method", ["IQR Filter", "Winsorize"])
    cols_sel = st.multiselect("Numeric columns (blank = all numeric)", num_cols)
    iqr_k = st.slider("IQR multiplier (typical=1.5)", 1.0, 3.0, 1.5, 0.1)
    submitted = st.button("Apply outlier handling")

    if submitted:
        cols = cols_sel or num_cols
        before = _num_summary(df, cols)

        thresholds = {}
        counts = {}

        for c in cols:
            q1 = df[c].quantile(0.25)
            q3 = df[c].quantile(0.75)
            iqr = q3 - q1
            lo = q1 - iqr_k * iqr
            hi = q3 + iqr_k * iqr
            thresholds[c] = {"low": float(lo), "high": float(hi)}

            if method == "IQR Filter":
                before_n = len(df)
                df_work = df[(df[c] >= lo) & (df[c] <= hi)]
                counts[c] = {"removed": int(before_n - len(df))}
            else:
                # Winsorize (clip)
                s_before = df[c].copy()
                df[c] = df[c].clip(lower=lo, upper=hi)
                changed = (s_before != df[c]).sum()
                counts[c] = {"clipped": int(changed)}

        st.session_state.cleaned_df = df
        after = _num_summary(df, cols)

        st.success("Outlier handling applied.")
        with st.container(border=True):
            st.markdown("Pre vs Post Summary")
            comp = before.merge(after, on="column", suffixes=("_before", "_after"))
            comp["std_delta"] = comp["std_after"] - comp["std_before"]
            st.dataframe(comp.round(6), use_container_width=True)

            st.caption("Boxplot (post)")
            fig, ax = plt.subplots()
            df[cols].plot(kind="box", ax=ax, vert=False)
            st.pyplot(fig, clear_figure=True)

        _log_step("outliers", {
            "method": method,
            "iqr_multiplier": iqr_k,
            "thresholds": thresholds,
            "impact": counts,
            "summary_post": after.round(6).to_dict(orient="records"),
        })


def _tab_encoding():
    df = st.session_state.cleaned_df.copy()
    st.subheader("Categorical Encoding")
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    cols_sel = st.multiselect("Categorical columns", cat_cols, default=cat_cols)
    method = st.selectbox("Method", ["One-Hot", "Target Encoding (mean price)"])
    keep_original = st.checkbox("Preserve original columns", True)

    if st.button("Apply encoding"):
        before_card = {c: int(df[c].nunique(dropna=False)) for c in cols_sel}

        # Standardize names
        for c in cols_sel:
            df[c] = _normalize_series(df[c])

        # Show normalization diffs
        with st.container(border=True):
            st.markdown("Normalization preview")
            prev = []
            for c in cols_sel:
                sample = (st.session_state.cleaned_df[c].astype("string")
                          .head(50).to_frame(c).assign(NORMALIZED=df[c].head(50)))
                prev.append(sample)
                st.write(f"Column: **{c}**")
                st.dataframe(sample, use_container_width=True)
            _ = prev  # just to avoid linter complaints

        if method == "One-Hot":
            df = pd.get_dummies(df, columns=cols_sel, drop_first=False, dtype=int)
            enc_info = {"created_columns": [c for c in df.columns if any(c.startswith(x + "_") for x in cols_sel)]}
        else:
            # Target encoding against 'price' (adjust if your target differs)
            enc_info = {"mapping": {}}
            for c in cols_sel:
                m = df.groupby(c)["price"].mean().to_dict()
                df[f"{c}__target"] = df[c].map(m)
                enc_info["mapping"][c] = {k: float(v) for k, v in m.items()}

        # Optionally drop originals
        if not keep_original and method != "One-Hot":
            df = df.drop(columns=cols_sel)

        after_card = {c: (int(df[c].nunique()) if c in df.columns else -1) for c in cols_sel}
        st.session_state.cleaned_df = df

        # Visibility: cardinality + leakage warning
        with st.container(border=True):
            st.markdown("Encoding Check")
            card_df = pd.DataFrame({
                "column": list(before_card.keys()),
                "cardinality_before": list(before_card.values()),
                "cardinality_after": [after_card[c] for c in before_card.keys()]
            })
            st.dataframe(card_df, use_container_width=True)
            if method == "Target Encoding (mean price)":
                st.warning(
                    "Ensure target encoding is fitted on TRAIN only to avoid leakage. In UI demo we apply to full DF for preview.")

        _log_step("encoding", {
            "method": method,
            "columns": cols_sel,
            "keep_original": keep_original,
            "cardinality_before": before_card,
            "cardinality_after": after_card,
            "details": enc_info,
        })
        st.success("Encoding applied.")


def _tab_scaling():
    df = st.session_state.cleaned_df.copy()
    st.subheader("Scaling")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cols_sel = st.multiselect("Numeric columns to scale", num_cols)
    scaler_name = st.selectbox("Scaler", ["StandardScaler", "RobustScaler"])
    if st.button("Apply scaling"):
        cols = cols_sel or num_cols
        before = _num_summary(df, cols)

        X = st.session_state.df[cols].values
        scaler = StandardScaler() if scaler_name == "StandardScaler" else RobustScaler()
        Xs = scaler.fit_transform(X)
        for i, c in enumerate(cols):
            st.session_state.df[c] = Xs[:, i]

        after = _num_summary(df, cols)

        with st.container(border=True):
            st.markdown("Post-scaling summary")
            comp = before.merge(after, on="column", suffixes=("_before", "_after"))
            st.dataframe(comp.round(6), use_container_width=True)
            st.caption(
                "Expectation: mean≈0 and std≈1 for StandardScaler; reduced influence of outliers for RobustScaler.")

            # Quick visual: pre vs post scatter for one column
            if cols:
                c0 = cols[0]
                fig, ax = plt.subplots()
                ax.scatter(df.index, df[c0])
                ax.set_title(f"Scaled values: {c0}")
                st.pyplot(fig, clear_figure=True)

        st.session_state.cleaned_df = df

        _log_step("scaling", {
            "scaler": scaler_name,
            "columns": cols,
            "summary_post": after.round(6).to_dict(orient="records"),
        })
        st.success("Scaling applied.")


def _tab_dedup():
    st.subheader("Deduplication")
    key = st.selectbox("Primary unique key", ["license_id", "title_id"])
    tie_break = st.selectbox("Retain rule", ["Most complete", "Latest window_end"])

    if st.button("Apply deduplication"):
        before_n = len(st.session_state.cleaned_df)

        work = st.session_state.cleaned_df.copy()
        # mark duplicates by key
        dup_mask = work.duplicated(subset=[key], keep=False)
        dups = work[dup_mask]

        retained_idx = []
        dropped_idx = []
        if not dups.empty:
            for val, grp in dups.groupby(key):
                if tie_break == "Most complete":
                    comp = grp.notna().sum(axis=1)
                    idx = comp.idxmax()
                else:
                    if "window_end" in grp.columns:
                        idx = grp["window_end"].astype("datetime64[ns]", errors="ignore").idxmax()
                    else:
                        idx = grp.index[0]
                retained_idx.append(idx)
                dropped_idx.extend([i for i in grp.index if i != idx])

            work = work.drop(index=dropped_idx)

        st.session_state.cleaned_df = work
        after_n = len(st.session_state.cleaned_df)

        with st.container(border=True):
            st.markdown("Deduplication report")
            st.metric("Rows before", before_n)
            st.metric("Rows after", after_n)
            st.metric("Removed duplicates", before_n - after_n)
            if dropped_idx:
                st.write("Sample dropped indices:", dropped_idx[:25])

        _log_step("deduplication", {
            "key": key,
            "rule": tie_break,
            "removed_count": int(before_n - after_n),
            "retained_sample_idx": retained_idx[:50],
        })
        st.success("Deduplication applied.")


def _tab_validation_dashboard():
    st.subheader("Validation Dashboard")
    steps = st.session_state.get("steps", [])
    if not steps:
        st.info("No steps recorded yet.")
        return

    # Simple status extraction
    summary_rows = []
    for s in steps:
        name, payload = next(iter(s.items()))
        if name == "post_impute_check":
            ok_mean = sum(1 for r in payload["summary"] if r["mean_stable"])
            ok_median = sum(1 for r in payload["summary"] if r["median_stable"])
            summary_rows.append({"step": "Imputation", "status": f"Mean stable {ok_mean}, Median stable {ok_median}"})
        elif name == "outliers":
            summary_rows.append({"step": "Outliers", "status": f"{payload['method']} applied"})
        elif name == "encoding":
            summary_rows.append(
                {"step": "Encoding", "status": f"{payload['method']} on {len(payload['columns'])} cols"})
        elif name == "scaling":
            summary_rows.append(
                {"step": "Scaling", "status": f"{payload['scaler']} on {len(payload['summary_post'])} cols"})
        elif name == "deduplication":
            summary_rows.append({"step": "Deduplication", "status": f"Removed {payload['removed_count']} rows"})
        else:
            summary_rows.append({"step": name, "status": "Recorded"})

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)


def _save_cleaned_dataset():
    st.caption("Save the cleaned dataset and transformation manifest.")
    cleaned_out_name = "feature_master_cleaned"
    if st.button("Apply & Save Clean Feature Master", type="primary"):
        run_id = st.session_state.run_id
        out_path = save_processed(st.session_state.cleaned_df, run_id, f"{cleaned_out_name}_{run_id}")
        mf_path = save_profile(
            {"run_id": run_id, "steps": st.session_state.steps},
            run_id,
            f"manifest_{run_id}"
        )
        st.session_state["last_cleaned_feature_master_path"] = out_path

        st.success(f"Saved cleaned dataset: {out_path}")
        st.success(f"Wrote transformation manifest: {mf_path}")
        st.session_state["preprocessing_performed"] = True


def render_cleaning_section():
    LOGGER.info("Rendering Cleaning panel....")
    st.header("Preprocessing (and Cleaning)")
    df, label = load_active_feature_master_from_session()
    if df is None:
        st.warning("No feature master found. Build it in Data Staging.")
        return
    st.caption(f"Using: {label} — shape={df.shape}")

    # Set defaults
    st.session_state.setdefault("cleaned_df", df.copy())
    st.session_state.setdefault("steps", [])

    with section_panel("Preprocessing (and Cleaning)", expanded=True):
        tabs = st.tabs(["Impute", "Outliers", "Encoding", "Scaling", "Deduplication", "Save Cleaned Dataset"])
        with tabs[0]:
            begin_tab_scroll()
            _tab_impute()
            end_tab_scroll()
        with tabs[1]:
            begin_tab_scroll()
            _tab_outliers()
            end_tab_scroll()
        with tabs[2]:
            begin_tab_scroll()
            _tab_encoding()
            end_tab_scroll()
        with tabs[3]:
            begin_tab_scroll()
            _tab_scaling()
            end_tab_scroll()
        with tabs[4]:
            begin_tab_scroll()
            _tab_dedup()
            end_tab_scroll()
        with tabs[5]:
            begin_tab_scroll()
            _tab_validation_dashboard()
            _save_cleaned_dataset()
            end_tab_scroll()

    LOGGER.info("Cleaning panel rendered")
