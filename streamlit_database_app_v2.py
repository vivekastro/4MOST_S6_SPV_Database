import sqlite3
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from astropy.coordinates import SkyCoord
import astropy.units as u
import subprocess
from pathlib import Path
import streamlit as st

import gdown

from pathlib import Path
import streamlit as st
import gdown

DB_FILE = Path("spectrum_fibsuccess_headers.sqlite3")

GDRIVE_FILE_ID = "1-A66x7YfubrS6yDPNBcJrsEPZRWGQtvT"
DB_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"


def ensure_database():
    if DB_FILE.exists() and DB_FILE.stat().st_size > 200 * 1024 * 1024:
        return

    st.info("Downloading SQLite database from Google Drive...")

    output = gdown.download(
        DB_URL,
        str(DB_FILE),
        quiet=False,
    )

    if output is None:
        st.error("gdown returned None. Check Google Drive sharing permissions.")
        st.stop()

    if not DB_FILE.exists() or DB_FILE.stat().st_size < 200 * 1024 * 1024:
        st.error(
            f"Database download failed or incomplete. "
            f"Size = {DB_FILE.stat().st_size / 1024**2:.2f} MB"
            if DB_FILE.exists()
            else "Database file was not created."
        )
        st.stop()


ensure_database()




print("Database exists:", DB_FILE.exists())

if DB_FILE.exists():
    print("Database size (MB):", DB_FILE.stat().st_size / 1024**2)
# ============================================================
# CONFIG
# ============================================================


VALID_FIELDS = [
    "eFEDS_1", "eFEDS_2", "eFEDS_3", "eFEDS_4", "eFEDS_5", "eFEDS_6",
    "no_eFEDS_4", "no_eFEDS_5", "no_eFEDS_8", "no_eFEDS_9",
    "no_eFEDS_10", "no_eFEDS_11", "no_eFEDS_17", "no_eFEDS_22",
]

MATCH_RADIUS_ARCSEC = 1.0
MIN_EXPTIME_SEC = 1200.0


# ============================================================
# STREAMLIT SETUP
# ============================================================

st.set_page_config(
    page_title="4MOST S6 SPV SQL Explorer",
    layout="wide",
)

st.title("4MOST S6 Survey - SPV SQL Explorer")

if not DB_FILE.exists():
    st.error(f"SQLite file not found: `{DB_FILE}`")
    st.stop()

last_updated = datetime.fromtimestamp(DB_FILE.stat().st_mtime)

st.caption(
    f"Database: `{DB_FILE}`  |  "
    f"Last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S')}"
)


# ============================================================
# SESSION STATE
# ============================================================

if "query_result" not in st.session_state:
    st.session_state.query_result = None

if "last_query" not in st.session_state:
    st.session_state.last_query = None


# ============================================================
# ABOUT DATABASE
# ============================================================

with st.expander("About this database and tables", expanded=False):

    st.markdown(
        """
        This SQLite database combines 4MOST SPV spectrum-level information,
        FITS primary-header metadata, fibre-success diagnostics, and the refined
        target catalog.

        | Table | One row represents | Purpose |
        |---|---|---|
        | `merged_spectra` | one observed spectrum / FITS file | Main science-ready table combining header, fibre-success, and catalog information |
        | `primary_headers` | one FITS spectrum | Metadata read directly from the `[0]` header of each science FITS file |
        | `fib_success` | one observed spectrum | Fibre-positioning diagnostics including `FIB_SUCCESS` and `DELTA_ARCSEC` |
        | `refined_catalog` | one unique target/source | Parent target catalog with redshift, magnitudes, selection stage, and field ID |

        `merged_spectra` is usually the main table to query.

        ```text
        refined_catalog   = unique input targets
        primary_headers   = FITS header metadata per spectrum
        fib_success       = fibre success diagnostics per spectrum
        merged_spectra    = combined exposure-level master table
        ```

        Important:

        ```text
        one source  →  many spectra / exposures
        ```
        ```text
        Database maintained by Vivek(getkeviv@gmail.com/vivek.m@iiap.res.in). Please contact for additional feature requests. 
        ```
        """
    )


# ============================================================
# DATABASE HELPERS
# ============================================================

@st.cache_data
def run_query(query):
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql_query(query, conn)


@st.cache_data
def get_table_names():
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            ORDER BY name
            """,
            conn,
        )


@st.cache_data
def get_table_preview(table):
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql_query(
            f"SELECT * FROM {table} LIMIT 5",
            conn,
        )


@st.cache_data
def get_table_columns(table):
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql_query(
            f"PRAGMA table_info({table})",
            conn,
        )


# ============================================================
# SORT HELPERS
# ============================================================

def field_sort_key(field):
    field = str(field)

    if field.startswith("eFEDS_"):
        return (0, int(field.split("_")[-1]))

    if field.startswith("no_eFEDS_"):
        return (1, int(field.split("_")[-1]))

    return (99, 999)


# ============================================================
# RA/DEC SOURCE MATCHING
# ============================================================

def crossmatch_sources_by_radec(df, match_radius_arcsec=1.0):
    matched_rows = []

    for field_id, g in df.groupby("field_id"):

        g = g.reset_index(drop=True)

        if len(g) == 0:
            continue

        coords = SkyCoord(
            ra=g["ra"].values * u.deg,
            dec=g["dec"].values * u.deg,
        )

        used = np.zeros(len(g), dtype=bool)

        for i in range(len(g)):

            if used[i]:
                continue

            sep = coords[i].separation(coords).arcsec
            members = sep < match_radius_arcsec

            used[members] = True

            sub = g.loc[members]

            matched_rows.append(
                {
                    "field_id": field_id,
                    "source_ra_mean": sub["ra"].mean(),
                    "source_dec_mean": sub["dec"].mean(),
                    "n_spectra": len(sub),
                    "filenames": ";".join(sub["filename"].astype(str)),
                    "objects": ";".join(sub["object"].astype(str).unique()),
                }
            )

    return pd.DataFrame(matched_rows)


@st.cache_data
def make_pivot(success_only=False, min_exptime_sec=1200.0, match_radius_arcsec=1.0):

    query = """
    SELECT
        field_id,
        filename,
        object,
        ra,
        dec,
        fib_success,
        hdr_EXPTIME
    FROM merged_spectra
    WHERE field_id IS NOT NULL
      AND ra IS NOT NULL
      AND dec IS NOT NULL
      AND hdr_EXPTIME IS NOT NULL
      AND CAST(hdr_EXPTIME AS FLOAT) >= ?
    """

    if success_only:
        query += " AND CAST(fib_success AS INTEGER) = 1"

    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query(query, conn, params=(float(min_exptime_sec),))

    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    df["field_id"] = df["field_id"].astype(str).str.strip()
    df = df[df["field_id"].isin(VALID_FIELDS)].copy()

    df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
    df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
    df["fib_success"] = pd.to_numeric(df["fib_success"], errors="coerce").fillna(0).astype(int)

    df = df.dropna(subset=["ra", "dec"]).copy()

    df = df.drop_duplicates(
        subset=["field_id", "filename", "ra", "dec", "fib_success"]
    )

    source_counts = crossmatch_sources_by_radec(
        df,
        match_radius_arcsec=match_radius_arcsec,
    )

    if len(source_counts) == 0:
        return pd.DataFrame(), pd.DataFrame()

    source_counts = source_counts.sort_values(
        ["field_id", "n_spectra", "source_ra_mean"],
        ascending=[True, False, True],
    )

    summary = (
        source_counts
        .groupby(["field_id", "n_spectra"])
        .size()
        .reset_index(name="n_sources")
    )

    pivot = summary.pivot(
        index="field_id",
        columns="n_spectra",
        values="n_sources",
    ).fillna(0).astype(int)

    pivot = pivot.loc[sorted(pivot.index, key=field_sort_key)]

    pivot["TOTAL_UNIQUE_SOURCES"] = pivot.sum(axis=1)
    pivot.loc["ALL_FIELDS"] = pivot.sum(axis=0)

    return pivot, source_counts


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Database tables")

tables = get_table_names()

if len(tables) > 0:

    st.sidebar.dataframe(tables, use_container_width=True)

    selected_table = st.sidebar.selectbox(
        "Preview table",
        tables["name"].tolist(),
    )

    st.sidebar.write(f"Preview: `{selected_table}`")

    st.sidebar.dataframe(
        get_table_preview(selected_table),
        use_container_width=True,
    )

    with st.sidebar.expander("Columns"):
        st.dataframe(
            get_table_columns(selected_table),
            use_container_width=True,
        )


# ============================================================
# SQL FRONTEND
# ============================================================

example_queries = {
    "Show first 100 rows": """
SELECT *
FROM merged_spectra
LIMIT 100;
""",

    "Successful spectra only": """
SELECT filename, object, ra, dec, field_id,
       fib_success, delta_arcsec, redshift, rmag, gmag, hdr_EXPTIME
FROM merged_spectra
WHERE fib_success = 1
LIMIT 200;
""",

    "Count by field and FIB_SUCCESS": """
SELECT field_id, fib_success, COUNT(*) AS n_spectra
FROM merged_spectra
GROUP BY field_id, fib_success
ORDER BY field_id, fib_success;
""",

    "DELTA_ARCSEC vs FIB_SUCCESS": """
SELECT field_id, filename, object, fib_success, delta_arcsec
FROM merged_spectra
WHERE delta_arcsec IS NOT NULL;
""",

"SNR vs rmag": """
SELECT
    object,
    field_id,

    AVG(rmag) AS mean_rmag,

    SQRT(
        SUM(
            CAST(hdr_SNR AS FLOAT) *
            CAST(hdr_SNR AS FLOAT)
        )
    ) AS summed_hdr_SNR,

    COUNT(*) AS n_spectra

FROM merged_spectra

WHERE rmag IS NOT NULL
  AND hdr_SNR IS NOT NULL
  AND (
        CAST(fib_success AS INTEGER) = 1
        OR (
            CAST(fib_success AS INTEGER) = 0
            AND CAST(delta_arcsec AS FLOAT) < 1
        )
      )

GROUP BY object, field_id

-- HAVING COUNT(*) = 6

ORDER BY field_id, mean_rmag;
""",    

    "Redshift vs magnitude": """
SELECT field_id, filename, object, redshift, rmag, rmag_err, gmag, gmag_err,
       fib_success
FROM merged_spectra
WHERE redshift IS NOT NULL
  AND rmag IS NOT NULL;
""",
"eFEDS members":"""
SELECT *
FROM merged_spectra
WHERE CAST(efeds_member AS INTEGER) = 1;
-- change efeds_member to erass_member or unwise_member
""",

    "Check repeats by rounded RA/DEC": """
SELECT field_id,
       ROUND(ra, 5) AS ra5,
       ROUND(dec, 5) AS dec5,
       COUNT(*) AS n
FROM merged_spectra
WHERE CAST(hdr_EXPTIME AS FLOAT) >= 1200
GROUP BY field_id, ra5, dec5
ORDER BY n DESC
LIMIT 100;
""",
}

st.subheader("SQL query frontend")

choice = st.selectbox(
    "Load example query",
    ["Custom query"] + list(example_queries.keys()),
)

default_query = example_queries.get(
    choice,
    """
SELECT *
FROM merged_spectra
LIMIT 100;
""",
)

query = st.text_area(
    "Enter SQL query",
    value=default_query,
    height=220,
)

if st.button("Run query", type="primary"):

    try:
        result = run_query(query)

        st.session_state.query_result = result
        st.session_state.last_query = query

    except Exception as e:
        st.error(f"SQL error: {e}")
        st.session_state.query_result = None


result = st.session_state.query_result

if result is not None:

    st.success(f"Returned {len(result)} rows")

    with st.expander("Show last executed SQL query", expanded=False):
        st.code(st.session_state.last_query, language="sql")

    st.dataframe(result, use_container_width=True)

    st.download_button(
        label="Download result as CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="query_result.csv",
        mime="text/csv",
    )

    # ====================================================
    # PLOT QUERY RESULT
    # ====================================================

    st.divider()
    st.subheader("Plot query result")

    numeric_cols = result.select_dtypes(include="number").columns.tolist()
    all_cols = result.columns.tolist()

    if len(result) == 0:
        st.info("No rows to plot.")

    elif len(numeric_cols) == 0:
        st.info("No numeric columns available for plotting.")

    else:
        plot_type = st.radio(
            "Plot type",
            ["Histogram", "Scatter"],
            horizontal=True,
            key="plot_type",
        )

        if plot_type == "Histogram":

            hist_col = st.selectbox(
                "Column for histogram",
                numeric_cols,
                key="hist_col",
            )

            nbins = st.slider(
                "Number of bins",
                min_value=5,
                max_value=150,
                value=40,
                step=5,
                key="hist_nbins",
            )

            color_col = st.selectbox(
                "Optional color/group column",
                ["None"] + all_cols,
                key="hist_color_col",
            )

            marginal = st.selectbox(
                "Marginal plot",
                ["None", "box", "rug", "violin"],
                key="hist_marginal",
            )

            marginal_arg = None if marginal == "None" else marginal

            if color_col == "None":
                fig = px.histogram(
                    result,
                    x=hist_col,
                    nbins=nbins,
                    marginal=marginal_arg,
                )
            else:
                fig = px.histogram(
                    result,
                    x=hist_col,
                    color=color_col,
                    nbins=nbins,
                    marginal=marginal_arg,
                )

            fig.update_layout(
                title=f"Histogram of {hist_col}",
                xaxis_title=hist_col,
                yaxis_title="Count",
                template="plotly_white",
            )

            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Scatter":

            x_col = st.selectbox(
                "X column",
                numeric_cols,
                index=0,
                key="scatter_x_col",
            )

            y_col = st.selectbox(
                "Y column",
                numeric_cols,
                index=min(1, len(numeric_cols) - 1),
                key="scatter_y_col",
            )

            yerr_col = st.selectbox(
                "Y error column",
                ["None"] + numeric_cols,
                key="scatter_yerr_col",
            )

            color_col = st.selectbox(
                "Optional color column",
                ["None"] + all_cols,
                key="scatter_color_col",
            )

            symbol_col = st.selectbox(
                "Optional marker-symbol column",
                ["None"] + all_cols,
                key="scatter_symbol_col",
            )

            size_col = st.selectbox(
                "Optional marker-size column",
                ["None"] + numeric_cols,
                key="scatter_size_col",
            )

            hover_cols = st.multiselect(
                "Hover columns",
                all_cols,
                default=[
                    c for c in [
                        "filename",
                        "object",
                        "field_id",
                        "fib_success",
                        "delta_arcsec",
                    ]
                    if c in all_cols
                ],
                key="scatter_hover_cols",
            )

            log_x = st.checkbox("Log X axis", value=False, key="scatter_log_x")
            log_y = st.checkbox("Log Y axis", value=False, key="scatter_log_y")

            plot_df = result.dropna(subset=[x_col, y_col]).copy()

            if yerr_col != "None":
                plot_df = plot_df.dropna(subset=[yerr_col])

            kwargs = {
                "data_frame": plot_df,
                "x": x_col,
                "y": y_col,
                "hover_data": hover_cols,
                "log_x": log_x,
                "log_y": log_y,
            }

            if color_col != "None":
                kwargs["color"] = color_col

            if symbol_col != "None":
                kwargs["symbol"] = symbol_col

            if size_col != "None":
                kwargs["size"] = size_col

            fig = px.scatter(**kwargs)

            if yerr_col != "None":
                fig.update_traces(
                    error_y=dict(
                        type="data",
                        array=plot_df[yerr_col],
                        visible=True,
                    )
                )

            fig.update_layout(
                title=f"{y_col} vs {x_col}",
                xaxis_title=x_col,
                yaxis_title=y_col,
                template="plotly_white",
            )

            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PIVOT TABLES
# ============================================================

st.divider()

st.subheader("Pivot tables: number of RA/DEC-matched sources with N spectra")

col1, col2 = st.columns(2)

with col1:
    min_exptime_sec = st.number_input(
        "Minimum EXPTIME in seconds",
        min_value=0.0,
        value=MIN_EXPTIME_SEC,
        step=100.0,
    )

with col2:
    match_radius_arcsec = st.number_input(
        "RA/DEC source match radius [arcsec]",
        min_value=0.1,
        value=MATCH_RADIUS_ARCSEC,
        step=0.1,
    )

tab1, tab2 = st.tabs(
    [
        "All spectra",
        "FIB_SUCCESS = 1 only",
    ]
)

with tab1:

    pivot_all, sources_all = make_pivot(
        success_only=False,
        min_exptime_sec=min_exptime_sec,
        match_radius_arcsec=match_radius_arcsec,
    )

    st.write(
        "Number of unique RA/DEC-matched sources with N spectra, "
        "using all spectra."
    )

    st.dataframe(pivot_all, use_container_width=True)

    if len(pivot_all) > 0:
        st.download_button(
            "Download all-spectra pivot CSV",
            pivot_all.to_csv().encode("utf-8"),
            file_name="pivot_sources_vs_nspectra_all_radec.csv",
            mime="text/csv",
        )

    if len(sources_all) > 0:
        st.download_button(
            "Download all-spectra source counts CSV",
            sources_all.to_csv(index=False).encode("utf-8"),
            file_name="source_counts_all_radec.csv",
            mime="text/csv",
        )

with tab2:

    pivot_success, sources_success = make_pivot(
        success_only=True,
        min_exptime_sec=min_exptime_sec,
        match_radius_arcsec=match_radius_arcsec,
    )

    st.write(
        "Number of unique RA/DEC-matched sources with N successful spectra only."
    )

    st.dataframe(pivot_success, use_container_width=True)

    if len(pivot_success) > 0:
        st.download_button(
            "Download successful-spectra pivot CSV",
            pivot_success.to_csv().encode("utf-8"),
            file_name="pivot_sources_vs_nspectra_fibsuccess1_radec.csv",
            mime="text/csv",
        )

    if len(sources_success) > 0:
        st.download_button(
            "Download successful-spectra source counts CSV",
            sources_success.to_csv(index=False).encode("utf-8"),
            file_name="source_counts_fibsuccess1_radec.csv",
            mime="text/csv",
        )


# ============================================================
# QUICK SUMMARY
# ============================================================

st.divider()

st.subheader("Quick database summary")

summary_query = """
SELECT
    COUNT(*) AS total_spectra,
    SUM(CASE WHEN CAST(fib_success AS INTEGER) = 1 THEN 1 ELSE 0 END) AS successful_spectra,
    SUM(CASE WHEN CAST(fib_success AS INTEGER) = 0 THEN 1 ELSE 0 END) AS failed_spectra,
    COUNT(DISTINCT filename) AS unique_filenames
FROM merged_spectra;
"""

summary = run_query(summary_query)

st.dataframe(summary, use_container_width=True)
