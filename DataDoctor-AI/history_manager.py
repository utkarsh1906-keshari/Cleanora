"""
DataDoctor AI - Dataset Health History Module (Feature 6)
Persistent SQLite-backed audit history tracking, health score progress timelines,
and cross-session quality comparisons with zero PII retention.
"""

import os
import sqlite3
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DB_PATH = os.path.join(os.path.dirname(__file__), "health_history.db")


def init_history_db(db_path=DB_PATH):
    """
    Initializes SQLite table for audit sessions if it does not exist.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS health_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            workspace TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            row_count INTEGER,
            col_count INTEGER,
            health_score REAL,
            missing_count INTEGER,
            duplicate_count INTEGER,
            outlier_count INTEGER,
            anomaly_count INTEGER,
            cleaning_applied TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_health_session(
    workspace,
    dataset_name,
    row_count,
    col_count,
    health_score,
    missing_count,
    duplicate_count,
    outlier_count,
    anomaly_count=0,
    cleaning_applied="None",
    db_path=DB_PATH
):
    """
    Saves an audit session record to SQLite without storing any raw or sensitive data.
    """
    init_history_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("""
        INSERT INTO health_sessions (
            timestamp, workspace, dataset_name, row_count, col_count,
            health_score, missing_count, duplicate_count, outlier_count,
            anomaly_count, cleaning_applied
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        str(workspace),
        str(dataset_name),
        int(row_count),
        int(col_count),
        round(float(health_score), 1),
        int(missing_count),
        int(duplicate_count),
        int(outlier_count),
        int(anomaly_count),
        str(cleaning_applied)
    ))
    
    conn.commit()
    conn.close()
    return True


def get_health_history(workspace=None, db_path=DB_PATH):
    """
    Retrieves historical session records as a pandas DataFrame.
    """
    init_history_db(db_path)
    conn = sqlite3.connect(db_path)
    
    if workspace and workspace != "All Workspaces":
        query = "SELECT * FROM health_sessions WHERE workspace = ? ORDER BY id DESC"
        df_hist = pd.read_sql_query(query, conn, params=(workspace,))
    else:
        query = "SELECT * FROM health_sessions ORDER BY id DESC"
        df_hist = pd.read_sql_query(query, conn)
        
    conn.close()
    return df_hist


def clear_health_history(db_path=DB_PATH):
    """
    Clears all saved audit records from SQLite.
    """
    init_history_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM health_sessions")
    conn.commit()
    conn.close()
    return True


def create_history_trend_chart(df_hist):
    """
    Builds a Plotly line chart tracking Health Score evolution across sessions.
    """
    if df_hist is None or df_hist.empty:
        return None

    # Chronological ordering for plotting
    plot_df = df_hist.sort_values("id").copy()
    plot_df["Session Label"] = [f"#{row.id}: {row.workspace}" for _, row in plot_df.iterrows()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["Session Label"],
        y=plot_df["health_score"],
        mode="lines+markers+text",
        text=plot_df["health_score"].astype(str) + "/100",
        textposition="top center",
        line=dict(color="#38BDF8", width=3, shape="spline"),
        marker=dict(size=10, color="#34D399", line=dict(width=2, color="#0F172A")),
        name="Health Score"
    ))

    # Add 80/100 benchmark reference line
    fig.add_hline(
        y=80,
        line_dash="dot",
        line_color="#34D399",
        annotation_text="Optimal Health Benchmark (80/100)",
        annotation_position="bottom right"
    )

    fig.update_layout(
        title="Dataset Health Score Evolution Across Audits",
        xaxis_title="Audit Session",
        yaxis_title="Health Score (/100)",
        yaxis=dict(range=[0, 110]),
        paper_bgcolor="#0B1120",
        plot_bgcolor="#16243A",
        font_color="#E2E8F0"
    )

    return fig
