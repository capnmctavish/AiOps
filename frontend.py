import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# ----------------------
# Mock Telecom Hierarchy for Telescopic Filters
# ----------------------
TELECOM_HIERARCHY = {
    "Consumer": {
        "Wireless": {
            "North America": ["5G Service", "4G LTE", "Fixed Wireless"],
            "EMEA": ["5G Service", "4G LTE"],
            "APAC": ["5G Service", "4G LTE", "NB-IoT"],
            "LATAM": ["5G Service", "4G LTE"]
        },
        "Broadband": {
            "North America": ["Fiber Internet", "DSL"],
            "EMEA": ["Fiber Internet"],
            "APAC": ["Fiber Internet", "DSL"],
            "LATAM": ["DSL"]
        }
    },
    "Enterprise": {
        "Managed Services": {
            "North America": ["VPN Solutions", "SASE", "SD-WAN"],
            "EMEA": ["VPN Solutions", "SD-WAN"],
            "APAC": ["SD-WAN", "IoT Platform"],
            "LATAM": ["VPN Solutions"]
        },
        "Cloud & Hosting": {
            "North America": ["Cloud Hosting", "Bare Metal Servers"],
            "EMEA": ["Cloud Hosting"],
            "APAC": ["Cloud Hosting", "Bare Metal Servers"],
            "LATAM": ["Cloud Hosting"]
        }
    },
    "Wholesale": {
        "Carrier Services": {
            "North America": ["Long-Haul Transport", "Colocation"],
            "EMEA": ["Long-Haul Transport"],
            "APAC": ["Submarine Cables"],
            "LATAM": ["Long-Haul Transport"]
        }
    }
}

NETWORK_TYPES = ["5G RAN", "4G LTE", "Fiber", "DSL", "Cable", "IoT Network"]
ENVIRONMENTS = ["Production", "Pre-Production", "Test", "Sandbox"]
TIME_RANGE_OPTIONS = ["Last 6 hrs", "Last 12 hrs", "Last 24 hrs", "Last 48 hrs"]
TIME_RANGE_TO_HOURS = {
    "Last 6 hrs": 6,
    "Last 12 hrs": 12,
    "Last 24 hrs": 24,
    "Last 48 hrs": 48
}

def get_divisions():
    return list(TELECOM_HIERARCHY.keys())

def get_business_units(selected_division):
    return list(TELECOM_HIERARCHY[selected_division].keys())

def get_regions(selected_division, selected_bu):
    return list(TELECOM_HIERARCHY[selected_division][selected_bu].keys())

def get_services(selected_division, selected_bu, selected_region):
    return TELECOM_HIERARCHY[selected_division][selected_bu][selected_region]

# ----------------------
# Mock Data Generators
# ----------------------
def generate_infrastructure_data(num_points=24):
    now = datetime.now()
    times = [now - timedelta(hours=i) for i in reversed(range(num_points))]
    cpu_usage = np.random.randint(10, 95, size=num_points)
    mem_usage = np.random.randint(20, 90, size=num_points)
    network_in = np.random.randint(100, 2000, size=num_points)
    network_out = np.random.randint(100, 2000, size=num_points)
    disk_io = np.random.randint(10, 1000, size=num_points)
    df = pd.DataFrame({
        'timestamp': times,
        'CPU %': cpu_usage,
        'Memory %': mem_usage,
        'Network IN (MB)': network_in,
        'Network OUT (MB)': network_out,
        'Disk I/O (IOPS)': disk_io
    })
    return df

def generate_incidents_data():
    mock_incidents = []
    for i in range(1, 10):
        severity = random.choice(["P1", "P2", "P3"])
        is_machine_identified = random.choice([True, False])
        status = random.choice(["Open", "Resolved"])
        desc = f"Mock incident {i} - {random.choice(['High CPU', 'Network Latency', 'Memory Leak', 'Disk Errors'])}"
        mock_incidents.append({
            "id": 1000 + i,
            "severity": severity,
            "machine_identified": is_machine_identified,
            "status": status,
            "description": desc
        })
    df = pd.DataFrame(mock_incidents)
    return df

def generate_security_data():
    critical_events = random.randint(0, 5)
    high_events = random.randint(5, 15)
    medium_events = random.randint(5, 25)
    compliance_score = random.randint(80, 95)
    data = {
        "Critical Events": critical_events,
        "High Events": high_events,
        "Medium Events": medium_events,
        "Compliance Score (%)": compliance_score
    }
    return data

def generate_cost_data():
    current_spend = random.randint(50000, 90000)
    budget = 80000
    anomalies = random.choice([True, False])
    return current_spend, budget, anomalies

# ----------------------
# Basic ML Functions
# ----------------------
def detect_anomalies(df):
    numeric_cols = ['CPU %', 'Memory %', 'Network IN (MB)', 'Network OUT (MB)', 'Disk I/O (IOPS)']
    data = df[numeric_cols].values
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(data)
    scores = iso_forest.decision_function(data)
    outliers = iso_forest.predict(data)
    df['anomaly_score'] = scores
    df['is_anomaly'] = (outliers == -1)
    return df

def cluster_usage_patterns(df):
    numeric_cols = ['CPU %', 'Memory %', 'Network IN (MB)', 'Network OUT (MB)', 'Disk I/O (IOPS)']
    data = df[numeric_cols].values
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    kmeans.fit(data)
    clusters = kmeans.labels_
    df['usage_cluster'] = clusters
    return df

def generate_predicted_data(df):
    df_sorted = df.sort_values(by='timestamp')
    df_sorted['hour_index'] = range(len(df_sorted))
    X = df_sorted[['hour_index']]
    y = df_sorted['CPU %']
    model = LinearRegression()
    model.fit(X, y)
    future_hours = 6
    last_index = df_sorted['hour_index'].max()
    future_indices = np.arange(last_index+1, last_index+1+future_hours)
    y_pred = model.predict(future_indices.reshape(-1, 1))
    future_ts = [df_sorted['timestamp'].iloc[-1] + timedelta(hours=i+1) for i in range(future_hours)]
    pred_df = pd.DataFrame({
        'timestamp': future_ts,
        'Predicted CPU %': y_pred
    })
    return pred_df

def main():
    st.set_page_config(page_title="Telco AIOps with ML", layout="wide")
    st.title("Telco AIOps Dashboard with ML (Telescopic Filters, Anomalies, Clustering, Prediction)")

    # ----------------------
    # Sidebar: Telco Filters + ML Toggles
    # ----------------------
    st.sidebar.title("Telco Scoping & Settings")
    divisions = get_divisions()
    selected_division = st.sidebar.selectbox("Select Division", divisions)
    bus_units = get_business_units(selected_division)
    selected_bu = st.sidebar.selectbox("Business Unit", bus_units)
    regions = get_regions(selected_division, selected_bu)
    selected_region = st.sidebar.selectbox("Region", regions)
    services = get_services(selected_division, selected_bu, selected_region)
    selected_service = st.sidebar.selectbox("Service", services)
    selected_net_type = st.sidebar.selectbox("Network Type", NETWORK_TYPES)
    selected_env = st.sidebar.selectbox("Environment", ENVIRONMENTS)
    selected_time_range = st.sidebar.selectbox("Time Range", TIME_RANGE_OPTIONS)
    hours_to_fetch = TIME_RANGE_TO_HOURS[selected_time_range]
    st.sidebar.markdown("---")
    show_anomalies = st.sidebar.checkbox("Show Anomalies (IsolationForest)", value=True)
    show_clusters = st.sidebar.checkbox("Show Clusters (KMeans)", value=True)
    show_prediction = st.sidebar.checkbox("Show Predictive CPU Trend", value=True)

    st.write(f"**Scope:** {selected_division} > {selected_bu} > {selected_region} > {selected_service} | **Network:** {selected_net_type} | **Env:** {selected_env} | **Time Range:** {selected_time_range}")

    st.write(f"**ML Toggles:** Anomalies={show_anomalies}, Clusters={show_clusters}, Prediction={show_prediction}")

    # ----------------------
    # Generate Mock Data
    # ----------------------
    df_infra = generate_infrastructure_data(num_points=hours_to_fetch)
    df_incidents = generate_incidents_data()
    security_data = generate_security_data()
    cost_spend, budget, cost_anomaly = generate_cost_data()

    # ----------------------
    # Apply ML if toggled
    # ----------------------
    if show_anomalies:
        df_infra = detect_anomalies(df_infra)
    if show_clusters:
        df_infra = cluster_usage_patterns(df_infra)
    if show_prediction:
        pred_df = generate_predicted_data(df_infra)

    # ----------------------
    # KPI / Alerts section
    # ----------------------
    st.subheader("Incidents & Alerts")
    open_incidents = df_incidents[df_incidents['status'] == "Open"]
    p1_incidents = open_incidents[open_incidents['severity'] == "P1"]
    col_kpi1, col_kpi2 = st.columns(2)
    with col_kpi1:
        st.metric(label="Total Open Incidents", value=len(open_incidents))
    with col_kpi2:
        st.metric(label="P1 Incidents", value=len(p1_incidents))
    with st.expander("View All Incidents"):
        st.dataframe(df_incidents)

    # ----------------------
    # Main Layout: 3 columns
    # ----------------------
    left_col, center_col, right_col = st.columns([1, 2, 1])
    # Left: Real-Time Alerts / Potential
    with left_col:
        st.subheader("Alerts Feed (Sample)")
        alerts_feed = [
            "Critical: 5G Cell Overload in North America",
            "Major: Fiber Cut - EMEA Route",
            "Warning: SD-WAN Latency Spike in APAC",
            "Info: IoT Sensor Surge in LATAM"
        ]
        for alert in alerts_feed:
            st.write(f"- {alert}")
        st.subheader("Security & Compliance")
        sec_df = pd.DataFrame([security_data])
        st.table(sec_df)
        st.subheader("Cost & Resource Efficiency")
        if cost_anomaly:
            st.warning("Cost anomaly detected! Investigate usage spikes or potential misconfigurations.")
        else:
            st.success("No current cost anomalies detected.")
        st.write("Underutilized Resources (Mock)")
        underutilized = [
            {"Resource": "5G-Cell-NY-001", "CPU%": 12, "Memory%": 25},
            {"Resource": "SD-WAN-APAC-Node3", "CPU%": 10, "Memory%": 20},
        ]
        st.table(pd.DataFrame(underutilized))

    # Center: Charts for Infra Metrics
    with center_col:
        st.subheader("Infrastructure Metrics Over Time")
        df_infra_sorted = df_infra.sort_values(by='timestamp')
        st.line_chart(
            data=df_infra_sorted.set_index('timestamp')[['CPU %', 'Memory %', 'Disk I/O (IOPS)']],
            height=300
        )
        st.write("**Network Throughput (MB)**")
        st.line_chart(
            data=df_infra_sorted.set_index('timestamp')[['Network IN (MB)', 'Network OUT (MB)']],
            height=200
        )
        # Anomalies
        if show_anomalies:
            st.subheader("Detected Anomalies")
            anom_df = df_infra_sorted[df_infra_sorted['is_anomaly'] == True]
            if not anom_df.empty:
                st.write("Below rows flagged as anomalies by IsolationForest:")
                st.table(anom_df[['timestamp','CPU %','Memory %','Network IN (MB)','Network OUT (MB)','Disk I/O (IOPS)','anomaly_score']])
            else:
                st.success("No anomalies detected in this time range.")
        # Clusters
        if show_clusters:
            st.subheader("Usage Clusters")
            cluster_map = {0: "Low Usage Pattern", 1: "Medium Usage Pattern", 2: "High Usage Pattern"}
            if 'usage_cluster' in df_infra_sorted.columns:
                cluster_counts = df_infra_sorted['usage_cluster'].value_counts().sort_index()
                st.write("**Cluster distribution:**")
                for c in cluster_counts.index:
                    st.write(f"- Cluster {c} ({cluster_map.get(c, 'Unknown')}): {cluster_counts[c]} data points")
                st.write("**Sample data with cluster labels:**")
                st.write(df_infra_sorted[['timestamp','CPU %','Memory %','usage_cluster']].head(10))

    # Right: Predictive Analysis
    with right_col:
        if show_prediction:
            st.subheader("Predictive CPU Usage (Next 6 Hours)")
            st.write("A simple linear regression model on CPU usage over time.")
            hist_df = pd.DataFrame({
                'timestamp': df_infra_sorted['timestamp'],
                'CPU %': df_infra_sorted['CPU %']
            })
            hist_df['Predicted CPU %'] = np.nan
            merged_df = hist_df.set_index('timestamp')
            future_df = pred_df.set_index('timestamp') if 'pred_df' in locals() else None
            if future_df is not None:
                merged_df = pd.concat([merged_df, future_df], axis=1)
            st.line_chart(merged_df[['CPU %','Predicted CPU %']], height=300)

    st.info("This combined dashboard demonstrates telescopic filtering (org > BU > region > service) along with basic AIOps ML: anomaly detection (IsolationForest), clustering (KMeans), and prediction (linear regression). Replace mock data with real metrics and refine models for production-level analytics.")

if __name__ == "__main__":
    main()
