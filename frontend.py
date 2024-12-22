import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# ----------------------
# Custom CSS for Dark Theme
# ----------------------
# You could tweak colors, fonts, etc. to match your brand or the screenshot style.
DARK_THEME_CSS = """
<style>
body {
    background-color: #222222;
    color: #f0f0f0;
}
.sidebar .sidebar-content {
    background-color: #2e2e2e;
}
.reportview-container .main .block-container {
    color: #f0f0f0;
    background-color: #222222;
}
h1, h2, h3, h4, h5, h6 {
    color: #ffffff;
}
.stMetric, .st-table, .stDataFrameContainer {
    background-color: #2d2d2d;
}
div.stButton > button:first-child {
    background-color: #444444;
    color: #f0f0f0;
    border: none;
}
</style>
"""

# Mock Telecom Hierarchy + Potential GenAI Divisions/Projects
TELECOM_HIERARCHY = {
    "Consumer": {
        "Wireless": {
            "North America": ["5G GenAI Chatbot Service", "4G Recommendation Engine", "Fixed Wireless Edge ML"],
            "EMEA": ["5G GenAI Chatbot Service", "4G LTE"],
            "APAC": ["5G RAN + LLM", "Edge GPU Inference"],
            "LATAM": ["4G Predictive Text", "NB-IoT ML"]
        },
        "Broadband": {
            "North America": ["Fiber Internet HPC Cluster", "DSL + Chatbot Edge"],
            "EMEA": ["Fiber HPC Shared"],
            "APAC": ["Fiber HPC Shared", "DSL + AI Pipeline"],
            "LATAM": ["DSL + GPU Node"]
        }
    },
    "Enterprise": {
        "Cloud & Hosting": {
            "North America": ["Cloud Hosting + LLM Deployment", "Bare Metal GPU Servers"],
            "EMEA": ["Cloud Hosting + HPC"],
            "APAC": ["Cloud Hosting + GPU HPC", "Bare Metal A100 Cluster"],
            "LATAM": ["Cloud Hosting + GPU HPC"]
        },
        "Managed Services": {
            "North America": ["VPN + LLM Access", "SASE + AI Monitoring", "SD-WAN + ML Ops"],
            "EMEA": ["VPN Solutions", "SD-WAN + HPC Edge"],
            "APAC": ["SD-WAN + AI Inference", "IoT Platform + GPT"],
            "LATAM": ["VPN Solutions"]
        }
    },
    "Wholesale": {
        "Carrier Services": {
            "North America": ["Long-Haul Transport HPC", "Colocation HPC AI Labs"],
            "EMEA": ["Long-Haul Transport + HPC AI"],
            "APAC": ["Submarine Cables + GPU Hosting"],
            "LATAM": ["Long-Haul Transport + LLM Models"]
        }
    }
}

NETWORK_TYPES = ["5G RAN", "4G LTE", "Fiber", "DSL", "Cable", "IoT Network", "Kubernetes/GKE", "Serverless (Cloud Run)", "Function Apps", "HPC Cluster"]
ENVIRONMENTS = ["Production", "Pre-Production", "Test", "Sandbox"]
TIME_RANGE_OPTIONS = ["Last 5 min", "Last 15 min", "Last 1 hr", "Last 24 hr"]
TIME_RANGE_MAP = {
    "Last 5 min": 5,
    "Last 15 min": 15,
    "Last 1 hr": 60,
    "Last 24 hr": 1440
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
def generate_infrastructure_data(num_points=10):
    now = datetime.now()
    # We'll assume one data point per minute for the range we have.
    times = [now - timedelta(minutes=i) for i in reversed(range(num_points))]
    cpu_usage = np.random.randint(5, 95, size=num_points)
    mem_usage = np.random.randint(10, 95, size=num_points)
    disk_io = np.random.randint(100, 3000, size=num_points)
    # For GPU usage, relevant to GenAI
    gpu_usage = np.random.randint(0, 100, size=num_points)
    network_in = np.random.randint(100, 2000, size=num_points)
    network_out = np.random.randint(100, 2000, size=num_points)
    df = pd.DataFrame({
        'timestamp': times,
        'CPU %': cpu_usage,
        'Memory %': mem_usage,
        'Disk I/O (IOPS)': disk_io,
        'GPU %': gpu_usage,
        'Network IN (MB)': network_in,
        'Network OUT (MB)': network_out
    })
    return df

def generate_incidents_data():
    # Include some GenAI & Kubernetes/GKE/Serverless-specific issues
    possible_issues = [
        "OOMKilled on GKE Pod",
        "GPU out of memory during LLM training",
        "High concurrency scaling error on Cloud Run",
        "Kernel panic on HPC node",
        "Python env mismatch in Function App",
        "High CPU usage on GPT inference",
        "Disk I/O bottleneck in HPC GPU cluster",
        "Network latency on multi-region ML inference"
    ]
    mock_incidents = []
    for i in range(1, 6):
        severity = random.choice(["P1", "P2", "P3"])
        machine_identified = random.choice([True, False])
        status = random.choice(["Open", "Resolved"])
        desc = f"Incident {i}: {random.choice(possible_issues)}"
        mock_incidents.append({
            "id": 1000 + i,
            "severity": severity,
            "machine_identified": machine_identified,
            "status": status,
            "description": desc
        })
    df = pd.DataFrame(mock_incidents)
    return df

def generate_security_data():
    critical_events = random.randint(0, 3)
    high_events = random.randint(4, 10)
    medium_events = random.randint(5, 20)
    compliance_score = random.randint(80, 97)
    data = {
        "Critical Events": critical_events,
        "High Events": high_events,
        "Medium Events": medium_events,
        "Compliance Score (%)": compliance_score
    }
    return data

def generate_cost_data():
    current_spend = random.randint(25000, 90000)
    budget = 70000
    anomalies = random.choice([True, False])
    return current_spend, budget, anomalies

# ----------------------
# Basic ML Approaches
# ----------------------
def detect_anomalies(df):
    numeric_cols = ['CPU %', 'Memory %', 'GPU %', 'Network IN (MB)', 'Network OUT (MB)', 'Disk I/O (IOPS)']
    data = df[numeric_cols].values
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(data)
    scores = iso_forest.decision_function(data)
    outliers = iso_forest.predict(data)
    df['anomaly_score'] = scores
    df['is_anomaly'] = (outliers == -1)
    return df

def cluster_usage_patterns(df):
    numeric_cols = ['CPU %', 'Memory %', 'GPU %', 'Network IN (MB)', 'Network OUT (MB)', 'Disk I/O (IOPS)']
    data = df[numeric_cols].values
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    kmeans.fit(data)
    clusters = kmeans.labels_
    df['usage_cluster'] = clusters
    return df

def generate_predicted_data(df):
    df_sorted = df.sort_values(by='timestamp')
    df_sorted['index'] = range(len(df_sorted))
    X = df_sorted[['index']]
    y = df_sorted['CPU %']
    model = LinearRegression()
    model.fit(X, y)
    future_points = 6
    last_index = df_sorted['index'].max()
    future_indices = np.arange(last_index+1, last_index+1+future_points)
    y_pred = model.predict(future_indices.reshape(-1, 1))
    future_ts = [df_sorted['timestamp'].iloc[-1] + timedelta(minutes=i+1) for i in range(future_points)]
    pred_df = pd.DataFrame({
        'timestamp': future_ts,
        'Predicted CPU %': y_pred
    })
    return pred_df

def main():
    st.set_page_config(page_title="AIOps Dashboard - Global Telecom Operator", layout="wide")
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
    st.title("AIOps Dashboard - Global Telecom Operator")
    st.sidebar.title("Telco Filters")

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
    minutes_to_fetch = TIME_RANGE_MAP[selected_time_range]
    st.sidebar.markdown("### ML Toggles")
    show_anomalies = st.sidebar.checkbox("Show Anomalies", value=True)
    show_clusters = st.sidebar.checkbox("Show Clusters", value=True)
    show_prediction = st.sidebar.checkbox("Show Predictive CPU Trend", value=True)

    scope_text = f"Scope: {selected_division} > {selected_bu} > {selected_region} > {selected_service} | Network: {selected_net_type} | Environment: {selected_env} | Time: {selected_time_range}"
    st.write(scope_text)

    # Mock key metrics (MTTI, MTTR, Machine Identified ratio, Monthly spend)
    mtti_value = round(random.uniform(3.0, 6.0), 1)
    mtti_delta = random.choice([-0.1, 0.15, 0.2, -0.05])
    mttr_value = round(random.uniform(10, 30), 1)
    mttr_delta = random.choice([-0.2, 0.25, 0.1, -0.1])
    machine_ratio = random.randint(50, 90)
    machine_delta = random.choice([-5, -2, 3, 5])
    cost_spend, budget, cost_anomaly = generate_cost_data()
    cost_diff = cost_spend - budget

    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        delta_color_mtti = "normal" if mtti_delta < 0 else "inverse"
        st.metric("MTTI (Mean Time to Identify)", f"{mtti_value} min", f"{mtti_delta*100:.1f}% vs prev", delta_color=delta_color_mtti)
    with col_k2:
        delta_color_mttr = "normal" if mttr_delta < 0 else "inverse"
        st.metric("MTTR (Mean Time to Repair)", f"{mttr_value} min", f"{mttr_delta*100:.1f}% vs prev", delta_color=delta_color_mttr)
    with col_k3:
        mir_delta_color = "normal" if machine_delta > 0 else "inverse"
        st.metric("Machine-Identified Alerts", f"{machine_ratio}%", f"{machine_delta}%", delta_color=mir_delta_color)
    with col_k4:
        cost_delta_color = "inverse" if cost_diff > 0 else "normal"
        st.metric("Monthly Cloud/Network Spend", f"${cost_spend}", f"${cost_diff} vs Budget", delta_color=cost_delta_color)

    # Generate Data
    df_infra = generate_infrastructure_data(num_points=minutes_to_fetch)
    df_infra_sorted = df_infra.sort_values(by="timestamp")
    df_incidents = generate_incidents_data()
    security_data = generate_security_data()

    # ML pipeline
    if show_anomalies:
        df_infra_sorted = detect_anomalies(df_infra_sorted)
    if show_clusters:
        df_infra_sorted = cluster_usage_patterns(df_infra_sorted)
    if show_prediction:
        pred_df = generate_predicted_data(df_infra_sorted)

    st.subheader("Active Incidents")
    open_incidents = df_incidents[df_incidents['status'] == "Open"]
    p1_incidents = open_incidents[open_incidents['severity'] == "P1"]
    st.write(f"Total Open Incidents: **{len(open_incidents)}** | P1 Incidents: **{len(p1_incidents)}**")
    st.dataframe(df_incidents, height=200)

    left_col, center_col, right_col = st.columns([1,2,1])
    with left_col:
        st.subheader("Infrastructure Metrics")
        st.line_chart(df_infra_sorted.set_index('timestamp')[['CPU %','Memory %','GPU %']], height=300)
        st.write("**Disk I/O (IOPS)**")
        st.line_chart(df_infra_sorted.set_index('timestamp')[['Disk I/O (IOPS)']], height=150)

    with center_col:
        st.subheader("Network Throughput (MB)")
        st.line_chart(df_infra_sorted.set_index('timestamp')[['Network IN (MB)','Network OUT (MB)']], height=200)
        if show_anomalies:
            st.subheader("Anomalies Detected")
            anoms = df_infra_sorted[df_infra_sorted['is_anomaly']==True]
            if len(anoms) > 0:
                st.table(anoms[['timestamp','CPU %','Memory %','GPU %','Network IN (MB)','Network OUT (MB)','Disk I/O (IOPS)','anomaly_score']])
            else:
                st.success("No anomalies detected in this time range.")
        if show_clusters:
            st.subheader("Usage Clusters")
            if 'usage_cluster' in df_infra_sorted.columns:
                cluster_counts = df_infra_sorted['usage_cluster'].value_counts().sort_index()
                cluster_map = {0:"Low Usage Pattern",1:"Medium Usage Pattern",2:"High Usage Pattern"}
                st.write("**Cluster distribution**:")
                for cidx in cluster_counts.index:
                    st.write(f"- Cluster {cidx} ({cluster_map.get(cidx,'Unknown')}): {cluster_counts[cidx]} data points")

    with right_col:
        st.subheader("Security & Compliance")
        sec_df = pd.DataFrame([security_data])
        st.table(sec_df)
        st.subheader("Cost & Resource Efficiency")
        if cost_anomaly:
            st.warning("Cost anomaly detected! Investigate for GPU or concurrency overuse.")
        else:
            st.success("No cost anomalies. Spend within expected range.")
        st.write("Underutilized HPC/GPU Nodes (Sample)")
        underutilized = [
            {"Resource":"GPU-Node-001","GPU%":10,"Memory%":25},
            {"Resource":"GKE-Pool-02","CPU%":12,"Memory%":30}
        ]
        st.table(pd.DataFrame(underutilized))

        if show_prediction:
            st.subheader("Predictive CPU Trend (Next 6 mins)")
            st.write("Simple linear regression forecast on CPU usage.")
            hist_df = pd.DataFrame({
                'timestamp': df_infra_sorted['timestamp'],
                'CPU %': df_infra_sorted['CPU %']
            })
            hist_df['Predicted CPU %'] = np.nan
            merged_df = hist_df.set_index('timestamp')
            if 'pred_df' in locals():
                future_df = pred_df.set_index('timestamp')
                merged_df = pd.concat([merged_df, future_df], axis=1)
            st.line_chart(merged_df[['CPU %','Predicted CPU %']], height=250)

    st.info("Dashboard includes telescopic filters, key metrics (MTTI, MTTR), and AIOps ML features (anomaly detection, clustering, predictive CPU). Common GenAI errors in the incident feed reflect real HPC/GKE/LLM challenges.")

if __name__ == "__main__":
    main()
