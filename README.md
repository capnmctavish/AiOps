# Telecom AIOps Dashboard with ML (GenAI/HPC/GKE Focus)

This repository contains a Streamlit-based AIOps dashboard tailored for a **global telecom** or **enterprise** environment. It combines:

- **Telescopic Filters** (Organization, Business Unit, Region, Service, Network Type, Environment)
- **Key Metrics** at the top (MTTI, MTTR, Machine-Identified Alerts, and Monthly Spend)
- **AIOps ML Features**:
  - Isolation Forest (anomaly detection)
  - KMeans (usage clustering)
  - Simple Linear Regression (CPU predictive analytics)
- **Common GenAI / HPC / GKE Incidents** (e.g., OOMKilled Pods, GPU memory exhaustion, concurrency errors)
- **Dark-Themed UI** to match modern dashboards and reduce eyestrain

## Dashboard Features

1. **Telescopic Filters**  
   - Allows scoping from **Division** -> **Business Unit** -> **Region** -> **Service**.  
   - Supports additional filters like **Network Type**, **Environment**, and **Time Range** (Last 5 min, 15 min, etc.).

2. **Key AIOps Metrics**  
   - **MTTI (Mean Time to Identify)**  
   - **MTTR (Mean Time to Repair)**  
   - **Machine-Identified Alerts** ratio  
   - **Monthly Cloud/Network Spend** with budget comparison

3. **Mock Data for Incidents & Metrics**  
   - Synthetic data for CPU, Memory, GPU usage, Disk I/O, Network in/out.  
   - Random incidents referencing HPC clusters, GKE pods, AI/LLM deployments, concurrency scaling issues, etc.

4. **ML Pipelines**  
   - **Anomaly Detection** with Isolation Forest (flags outliers among CPU, Memory, GPU, Disk, Network).  
   - **Clustering** with KMeans (groups usage patterns into Low/Medium/High).  
   - **Predictive CPU Trend** with Linear Regression (basic demonstration).

5. **Dark Mode**  
   - Custom CSS overrides for a **dark-themed** look.

## Prerequisites

- **Python 3.7+**  
- **pip** or **conda** for installing dependencies  
- Libraries:
  - **streamlit**  
  - **pandas**  
  - **numpy**  
  - **scikit-learn**  
  - (Optional) Jupyter or any environment for previewing code

## Installation

1. **Clone or Download** this repository:
   ```bash
   git clone https://github.com/capnmctavish/telecom-aiops-dashboard.git
   cd telecom-aiops-dashboard
