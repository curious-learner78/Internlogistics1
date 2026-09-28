# Strategic Planning & Data Exploration in Logistics

## 📌 Project Overview
This repository contains the Week 1 submission for the **Yuva Logistics Data Analyst Internship**. The project simulates the strategic planning, data exploration, and system architecture phase of an urban e-commerce last-mile delivery framework.

The primary objective is to demonstrate how data science concepts—including spatial clustering, predictive regression modeling, and dynamic routing—can solve modern logistics challenges like high fuel consumption, delayed deliveries, and inefficient fleet resource allocation.

---

## 🎯 Key Performance Indicators (KPIs)
To measure and optimize supply chain performance, the following metrics are tracked:

1. **On-Time Delivery Rate (OTDR):** Target $\ge 95\%$
   * Measures delivery punctuality and reliability across urban zones.
2. **Cost Per Delivery (CPD):** Target $\le \$4.50$ / order
   * Evaluates operational cost efficiency per order (fuel, driver labor, overhead).
3. **Vehicle Capacity Utilization:** Target $\ge 85\%$
   * Tracks volume and weight payload efficiency per trip.
4. **Order Fulfillment Lead Time:** Target $< 120$ minutes
   * Measures total elapsed time from dispatch to consumer handoff.

---

## 🛠️ Data Science Methodologies Applied

* **Spatial Clustering (K-Means):** Groups latitude and longitude delivery locations to determine optimal locations for regional micro-fulfillment hubs.
* **Predictive Modeling (Random Forest Regressor):** Forecasts estimated delivery transit duration using distance, dispatch time, parcel weight, and assigned zone features.
* **Route Optimization (Mathematical Framework):** Applies Vehicle Routing Problem (VRP) constraints to reduce total fleet transit distance.

---

## 📁 Repository Structure

```text
├── README.md               # Project documentation and strategic summary
└── logistics_analysis.py   # Python module for data cleaning, clustering, and modeling
```
🚀 How to Run the Python Code
Prerequisites
Make sure you have Python 3.8+ and the required packages installed:
```
pip install numpy pandas scikit-learn matplotlib
```
## Execution
Run the analysis script directly from your terminal:
```
python logistics_analysis.py
```
## 📊 Expected Outcomes & Strategic Business Impact

* 12%–18% Reduction in Transit Mileage: Reduced fuel usage and carbon footprint through spatial clustering.
* Improved Fleet Dispatch Accuracy: Accurate arrival window forecasting leads to enhanced customer satisfaction.
* Data-Driven Infrastructure Placement: Centroid mapping identifies high-demand zones for micro-warehouse expansion.
