# AI Education Product Analytics Platform

Simulating how a data science team would analyze, experiment on, and optimize an AI-powered education product.

This project mirrors real-world product analytics workflows used at companies like Google, focusing on metrics ownership, experimentation, causal inference, and LLM quality evaluation.

---

## 🚀 Project Overview

This project simulates analytics for a fictional product:

> **AI Study Assistance** - an LLM-powered tool that helps students summarize notes, generate quizzes, and explain concepts.

The goal is to act as the product data scientist reponsible for:

- Defining success metrics
- Understanding user behavior
- Running product experiments
- Evaluating LLM quality
- Forecasting growth
- Delivering insigts through dashboards

This is an end-to-end product analytcis and experimentation system.

---

## 🎯 Objectives

- Build a realistic large-scale product dataset
- Define and compute product KPIs
- Analyze retetion and engagement drivers
- Simulate and evaluate A/B experiments
- Design an LLM evaluation framework
- Forecase growth trends
- Democratize insights through dashboards
- Deliver executive-ready product recommendations

---

## 📊 Dataset

This project geenrates synthetic event-level product logs that mimic real analytics pipelines.

Each row represents a user event:

```
user_id
session_id
event_time
feature_used
engagement_time
is_teacher
experiment_group
llm_ratng
retained_7d
retained_30d
subscription
```

Scale:

- 1M+ siulated product events
- Thousand of users
- Multi-session behavior
- A/B experiment assignment
- LLM satisfaction signals

---

## 🧠 Key Analyses

### 1. Metrics Framework

Define core KPIs:

- Daily Active Users (DAU)
- Retention rate
- Feature adoption
- Session depth
- LLM satisfaction score
- Subscription conversion
- Teacher vs student engagement

---

### 2. User Behavior Analytics

- Cohort retention analysis
- Funnel analysis
- Segmentation
- Churn modeling
- Feature importance with SHAP
- Engagement drivers

---

### 3. Experimentation

Simulated A/B testing for new AI features:

- Control vs treatment analysis
- Statistical significance testing
- Confidence intervals
- Uplift estimation
- Causal inference methods

---

### 4. LLM Evaluation System

Framework for monitoring AI output quality:

- Human rating simulation
- Automated scoring pipelines
- Quality tracking metricss
- Teacher vs student satisfaction
- Hallucination detection signals

---

### 5. Forecasting

- DAU growth projections
- Retention forecasting
- Scenario simulation
- Subscription trend modeling

---

### 6. Dashboard

Interactive Dashboard for stakeholders:

- KPI tracking
- Experiment results
- User segmentation
- LLM quality monitoring

Built with Stremalit

---

## 🛠 Tech Stack

Python
Pandas
NumPy
Scikit-learn
Statsmodels
SHAP
SQL
Plotly
Streamlit
Jupyter

Optional:

DuckDB / BigQuery-style analytics

---

## 📁 Project Structure

<img width="548" height="906" alt="image" src="https://github.com/user-attachments/assets/18d4d118-0ec8-45a5-9ef9-4595013901d3" />

---

## ▶️ How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Generate synthetic data

```bash
python src/01_generate_data.py
```

### Run analytics pipeline

```bash
python src/02_metrics_pipeline.py
```

### Launch dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📈 Example Insights

- Quiz feature drivers +18% retention
- Teachers show 2x engagement depth
- LLM rating strongly predicts subscription
- Experiment varian improved session time by 11%

---

## 💡 Business Impact

This project demonstrates how analytics can:

- Guide product strategy
- Improve AI quality
- Increase retention
- Optimize feature rollout
- Drive growth decisions

It sumulates real product data science workflows used in large-scale consumer platforms.

---

## 🧾 Resume Bullet

Built an end-to-end analytics platform for an AI education product, defining KPIs, running A/B experiments, modeling retention drivers, and designing and LLM evaluation system to inform product growth strategy.
