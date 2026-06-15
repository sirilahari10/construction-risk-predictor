# Infrastructure Cost Overrun Risk Predictor

An end-to-end machine learning pipeline that predicts cost overrun risk for infrastructure projects **before construction begins** — using project type, procurement method, design readiness, and site-specific factors.

**[→ Live Dashboard](https://sirilahari10.github.io/construction-risk-predictor)**

---

## The Problem

McKinsey estimates that **98% of megaprojects** experience cost overruns averaging 80% above budget. Early risk identification — before contracts are signed — gives project directors the ability to intervene: tighten design packages, choose lower-risk procurement paths, or price risk into bids.

## What This Does

- Trains an **XGBoost classifier** on 1,200 infrastructure projects
- Predicts probability of cost overrun using 13 pre-construction attributes
- Segments projects into **High / Medium / Low risk tiers**
- Surfaces the most predictive risk factors for each project type
- Renders everything in an **interactive dashboard** with zero backend — one HTML file, deployable to GitHub Pages in 60 seconds

## Results

| Metric | Score |
| Accuracy (test set) | 70.4% |
| ROC-AUC | 0.655 |
| 5-Fold CV AUC | 0.568 ± 0.046 |
| Dataset size | 1,200 projects |
| Overrun base rate | 66.7% |

## Top Risk Drivers

1. **Prior delay history** — past behavior is the single strongest predictor
2. **Change orders** — volume of scope changes at time of contract
3. **Project complexity** — High complexity projects overrun at 75.6% vs 57.1% for Low
4. **Design completion %** — incomplete design at contract award is a leading cause of claims
5. **Procurement method** — Design-Bid-Build overruns at 71% vs 59% for Design-Build

## Key Findings

- **Bridges** have the highest overrun rate at 75.5%
- **Design-Build** procurement reduces overrun risk by ~12 percentage points vs Design-Bid-Build
- **Prior delay history** alone lifts overrun probability by 10+ percentage points
- Projects with **< 60% design completion** at contract award are significantly more likely to overrun

## Stack

```
Python 3.11
├── pandas / numpy          — data generation & feature engineering
├── scikit-learn            — preprocessing, train/test split, CV
├── xgboost                 — classification model
└── Chart.js (frontend)     — interactive dashboard charts
```



## Data

Synthetic dataset of 1,200 projects calibrated to patterns from:
- FHWA (Federal Highway Administration) project data
- World Bank infrastructure project database
- McKinsey Global Institute: *Reinventing Construction* (2017)

## About

Built by **Siri Lahari Chava** — Data Scientist & ML Engineer.
