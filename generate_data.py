"""
generate_data.py
Generates a realistic synthetic dataset of infrastructure construction projects.
Based on patterns from FHWA, World Bank, and McKinsey infrastructure studies.
"""

import pandas as pd
import numpy as np

np.random.seed(42)
N = 1200

project_types = ["Highway", "Bridge", "Water Treatment", "Rail", "Airport", "Tunnel", "Pipeline"]
regions = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
procurement = ["Design-Bid-Build", "Design-Build", "CMAR", "P3"]
complexity = ["Low", "Medium", "High"]

def generate_dataset(n=N):
    data = []

    for i in range(n):
        ptype = np.random.choice(project_types, p=[0.25, 0.15, 0.15, 0.1, 0.1, 0.1, 0.15])
        region = np.random.choice(regions)
        proc = np.random.choice(procurement, p=[0.45, 0.3, 0.15, 0.1])
        comp = np.random.choice(complexity, p=[0.3, 0.45, 0.25])

        budget_m = np.random.lognormal(mean=3.5, sigma=1.2)  # $M
        duration_months = np.random.randint(6, 84)
        num_contractors = np.random.randint(1, 12)
        change_orders = np.random.poisson(lam=3 if comp == "High" else 1.5)
        soil_risk = np.random.choice([0, 1], p=[0.65, 0.35])
        weather_delays = np.random.poisson(lam=2)
        design_complete_pct = np.random.uniform(0.3, 1.0)
        owner_experience = np.random.randint(1, 20)
        prior_delays = np.random.choice([0, 1], p=[0.7, 0.3])

        # Cost overrun logic — grounded in real research
        overrun_prob = 0.35  # base rate ~35% from McKinsey
        if comp == "High": overrun_prob += 0.20
        elif comp == "Medium": overrun_prob += 0.10
        if proc == "Design-Bid-Build": overrun_prob += 0.10
        if soil_risk: overrun_prob += 0.12
        if change_orders > 4: overrun_prob += 0.15
        if design_complete_pct < 0.6: overrun_prob += 0.12
        if prior_delays: overrun_prob += 0.10
        if weather_delays > 3: overrun_prob += 0.08
        if owner_experience < 5: overrun_prob += 0.08
        overrun_prob = min(overrun_prob, 0.95)

        overrun = int(np.random.random() < overrun_prob)

        # Overrun magnitude if overrun occurred
        overrun_pct = 0.0
        if overrun:
            overrun_pct = np.random.uniform(5, 80)

        data.append({
            "project_id": f"PROJ-{i+1:04d}",
            "project_type": ptype,
            "region": region,
            "procurement_method": proc,
            "complexity": comp,
            "budget_million_usd": round(budget_m, 2),
            "planned_duration_months": duration_months,
            "num_contractors": num_contractors,
            "change_orders": change_orders,
            "soil_risk": soil_risk,
            "weather_delays_days": weather_delays,
            "design_complete_pct": round(design_complete_pct, 2),
            "owner_experience_years": owner_experience,
            "prior_project_delays": prior_delays,
            "cost_overrun": overrun,
            "overrun_pct": round(overrun_pct, 1)
        })

    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/projects.csv", index=False)
    print(f"Dataset generated: {len(df)} projects")
    print(f"Overrun rate: {df['cost_overrun'].mean():.1%}")
    print(df.head())
