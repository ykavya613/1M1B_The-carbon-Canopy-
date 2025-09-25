# Creating a tailored Teak (Tectona grandis) cohort simulation for Telangana
# - 20-year annual simulation for a cohort of 1000 trees
# - Weekly observational "measurements" over 4 weeks for trees of different ages (1,3,5,8,12,16,20 yrs)
# - Uses a logistic growth model for above-ground dry biomass per tree (kg)
# - Carbon fraction and CO2 conversion use IPCC defaults (carbon fraction = 0.47, CO2 = C * 44/12)
# Output:
# - cohort_results_teak.csv (annual results for 1000-tree cohort)
# - observed_samples_teak.csv (simulated weekly observations for different ages)
# - Two plots saved in /mnt/data: biomass_plot_teak.png and co2_plot_teak.png
# - A short summary printed below
# Note: this is a modelled approximation using literature-informed, conservative parameters for teak.
#       If you want a different species or to repeat for other species from your list, tell me and I'll run them next.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Parameters (literature-informed, conservative estimates)
species = "Teak (Tectona grandis)"
region = "Telangana, India"
cohort_size = 1000
years = np.arange(0, 21)  # 0 to 20 years inclusive

# Logistic growth parameters (per-tree above-ground dry biomass in kg)
A = 600.0   # asymptotic above-ground dry biomass per tree (kg) at maturity (~20 years)
k = 0.35    # growth rate parameter (1/yr) - controls how quickly biomass approaches A
t0 = 6.0    # inflection point (years)

def agb_per_tree(t):
    """Logistic growth for AGB (kg) per tree at age t (years)."""
    return A / (1.0 + np.exp(-k * (t - t0)))

# Carbon and CO2 conversions
carbon_fraction = 0.47  # IPCC default wood carbon fraction (dry mass -> carbon)
co2_per_kg_c = 44.0/12.0  # convert kg C to kg CO2

# Build annual cohort results (aggregate for cohort of trees)
agb_per_tree_values = agb_per_tree(years)
total_agb = agb_per_tree_values * cohort_size  # total above-ground biomass (kg)
total_carbon = total_agb * carbon_fraction     # kg C
total_co2 = total_carbon * co2_per_kg_c        # kg CO2

df_cohort = pd.DataFrame({
    "year": years,
    "age_years": years,
    "agb_per_tree_kg": np.round(agb_per_tree_values, 2),
    "total_agb_kg": np.round(total_agb, 2),
    "total_carbon_kg": np.round(total_carbon, 2),
    "total_co2_kg": np.round(total_co2, 2)
})

# Save cohort CSV
os.makedirs("/mnt/data", exist_ok=True)
cohort_csv_path = "/mnt/data/cohort_results_teak.csv"
df_cohort.to_csv(cohort_csv_path, index=False)

# --- Create simulated weekly observations for different ages over 4 weeks ---
obs_ages = np.array([1, 3, 5, 8, 12, 16, 20])  # years
weeks = np.arange(1, 5)  # 4 weeks of observations
rows = []
rng = np.random.default_rng(42)
for age in obs_ages:
    true_agb = agb_per_tree(age)
    for w in weeks:
        # Simulate small measurement noise (±2-5% random)
        noise = rng.normal(loc=0.0, scale=0.03)  # 3% std dev
        measured_agb = true_agb * (1.0 + noise)
        measured_carbon = measured_agb * carbon_fraction
        measured_co2 = measured_carbon * co2_per_kg_c
        rows.append({
            "obs_week": int(w),
            "tree_age_years": age,
            "measured_agb_kg": round(float(measured_agb), 2),
            "measured_carbon_kg": round(float(measured_carbon), 2),
            "measured_co2_kg": round(float(measured_co2), 2)
        })

df_obs = pd.DataFrame(rows)
obs_csv_path = "/mnt/data/observed_samples_teak.csv"
df_obs.to_csv(obs_csv_path, index=False)

# --- Plots ---
# 1) Biomass plot (total cohort AGB in tonnes over 20 years)
plt.figure(figsize=(8,5))
plt.plot(df_cohort["age_years"], df_cohort["total_agb_kg"] / 1000.0)  # tonnes
plt.xlabel("Age (years)")
plt.ylabel("Total AGB (tonnes) for cohort of 1000 trees")
plt.title(f"{species} — Cohort Above-Ground Biomass over 20 years ({region})")
plt.grid(True)
biomass_plot_path = "/mnt/data/biomass_plot_teak.png"
plt.savefig(biomass_plot_path, bbox_inches="tight")
plt.close()

# 2) CO2 plot (total cohort CO2 in tonnes over 20 years)
plt.figure(figsize=(8,5))
plt.plot(df_cohort["age_years"], df_cohort["total_co2_kg"] / 1000.0)  # tonnes CO2
plt.xlabel("Age (years)")
plt.ylabel("Total CO2 equivalent (tonnes) for cohort of 1000 trees")
plt.title(f"{species} — Cohort CO2 Sequestration over 20 years ({region})")
plt.grid(True)
co2_plot_path = "/mnt/data/co2_plot_teak.png"
plt.savefig(co2_plot_path, bbox_inches="tight")
plt.close()


print(summary)

# Show the observational table to the user
import caas_jupyter_tools as tools
tools.display_dataframe_to_user("Simulated weekly observations (Teak)", df_obs.head(20))

# Also return the cohort DF for quick inline viewing
df_cohort.head(10)
