# Carbon Canopy - Warangal Region Simulator (multi-species)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load species data from CSV or dictionary
species_params = pd.read_csv("species_params.csv")

def simulate_species(species, n_trees, years, seed=42):
    np.random.seed(seed)
    row = species_params[species_params["species"] == species].iloc[0]
    init_biomass = row["init_biomass_kg"]
    increment = row["annual_increment_kg"]
    survival = row["annual_survival_prob"]
    carbon_fraction = row["carbon_fraction"]

    alive = np.ones(n_trees, dtype=bool)
    biomass = np.full(n_trees, init_biomass, dtype=float)
    records = []

    for year in range(1, years+1):
        survived = np.random.binomial(1, survival, size=n_trees).astype(bool)
        alive = alive & survived
        biomass[alive] += increment
        biomass[~alive] = 0.0

        total_biomass = biomass.sum()
        total_carbon = total_biomass * carbon_fraction
        total_co2 = total_carbon * (44/12)

        records.append({
            "year": year,
            "species": species,
            "alive_trees": int(alive.sum()),
            "total_biomass_kg": float(total_biomass),
            "total_carbon_kg": float(total_carbon),
            "total_co2_kg": float(total_co2)
        })

    return pd.DataFrame(records)

# Run simulation for all 5 species (1000 trees each, 20 years)
results = []
for sp in species_params["species"]:
    df = simulate_species(sp, n_trees=1000, years=20)
    results.append(df)

results_df = pd.concat(results, ignore_index=True)

# Save to CSV
results_df.to_csv("warangal_results.csv", index=False)

# Plot example: CO2 sequestration comparison
plt.figure(figsize=(10,6))
for sp in species_params["species"]:
    sub = results_df[results_df["species"] == sp]
    plt.plot(sub["year"], sub["total_co2_kg"]/1000, label=sp, marker="o")
plt.xlabel("Year")
plt.ylabel("Total CO2 (tonnes)")
plt.title("CO2 Sequestration by Species (1000 trees each)")
plt.legend()
plt.grid(True)
plt.show()

results_df.head()

