import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('lcoc_tdc_odc_comparison.csv')

# Calculate Lifetime OPEX for a more nuanced breakdown
# Total Discounted Cost - Initial CAPEX = Discounted Lifetime OPEX
df['discounted_lifetime_opex'] = df['discounted_lifetime_cost_per_kw_it_usd'] - df['capex_per_kw_it_usd']

# Set visual style
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Levelized Cost of Compute (LCOC) Comparison', fontsize=20, fontweight='bold')

# 1. LCOC Comparison (The Efficiency Metric)
sns.barplot(x='configuration', y='LCOC_usd_per_EFLOP', data=df, ax=axes[0,0], palette='viridis')
axes[0,0].set_title('LCOC (USD per EFLOP)', fontsize=14)
axes[0,0].set_ylabel('USD / EFLOP')
axes[0,0].set_xlabel('')

# 2. CAPEX vs OPEX Breakdown
breakdown_df = df[['configuration', 'capex_per_kw_it_usd', 'discounted_lifetime_opex']]
breakdown_df.set_index('configuration').plot(kind='bar', stacked=True, ax=axes[0,1], color=['#2ecc71', '#e74c3c'])
axes[0,1].set_title('Lifetime Cost Composition per kW IT (Discounted)', fontsize=14)
axes[0,1].set_ylabel('Total Cost (USD)')
axes[0,1].set_xlabel('')
axes[0,1].legend(['CAPEX', 'Lifetime OPEX'])

# 3. Utilization & Longevity Scatter
# Larger bubbles indicate higher total lifetime compute output
sns.scatterplot(x='load_factor_k', y='lifetime_years_n', size='discounted_lifetime_compute_eflop_per_kw_it', 
                hue='configuration', data=df, ax=axes[1,0], sizes=(100, 1000), legend='brief')
axes[1,0].set_title('Utilization & Longevity vs Total Lifetime Compute', fontsize=14)
axes[1,0].set_ylabel('Lifetime (Years)')
axes[1,0].set_xlabel('Load Factor (k)')

# 4. Annual Energy Cost per kW IT
sns.barplot(x='configuration', y='annual_energy_cost_per_kw_it_yr_usd', data=df, ax=axes[1,1], palette='magma')
axes[1,1].set_title('Annual Energy Cost per kW IT', fontsize=14)
axes[1,1].set_ylabel('USD / Year')
axes[1,1].set_xlabel('')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('lcoc_visuals.png')