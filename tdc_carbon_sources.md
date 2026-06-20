# TDC Carbon Model: Data Provenance

Scope: every value used by the TDC side of the carbon model (`tdc_carbon` and
the shared `it_quantities`). Variable names refer to fields of `CarbonScenario`.

TDC life-cycle carbon = operational (grid) carbon + facility embodied
(core & shell, MEP) + IT embodied (refreshed every 5 years). The facility and
operational assumptions trace to a single anchor study:

- **Alissa 2025** = Alissa, H., Nick, T., Raniwala, A., et al. (2025).
  *Using life cycle assessment to drive innovation for sustainable cool clouds.*
  Nature 641, 331-338. doi:10.1038/s41586-025-08832-3.

---

## 1. Operational (grid) carbon

Annual operational carbon = `it_capacity_mw` x `pue` x `load_factor` x 8760 h
x `grid_intensity`, decayed by `grid_decarb` per year.

| Parameter | Variable | Value | Units | Source | Notes |
|---|---|---|---|---|---|
| Power usage effectiveness | `pue` | 1.12 | - | Alissa 2025 | Microsoft reports PUE as low as 1.12 |
| Load factor | `load_factor` | 1.0 | - | model assumption | 100% utilization, matches ODC |
| Grid carbon intensity | `grid_intensity` | 0.40 | tCO2e/MWh | assumption, see note (a) | == kgCO2e/kWh |
| Grid decarbonization | `grid_decarb` | 0.0 | per yr | model assumption | flat grid over mission by default |

### Notes

(a) 0.40 tCO2e/MWh is a round baseline near the global / US-average grid
intensity. It is **not** attributed to a specific reference in the model. The
comparison figure annotates "US avg ~0.37" and "France ~0.05" as context.
**Choose and cite a grid-intensity source** (for example IEA, Ember, or US EPA
eGRID for the US average; national inventories for country cases). Because TDC
operational carbon scales linearly with this value, it is the dominant TDC
sensitivity and sets the ODC/TDC crossover.

Alissa 2025 supports the structure: for data centres the use (operational) phase
is the largest contributor to GHG impact, with embodied server carbon second.

---

## 2. Facility embodied (core & shell, MEP)

Floor area = `it_capacity_mw` x 1000 / `power_density_kw_m2`; embodied carbon =
area x intensity.

| Parameter | Variable | Value | Units | Source |
|---|---|---|---|---|
| Power density | `power_density_kw_m2` | 2.0 | kW/m^2 | facility design assumption, see note (b) |
| Core & shell intensity | `shell_kgco2e_m2` | 650 | kgCO2e/m^2 | Alissa 2025 (facility LCA) |
| MEP intensity | `mep_kgco2e_m2` | 400 | kgCO2e/m^2 | Alissa 2025 (facility LCA) |

(b) Power density sets floor area from capacity; confirm whether 2.0 kW/m^2 is
from Alissa 2025 or an independent facility assumption, and cite accordingly.

---

## 3. IT embodied (shared with ODC, apples-to-apples)

Identical method and values as the ODC IT embodied, so the comparison isolates
launch/facility differences rather than chip differences.

| Parameter | Variable | Value | Units | Source |
|---|---|---|---|---|
| Node power | `node_power_kw` | 10.4 | kW/node | sets node count from capacity |
| GPU baseboard PCF | `gpu_baseboard_kgco2e` | 2274 | kgCO2e/node | NVIDIA HGX B200 product carbon footprint |
| Host server PCF | `host_server_kgco2e` | 1300 | kgCO2e/node | NVIDIA HGX B200 PCF / server LCA |
| IT mass density | `it_mass_density` | 55000 | kg/MW | data-center rack inventory (also used for ODC launch mass) |
| Mission length | `mission_years` | 15 | yr | comparison basis |
| IT refresh interval | `it_refresh_years` | 5 | yr | refreshes at years 0, 5, 10 (3 generations) |

IT embodied methodology follows Falk et al. 2025 (cradle-to-grave GPU LCA); the
per-node figures are taken from the NVIDIA HGX B200 PCF summary, whose reported
embodied intensity is about 0.50 gCO2e per FLOPS (24% below the prior H100
generation).

---

## References

- Alissa, H., Nick, T., Raniwala, A., et al. (2025). Using life cycle assessment to drive innovation for sustainable cool clouds. Nature 641, 331-338. doi:10.1038/s41586-025-08832-3.
- Falk, S., Ekchajzer, D., Pirson, T., Lees-Perasso, E., Wattiez, A., Biber-Freudenberger, L., Luccioni, S., & van Wynsberghe, A. (2025). More than Carbon: Cradle-to-Grave environmental impacts of GenAI training on the Nvidia A100 GPU. arXiv:2509.00093.
- NVIDIA (2025). NVIDIA HGX B200 Reduces Embodied Carbon Emissions Intensity (Product Carbon Footprint summary). NVIDIA Technical Blog.

## Open items to confirm before submission

1. Choose and cite a grid carbon-intensity source for the 0.40 tCO2e/MWh baseline (IEA / Ember / EPA eGRID).
2. Confirm whether `power_density_kw_m2` = 2.0 and the 650 / 400 kgCO2e/m^2 facility intensities all come from Alissa 2025, or from a mix of sources.
3. Confirm the NVIDIA HGX B200 PCF node figures (2274, 1300) against the published summary.
4. Cite the source for `it_mass_density` = 55000 kg/MW (rack inventory) if it is to appear in the carbon paper as well as the financial one.
