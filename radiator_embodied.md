# Embodied Carbon of the ODC Radiator: Sources and Method

**Subsystem (from the paper):** Two flat double-sided panels deployed at 45° from vertical, aluminum structure coated with a 1 mm high-emissivity, low-solar-absorptance **BaSO₄-class radiative-cooling paint** (α_solar = 0.05, α_IR = 0.95). Deployed area A_rad = 85,564 m² for the 100 MW node.
**Metric:** GWP100, cradle-to-gate.

## Headline result

The estimate splits into two readings because of how mass is booked in the paper (see "Key modeling note" below).

| Reading | per m² | per kg | Whole 100 MW radiator |
|---|---|---|---|
| **A. As modeled (BaSO₄ coating only)** | ~4 kg CO₂e/m² (1–9) | ~1 kg CO₂e/kg | ~0.4 kt CO₂e |
| **B. Physically complete (coating + Al structure)** | ~86 kg CO₂e/m² (37–131) | ~9 kg CO₂e/kg | ~7.3 kt CO₂e |

Under Reading B, aluminum is ~95% of the total. The BaSO₄ coating itself is carbon-trivial under either reading.

**Recommendation for the life-cycle carbon paper:** use Reading B (~86 kg CO₂e/m², radiator mass ~790 t). The aluminum substrate is physically required and its added launch mass also carries launch carbon. Neither reading threatens any LCOC conclusion, since the paper's own sensitivity analysis ranks all radiator parameters as negligible.

## Key modeling note (important)

In the paper, the radiator areal density of **4.5 kg/m²** equals exactly 1 mm of BaSO₄ (density ~4,500 kg/m³ × 1 mm = 4.5 kg/m²), and Table III sources it as "BaSO₄ property" [Lee & Banjai]. This reproduces the stated 385 t radiator mass. It implies the mass budget counts only the coating and **omits the aluminum honeycomb panel and heat-pipe network** that the text otherwise says the radiator is built from. Reading B restores that aluminum (~4.7 kg/m²), roughly doubling radiator mass to ~790 t. Worth confirming with the thermal lead whether 4.5 kg/m² was intended as coating-only or as the full panel.

## Bottom-up method

### 1. BaSO₄ coating (the radiating surface)
- Coating: 1 mm thick, modeled at BaSO₄ bulk density → **4.5 kg/m²**; optical properties α_solar = 0.05, α_IR = 0.95 [Lee & Banjai; Fan et al. 2025].
- BaSO₄ is a low-carbon mineral. Ground natural barite is ~0.1–0.3 kg CO₂e/kg; precipitated grades (blanc fixe) are higher due to carbothermal barite-to-sulfide processing. Central **1.0 kg CO₂e/kg** (range 0.3–2.0), benchmarked against comparable precipitated mineral fillers such as PCC at ~0.85 kg CO₂e/kg [Joo et al. 2024].
- Coating contribution: 4.5 kg/m² × ~1.0 = **~4 kg CO₂e/m²**.

### 2. Aluminum structure (Reading B only)
- Engineering bill of materials for an Al honeycomb panel with embedded heat pipes: two ~0.4 mm facesheets (2.16 kg/m²), ~25 mm honeycomb core at ~50 kg/m³ (1.25 kg/m²), extruded Al heat pipes (~1.3 kg/m²) → **~4.7 kg/m² aluminum**.
- Emission factor: IAI 2023 global-average primary aluminium **14.8 kg CO₂e/kg** [IAI 2024], plus a **2.5 kg/kg** semi-fabrication adder (rolling, extrusion, honeycomb expansion/bonding, machining) → ~17.3 kg CO₂e/kg central. Range from EU/low-carbon primary (6.6 kg/kg [European Aluminium]) up to high-carbon-grid primary (22 kg/kg [IAI 2024]) plus processing.
- Aluminum contribution: 4.7 kg/m² × 17.3 = **~81 kg CO₂e/m²**.

### 3. Adhesive, ammonia charge, inserts
Negligible mass and carbon; folded into rounding.

## Sensitivity
The range is driven almost entirely by aluminum smelter grid carbon intensity (clean hydro/EU vs coal-heavy). The BaSO₄ emission factor and the coating mass have little leverage on the total under Reading B. If the radiator is kept as coating-only (Reading A), the whole subsystem is carbon-trivial.

## Context vs the PV cell
The radiator is roughly 16x lighter on embodied carbon per m² than the triple-junction PV cell (~86 vs ~1,400 kg CO₂e/m²) and far lighter per kg (~9 vs ~1,300 kg CO₂e/kg). The PV array remains the embodied-carbon hotspot of the spacecraft; the radiator is a secondary, aluminum-dominated term.

## Bibliography

1. Lee, T. R., Banjai, C. (2026). Space Computing: A Techno-Economic Analysis of Orbital Data Centers. Course report, 1.086, MIT. [Radiator spec, areal density, A_rad, mass budget; Tables I and III.]

2. Fan, Y., Chen, H., Liu, X., Zhao, Y., Huang, Y., Liu, J., Wang, C. (2025). Radiative cooling in outer space: Fundamentals, advances in materials and applications, and perspectives. *Advanced Materials*, 37, e06795. [BaSO₄-class coating optical properties α_solar, α_IR; ref [11] in the paper.]

3. International Aluminium Institute (2024). Primary Aluminium Greenhouse Gas Emissions Intensity, and Aluminium Carbon Footprint FAQs. Global-average primary aluminium = 14.8 t CO₂e/t (2023); cradle-to-gate range 4.5–22. https://international-aluminium.org/statistics/greenhouse-gas-emissions-primary-aluminium/

4. European Aluminium (2024). A Low Carbon Footprint. EU primary aluminium ~6.6 kg CO₂e/kg (2023). https://european-aluminium.eu/

5. Joo, S., et al. (2024). Techno-economic and life cycle environmental assessments of CO₂ utilization for precipitated calcium carbonate and ammonium sulfate co-production. *Journal of CO₂ Utilization*. [PCC ~0.85 kg CO₂e/kg, benchmark for precipitated mineral filler carbon intensity.]

*Note: Item 1 is the load-bearing source for the radiator geometry, areal density, and the mass-budget ambiguity that drives the A/B split. Item 3 is the load-bearing source for the dominant aluminum term under Reading B. The BaSO₄ emission factor (item 5 as analogue) is an estimate; no dedicated BaSO₄-paint LCA was located, but the coating is carbon-trivial under either reading, so this uncertainty does not affect the result.*