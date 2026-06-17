# Embodied Carbon of a Space-Grade Triple-Junction Cell: Sources and Method

**Cell type:** GaInP/GaInAs/Ge triple-junction, grown on a Ge substrate (the standard space PV stack, e.g. AZUR Space 3G30C, Spectrolab XTE).
**Metric:** GWP100, cradle-to-gate.

## Headline result

| Basis | Central | Range |
|---|---|---|
| per m² of cell | ~1,400 kg CO2e/m² | ~500 – 2,200 |
| per kWp (AM0, 30%) | ~3,400 kg CO2e/kWp | ~1,300 – 5,400 |
| per kg of cell + coverglass | ~1,300 kg CO2e/kg | ~480 – 2,000 |

The germanium substrate accounts for ~78% of the total.

There is no published cradle-to-gate LCA of a 1-sun Ge-substrate space triple-junction cell. This figure is therefore synthesized bottom-up from primary sub-component inventories, with the dominant term resting on a single primary source (Swart et al. 2011).

## Bottom-up method

The cell is decomposed into four contributions, each sourced separately and summed.

### 1. Germanium substrate (dominant, ~78%)
Anchored on Swart et al. (2011), whose life-cycle inventory was supplied directly by Umicore, a leading wafer producer. Key inventory numbers used:
- Wafer: 100 mm diameter, 150 µm thick, mass **6.16 g** [Swart 2011].
- Cumulative energy demand: **216 MJ per wafer** (base case, US grid), varying **122–253 MJ** with fab electricity mix; **~75% of CED is electricity**; primary-energy conversion factor **13.3 MJ per kWh** for the US non-renewable mix [Swart 2011].

Derived quantities (this work): wafer area = 78.5 cm², so a 1-sun array needs **~127 wafers/m²**, giving a substrate CED of **27.5 GJ/m²**. This is more than 3x the entire terrestrial thin-film module of Mohr et al. (2007), because that cell reuses its GaAs wafer via epitaxial lift-off and contains no germanium. The electricity fraction was converted to GWP at a modern mixed-grid intensity of **~0.40 kg CO2e/kWh** (central), yielding ~1,100 kg CO2e/m².

### 2. MOVPE epitaxy + III-V precursors (~15%)
From the contribution analysis of Mohr et al. (2007), in which MOVPE cell growth is ~2/3 of the module's cumulative energy demand. The terrestrial GaInP/GaAs module total is ~30 GJ/kWp at 28.5% efficiency and 3.51 m²/kWp, i.e. **~8.6 GJ/m²** [Mohr et al. 2007; Mohr et al. 2009], giving an epitaxy term of **~5.7 GJ/m²** -> ~220 kg CO2e/m².

### 3. Metallization, ARC, processing (~4%)
Estimated at ~1.5 GJ/m² (Ag/Au contacts, anti-reflective coating, etch, clean), scaled from the non-growth processing share in Mohr et al. (2007). ~60 kg CO2e/m².

### 4. Coverglass + interconnect (~3%)
Engineering estimate for a ~100 µm cover glass plus adhesive and silver interconnects: ~40 kg CO2e/m². No dedicated space-cell LCA source; minor contributor.

## Conversion and reference parameters
- AM0 irradiance **1366 W/m²** and assumed cell efficiency **30%** used for the per-kWp conversion (consistent with current space triple-junction datasheets).
- Sensitivity is dominated by two parameters, both exposed in the model: fab-grid carbon intensity (a clean hydro grid roughly thirds the substrate term; a coal-heavy grid nearly doubles it [grid range per Swart 2011]) and Ge wafer thickness.

## Framing points (with sources)
- Published space/concentrator LCAs report deceptively low values, e.g. **16.4–18.4 g CO2e/kWh** for four-junction HCPV [Payet & Greffe 2019], because optical concentration lets one wafer cover hundreds of times its area. A 1-sun orbital array receives no such benefit, so the full per-m² substrate burden applies.
- Payet & Greffe (2019) independently identify germanium as the single dominant driver of their resource-depletion impact, corroborating the substrate-dominated result here.
- For reference, terrestrial crystalline-silicon modules are roughly **420–810 kg CO2e/kWp** depending on design and production location [Reichel et al. 2022], and EPEAT low/ultra-low-carbon thresholds are 630 / 400 kg CO2e/kWp [Global Electronics Council 2023]. The space triple-junction cell is several times higher per kWp.

## Bibliography

1. Swart, P., Dewulf, J., Van Langenhove, H., Moonens, K., Dessein, K., Quaeyhaegens, C. (2011). Assessment of the overall resource consumption of germanium wafer production for high concentration photovoltaics. *Resources, Conservation and Recycling*, 55(12), 1119–1128. https://doi.org/10.1016/j.resconrec.2011.06.016

2. Mohr, N. J., Schermer, J. J., Huijbregts, M. A. J., Meijer, A., Reijnders, L. (2007). Life cycle assessment of thin-film GaAs and GaInP/GaAs solar modules. *Progress in Photovoltaics: Research and Applications*, 15(2), 163–179. https://doi.org/10.1002/pip.735

3. Mohr, N., Meijer, A., Huijbregts, M. A. J., Reijnders, L. (2009). Environmental impact of thin-film GaInP/GaAs and multicrystalline silicon solar modules produced with solar electricity. *The International Journal of Life Cycle Assessment*, 14(3), 225–235. https://doi.org/10.1007/s11367-009-0062-z

4. Payet, J., Greffe, T. (2019). Life Cycle Assessment of New High Concentration Photovoltaic (HCPV) Modules and Multi-Junction Cells. *Energies*, 12(15), 2916. https://doi.org/10.3390/en12152916

5. Reichel, C., Müller, A., et al. (Fraunhofer ISE) (2022). CO2 Emissions of Silicon Photovoltaic Modules: Impact of Module Design and Production Location. *8th World Conference on Photovoltaic Energy Conversion (WCPEC-8)*.

6. Global Electronics Council (2023). Criteria for the Assessment of Ultra-Low Carbon Solar Modules (EPEAT-ULCS-2023).

*Note: Items 3 and 4 are open access. Item 1 (Swart 2011) is the load-bearing citation for the dominant ~78% substrate term. Item 2 (Mohr 2007) underlies the epitaxy and processing terms and is paywalled (Wiley); its key parameters are also reported in the open-access follow-up, item 3.*
