# ODC Carbon Model: Data Provenance

Scope: every value used by the ODC side of the carbon model
(`odc_per_launch`, `odc_hardware_embodied`, `odc_carbon`, and the shared
`it_quantities`). Variable names refer to fields of `CarbonScenario`.

The ODC **launch** carbon is taken from two primary sources:

- **Dirty Bits** = Ohs, R., Stock, G. F., Schmidt, A., Fraire, J. A., & Hermanns, H. (2025).
  *Dirty Bits in Low-Earth Orbit: The Carbon Footprint of Launching Computers.*
  ACM SIGEnergy Energy Informatics Review 5(2). arXiv:2508.06250. doi:10.1145/3757892.3757896.
- **Miraux 2022** = Miraux, L., Wilson, A. R., & Dominguez Calabuig, G. J. (2022).
  *Environmental sustainability of future proposed space activities.*
  Acta Astronautica 200, 329-346. doi:10.1016/j.actaastro.2022.07.034.

---

## 1. Launch carbon (per reused delivery, tCO2e/launch)

Per-launch carbon = vehicle embodied (amortized over reuses) + propellant
production + combustion (+ optional re-entry).

| Parameter | Variable | Value | Units | Source | Notes |
|---|---|---|---|---|---|
| Super Heavy dry mass | `sh_dry_t` | 275 | t | Miraux 2022 / SpaceX vehicle spec | stainless-steel structure + engines |
| Starship dry mass | `ss_dry_t` | 120 | t | Miraux 2022 / SpaceX vehicle spec | stainless-steel structure + engines |
| Raptor count, booster | `n_raptor_sh` | 33 | - | SpaceX spec (via Dirty Bits) | |
| Raptor count, ship | `n_raptor_ss` | 6 | - | SpaceX spec (via Dirty Bits) | |
| Raptor engine mass | `raptor_mass_t` | 1.6 | t | SpaceX spec (via Dirty Bits) | used to split structure vs engine mass |
| Propellant per launch | `propellant_t` | 4600 | t | Miraux 2022 / SpaceX | methalox (LOx + LCH4) |
| Oxidizer/fuel ratio | `o_f_ratio` | 3.6 | - | Miraux 2022 | see note (a) |
| Payload to LEO | `payload_cap_t` | 100 | t | SpaceX / Dirty Bits | delivery basis for launch count |
| Booster reuses | `n_sh_reuse` | 100 | - | Dirty Bits (assumption) | amortizes booster embodied carbon |
| Ship reuses | `n_ss_reuse` | 25 | - | Dirty Bits (assumption) | amortizes ship embodied carbon |
| Stainless steel CI | `ci_steel` | 6.0 | kgCO2e/kg | see note (b) | applied to (dry mass - engine mass) |
| Engine alloy CI | `ci_engine` | 40.0 | kgCO2e/kg | see note (b) | Ni-superalloy, applied to engine mass |
| Methalox production CI | `ci_prop_prod` | 2.5 | kgCO2e/kg | Miraux 2022 / Dirty Bits | LOx + LCH4 production, see note (c) |
| Combustion per launch | `combustion_ea_t` | 4646 | tCO2e/launch | FAA Starship EA, see note (d) | ascent + static fires + landing |
| Re-entry (NOx) | (inline) | 13.2 | tCO2e per t payload | Dirty Bits | OFF by default (`include_reentry=False`), see note (e) |

### Notes

(a) The O/F ratio of 3.6 is stored but the per-launch *detail* plot splits
propellant 50/50 between LOx and LCH4 rather than 3.6:1. This affects only the
LOx/LCH4 breakdown bars, not the total (the production CI is applied to the full
`propellant_t`). Worth reconciling if the detail plot is used in the paper.

(b) Stainless-steel (6.0) and Ni-superalloy (40.0) carbon intensities are not
attributed to a specific reference in the model comments. They are consistent
with common materials-LCA values (worldsteel-class stainless, Ni-superalloy),
and are likely drawn from the Miraux launcher inventory. **Confirm and cite the
exact materials source before submission.**

(c) Propellant production at 2.5 kgCO2e/kg over 4600 t gives ~11,500 tCO2e per
launch, which is the single largest per-launch term, larger than combustion.
This makes the green-vs-grey methane assumption the dominant launch sensitivity;
it should be stated explicitly and defended.

(d) Combustion is labeled "FAA EA" in the code (FAA Final Programmatic
Environmental Assessment for the Starship/Super Heavy program, Boca Chica, 2022).
Since the intended launch provenance is Dirty Bits + Miraux, **reconcile whether
4646 tCO2e/launch comes directly from the FAA EA or is quoted via Dirty Bits /
Miraux**, and cite accordingly.

(e) Re-entry is currently disabled because the vehicle is reused, so per-delivery
re-entry is taken as ~0. See the flag in Section 3: this excludes high-altitude
NOx and alumina forcing that the literature argues can be large.

---

## 2. ODC hardware embodied (tCO2e)

| Parameter | Variable | Value | Units | Source |
|---|---|---|---|---|
| PV embodied | `pv_kgco2e_m2` | 1400 | kgCO2e/m^2 | Swart 2011 + Mohr 2007 (PV LCA), see note (f) |
| Radiator embodied | `rad_kgco2e_m2` | 86 | kgCO2e/m^2 | BaSO4 coating + IAI 2023 (aluminium substrate) |
| GPU baseboard PCF | `gpu_baseboard_kgco2e` | 2274 | kgCO2e/node | NVIDIA HGX B200 product carbon footprint |
| Host server PCF | `host_server_kgco2e` | 1300 | kgCO2e/node | NVIDIA HGX B200 PCF / server LCA |
| Node power | `node_power_kw` | 10.4 | kW/node | sets node count from capacity |
| IT method | (function) | - | - | Falk et al. 2025 (cradle-to-grave GPU LCA methodology) |

IT embodied is identical for ODC and TDC (apples-to-apples). PV and radiator
areal *masses* (for launch) are documented in the financial/sizing provenance;
here only the embodied carbon intensities appear.

(f) Swart 2011 and Mohr 2007 are cited in the model comments but without full
bibliographic details. **Complete these citations before submission** (triple-
junction GaAs space PV embodied carbon).

---

## 3. FLAG: ODC launch climate impact is likely a LOWER BOUND

The launch carbon above counts combustion CO2 (FAA EA) plus propellant
production and amortized vehicle embodied carbon. It does **not** include the
non-CO2, high-altitude radiative forcing of rocket exhaust: stratospheric and
mesospheric **water vapor**, **black carbon (soot)**, **NOx**, and **alumina**
from any demising hardware.

> Dominguez Calabuig, G. J., Wilson, A., Bi, S., Vassile, M., Sippel, M., &
> Tajmar, M. (2024). *Environmental life cycle assessment of reusable launch
> vehicle fleets: large climate impact driven by rocket exhaust emissions.*
> Acta Astronautica 221, 1-11. doi:10.1016/j.actaastro.2024.05.009.

This study reports that omitting high-altitude characterisation of rocket
exhaust and demised aluminium oxides can underestimate launch climate impact
by **two to three orders of magnitude**, and that **methalox** fleets are
markedly worse than hydrogen ones, driven largely by **black carbon**. Starship
burns methalox, so this concern applies directly to the ODC here. Miraux 2022
makes a consistent point: space-sector impacts become significant only once
high-altitude launch effects are included.

Implications for this model:

- The ODC carbon total reported here should be framed as a **conservative lower
  bound** on climate impact. The true figure, including non-CO2 high-altitude
  forcing, could be substantially higher.
- This shifts the ODC-vs-TDC comparison **against** ODC and moves the grid-
  intensity crossover toward higher grid intensities (a dirtier grid would be
  required before ODC "wins").
- A defensible treatment is a sensitivity case that scales launch forcing by the
  high-altitude multiplier from Dominguez Calabuig et al. 2024 (ideally on a
  GWP100 and GWP20 basis, since the methalox penalty grows on the 20-year
  horizon), reported alongside the CO2-only baseline.

---

## References

- Ohs, R., Stock, G. F., Schmidt, A., Fraire, J. A., & Hermanns, H. (2025). Dirty Bits in Low-Earth Orbit: The Carbon Footprint of Launching Computers. ACM SIGEnergy Energy Informatics Review 5(2). arXiv:2508.06250. doi:10.1145/3757892.3757896.
- Miraux, L., Wilson, A. R., & Dominguez Calabuig, G. J. (2022). Environmental sustainability of future proposed space activities. Acta Astronautica 200, 329-346. doi:10.1016/j.actaastro.2022.07.034.
- Dominguez Calabuig, G. J., Wilson, A., Bi, S., Vassile, M., Sippel, M., & Tajmar, M. (2024). Environmental life cycle assessment of reusable launch vehicle fleets: large climate impact driven by rocket exhaust emissions. Acta Astronautica 221, 1-11. doi:10.1016/j.actaastro.2024.05.009.
- Falk, S., Ekchajzer, D., Pirson, T., Lees-Perasso, E., Wattiez, A., Biber-Freudenberger, L., Luccioni, S., & van Wynsberghe, A. (2025). More than Carbon: Cradle-to-Grave environmental impacts of GenAI training on the Nvidia A100 GPU. arXiv:2509.00093.
- NVIDIA (2025). NVIDIA HGX B200 Reduces Embodied Carbon Emissions Intensity (Product Carbon Footprint summary). NVIDIA Technical Blog.
- International Aluminium Institute (2023). Life cycle inventory / GHG intensity of primary aluminium.
- FAA (2022). Final Programmatic Environmental Assessment for the SpaceX Starship/Super Heavy Launch Vehicle Program, Boca Chica, TX.
- Swart (2011); Mohr (2007). PV life-cycle assessment (full citations to be completed).

## Open items to confirm before submission

1. Reconcile the 4646 tCO2e/launch combustion figure: direct from FAA EA, or via Dirty Bits / Miraux?
2. Cite the exact source for stainless-steel (6.0) and Ni-superalloy (40.0) carbon intensities.
3. Complete the Swart 2011 and Mohr 2007 PV citations.
4. Confirm the NVIDIA HGX B200 PCF node figures (2274, 1300) against the published PCF summary.
5. Decide whether to add a high-altitude-forcing sensitivity case per Dominguez Calabuig et al. 2024.
6. Fix or document the 50/50 LOx/LCH4 detail split vs the 3.6 O/F ratio.
