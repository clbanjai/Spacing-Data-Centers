# ODC Model — Checklist Resolutions and Source Justifications

**Date:** 2026-06-20
**Scope:** Five open items from the prior review, resolved in `odc_lcoc_v3.ipynb` (notebook re-run end-to-end; all cells execute). Each item below states the decision, the implementation, and the defensible sources. Net effect on the headline result is small: the 2026 baseline ODC LCOC is **$0.199/EFLOP**, ODC/TDC ≈ **10.2×** at 2026, closing to ≈2× by 2045 — unchanged in character from before. Launch (57%) and hardware (43%) still dominate; radiator, data, and labor are each ≤0.1%.

---

## Item 1 — Radiator coating material: BaSO₄ is the right choice

**Decision:** Keep a BaSO₄-class radiative-cooling coating (α_solar = 0.05, ε_IR = 0.95). No change to optical values.

Against the five criteria you set (high IR emissivity, low solar absorption, cheap, lightweight, mature), BaSO₄ is the best-supported option in the Fan et al. (2025) review and the wider literature:

- **High IR emissivity:** Fan et al. report monolayer BaSO₄ paints/films combining "ultralow solar absorptance with a high infrared emissivity of 0.96." The Purdue ultrawhite formulation independently measures a sky-window emissivity of ~0.95. Our 0.95 is conservative.
- **Low solar absorption:** Fan et al. note BaSO₄ films reaching 99.6% solar reflectance; the Purdue paint reaches 98.1%. Both correspond to α_solar ≈ 0.02–0.05; our 0.05 is again conservative.
- **Physical basis:** BaSO₄ has a high electronic band gap (→ low solar absorptance) and a phonon resonance near 9 µm sitting in the atmospheric/space IR window (→ high emissivity). This is a material-intrinsic property, not a fragile photonic structure.
- **Cheap & common:** BaSO₄ (barite) is an abundant, low-cost mineral pigment already mass-produced for paints. It is far cheaper than metallized optical-solar-reflector (OSR) tiles or silvered Teflon.
- **Lightweight & mature:** It is applied as a thin paint coat rather than a heavy multilayer stack. The Purdue paint holds a Guinness World Record ("whitest paint") and is a simple inorganic coating; Fan et al. classify BaSO₄ among the most mature static RC coatings.

**Why not the alternatives (per Fan et al.):** early inorganic white coatings (TiO₂, ZnO) suffer UV-driven photocatalytic darkening and α_solar increase on orbit; polymer films (PDMS, PMMA) are less mature and more vulnerable to atomic-oxygen/UV erosion; OSR/silvered-Teflon are heavier and costlier. BaSO₄ wins on all five axes simultaneously.

**Caveat to flag in the paper:** the model assumes **no coating degradation** (no UV/AO/dust). On-orbit α_solar can drift upward over years; since the radiator term is negligible to LCOC this does not affect conclusions, but it is worth one sentence of acknowledgment.

---

## Item 2 — Radiator thickness and costing: move from a flat $/m² to a mass-grounded build-up

**Your concern was right.** A free-floating $/m² is weak precisely because the thermal balance is 2-D and carries no thickness. The fix is to cost the radiator as the physical object it is — a thin **aluminum panel** carrying a thin coating — and build $/m² from a real **areal mass × $/kg**. Thermal sizing stays 2-D (the panel is effectively isothermal and thin); only **mass and cost** gain a thickness basis.

**What I changed (notebook, Cell 5 + Cell 9):**

| Quantity | Old | New | Basis |
|---|---|---|---|
| Radiator areal mass | 4.5 kg/m² (1 mm BaSO₄ only) | **5.5 kg/m²** | 5.0 Al structure + 0.5 BaSO₄ coat |
| Radiator areal cost | $600/m² (flat) | **~$230/m²** | 5.0 kg/m² × ~$40/kg (fabricated Al) + ~$30/m² coat |

The 5.0 kg/m² aluminum figure is the midpoint of real space-radiator practice: NASA Shuttle radiator panels use 0.28 mm 2024-T81 aluminum facesheets over 5056 aluminum honeycomb cores (13–23 mm thick); Advanced Cooling Technologies' honeycomb-panel heat pipes add minimal mass for large thermal spreading; state-of-the-art lightweight heat-pipe radiators target ≤3 kg/m², while conventional large radiator systems run ~10 kg/m². A robust deployable panel with embedded **axially-grooved aluminum heat pipes** (your own literature: "Heat Pipes for Space Applications — Axial Grooved Heat Pipes") plus facesheets and honeycomb lands naturally at ~4–5 kg/m². This is also consistent with the radiator embodied-carbon memo's "Reading B" aluminum bill-of-materials (~4.7 kg/m²).

**Is "heat pipes + structure are just aluminum" okay?** Yes — it is the standard first-order assumption. Real space radiators are overwhelmingly aluminum: aluminum facesheets, aluminum honeycomb core, and extruded/axially-grooved aluminum heat pipes (typically with an ammonia working fluid of negligible mass). Lumping the heat-pipe network, core, and facesheets into one effective aluminum areal mass and costing it as fabricated aluminum is defensible and keeps complexity low. The $40/kg figure = LME primary aluminum (~$3.6/kg, June 2026) × ~10× for rolling/extrusion/honeycomb bonding/heat-pipe fabrication/qualification, appropriate for a mass-produced (Starship-scale) panel.

**What did Turyshev/JPL do?** Honest answer: the Turyshev solar-gravitational-lens work is **not** a useful $/m² radiator-cost benchmark. That spacecraft is a ~1-m sailcraft whose thermal design either uses **large deployable radiators for electric propulsion** or, in the radioisotope variant, uses **RTG waste heat for thermal balance and needs no dedicated radiators at all**. It gives architecture, not panel costs. The defensible cost anchors are the NASA Shuttle/ISS radiators and ACT honeycomb-heat-pipe panels cited above, not the SGL mission.

**Materiality:** radiator cost is ~0.1% of LCOC, so even a 2–3× swing in $/m² is immaterial to every conclusion. The value of the change is defensibility, not magnitude.

---

## Item 3 — Radiator spine length from PV area

**Decision:** The spine length is no longer a free 300 m input. The PV array is treated as a **square**, so its side length is the spine length: **L = √A_PV**.

**Implementation (Cell 9, `size_radiator`):** `L = np.sqrt(A_PV)` is computed each call (so it tracks PV efficiency improvements over the mission), giving L ≈ **520 m** at the 2026 baseline (A_PV ≈ 270,130 m²). This matches the assumptions doc's "~519 × 519 m" array. With a square array, the half-height H = A_PV/(2L) = L/2, so the aspect ratio h = 0.5 — a clean, self-consistent geometry. `L_rad` was removed from the tornado charts since it is now a derived quantity, not an independent parameter.

---

## Item 4 — Orbit: 2,000 km LEO, eclipse-free

**Decisions implemented:** altitude **2,000 km** (was documented as 15,000 km MEO); Earth view factor **F_⊕ = 0.1757** (was 0.10); capacity factor **CF = 0.9987** applied to annual ODC compute; eclipse model = continuous sunlight with a residual lunar-eclipse loss.

- **Why LEO / 2,000 km:** Starship's baseline reusable capacity is 100+ t to **LEO** (official Payload User's Guide; ~150 t cited for later blocks), and essentially all near-term Starship missions target LEO — so a LEO ODC matches the launch reality. 2,000 km is the **upper edge of LEO** and sits just above the **dawn-dusk sun-synchronous eclipse-free floor (~1,600 km)**. Dawn-dusk orbits ride the terminator and never enter Earth's shadow; the ~1,600 km lower bound is the standard altitude that minimizes radiation dose and latency while still staying out of eclipse. So 2,000 km is well-justified as "the lowest practical altitude that gives a continuously-sunlit, eclipse-free orbit," which is exactly the design intent.
- **F_⊕ = 0.1757 — independently verified.** The geometric view factor of a downward-facing flat radiator is F = ½·(1 − √(1 − (R_⊕/(R_⊕+h))²)). With R_⊕ = 6,371 km and h = 2,000 km this evaluates to **0.1757** (reproduced exactly in the notebook). The previous 0.10 is replaced.
- **CF = 0.9987.** Applied as an availability factor on annual compute (`k_load = CAPACITY_FACTOR`). One note for the record: 0.9987 implies ~11.4 h/yr of downtime, not literally 1 h — if the intended figure is purely the lunar-eclipse contribution, double-check whether CF should be ~0.9999 (1 h) or whether ~11 h/yr aggregates several partial lunar occultations. I left it at your stated 0.9987; it moves LCOC by ~0.1%.

---

## Item 5 — Simple labor model for ODC and TDC

**Decision:** Add one fixed annual labor cost — an **on-Earth operations workforce** — applied **identically** to the ODC and the terrestrial baseline. Rationale: even an orbital data center is run from a ground network-operations center, so it carries the same human-operations cost a terrestrial facility does; making it identical keeps the comparison fair and removes the prior "ODC has zero OPEX" overstatement.

**Value: $3.75 M/yr = 25 FTE × $150k loaded.** A highly-automated hyperscale campus (>100 MW) runs on ~20–30 permanent operations staff per 100 MW; loaded compensation (salary + benefits + overhead) for skilled DC technicians/engineers lands around $130–150k. 25 × $150k ≈ $3.75 M/yr is a defensible midpoint. On a per-kW basis this is **$37.5/kW/yr**.

**Implementation:**
- ODC (Cell 11): a fixed `labor` cash-flow stream of $3.75 M/yr enters the NPV and appears as a new "Labor" line in the breakdown (~0.1% of LCOC).
- TDC (Cell 14): the same $37.5/kW/yr is added on top of Nøland's 10% O&M. The paper's labor-free baseline is preserved as the verification case, which still reproduces **$1.163/EFLOP** at γ = 0.1 (paper: $1.16).
- Note a small modeling overlap to acknowledge: Nøland's d = 10%-of-CAPEX O&M may already embed some labor, so the explicit TDC line is mildly conservative (double-counts slightly). Because it is identical on both sides and ~0.1% of LCOC, it does not change the comparison.

---

## Sources

**Item 1 — BaSO₄ coating**
- Fan, Y. et al. (2025). *Radiative Cooling in Outer Space: Fundamentals, Advances in Materials and Applications, and Perspectives.* Advanced Materials 37, e06795. (Local: `Literature 2/Advanced Materials - 2025 - Fan ...pdf`)
- Li, X., Peoples, J., Yao, P., Ruan, X. (2021). *Ultrawhite BaSO₄ Paints and Films for Remarkable Daytime Subambient Radiative Cooling.* ACS Applied Materials & Interfaces. https://pubs.acs.org/doi/10.1021/acsami.1c02368
- Purdue ultrawhite paint (Guinness record; 98.1% reflectance, ε≈0.95). https://www.greencarcongress.com/2021/10/20211020-ruan.html
- *Electronic and phononic origins of BaSO₄ as an ultra-efficient radiative cooling pigment.* ScienceDirect. https://www.sciencedirect.com/science/article/abs/pii/S2542529322000566

**Item 2 — Radiator structure & costing**
- NASA NTRS, *Honeycomb panel heat pipe development for space radiators* / *Advanced radiator concepts utilizing honeycomb panel heat pipes.* https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19880003365.pdf
- Advanced Cooling Technologies, *Thermally Enhanced Honeycomb Panels for Spacecraft.* https://www.1-act.com/resources/blog/thermally-enhanced-honeycomb-panels-for-spacecraft/
- ISNPS, *Advanced Lightweight Heat Rejection Radiators for Space* (≤3 kg/m² target; conventional ~10 kg/m²). https://isnps.unm.edu/reports/ISNPS_Tech_Report_103.pdf
- Shuttle radiator construction (0.28 mm 2024-T81 facesheets, 5056 Al honeycomb), AIAA JSR pumped-loop radiator. https://arc.aiaa.org/doi/10.2514/1.A35030
- LME aluminum price ~$3.6/kg (June 2026). https://tradingeconomics.com/commodity/aluminum ; https://aluminiummagazine.com/mag/business/current-aluminium-price.html
- Turyshev SGL mission architecture (radiators vs RTG thermal balance). https://arc.aiaa.org/doi/10.2514/1.A35493 ; https://www.centauri-dreams.org/2020/12/16/jpl-work-on-a-gravitational-lensing-mission/
- Local: `Literature 2/Heat Pipes for Space Applications Part 1 - Axial Grooved Heat pipes.pdf`; `radiator_embodied.md` (Reading B Al BOM).

**Item 4 — Orbit**
- SpaceX Starship payload to LEO (100+ t baseline). https://www.eoportal.org/other-space-activities/starship-of-spacex ; https://en.wikipedia.org/wiki/SpaceX_Starship_(spacecraft)
- Dawn-dusk sun-synchronous eclipse-free orbits (~1,600 km floor; continuous sunlight). https://handwiki.org/wiki/Astronomy:Sun-synchronous_orbit ; https://www.esa.int/ESA_Multimedia/Images/2020/03/Polar_and_Sun-synchronous_orbit
- Tether-based architecture for solar-powered orbital AI data centers (eclipse-free orbit context). https://arxiv.org/pdf/2512.09044

**Item 5 — Labor**
- Hyperscale 100 MW staffing (~20–30 FTE/100 MW automated). https://www.datacenterknowledge.com/data-center-career-development/data-center-staffing-what-drives-on-site-headcount ; https://broadstaffglobal.com/data-center-staffing-levels-how-many-people-does-a-facility-need
- DC labor cost / compensation benchmarks. https://thenetworkinstallers.com/blog/data-center-operating-costs/ ; https://encoradvisors.com/data-center-cost/
- LCOC methodology baseline: Nøland et al. (2024), *Will Energy-Hungry AI Create a Baseload Power Demand Boom?* (Local: `Literature 2/LCOC Paper ...pdf`)
