# ODC Model v3 — Decision & Research Memo (pre-implementation)

**Date:** 2026-06-20
**Author:** prepared for FPS
**Scope:** Four workstreams requested for the publication-bound `odc_lcoc_v3.ipynb`: (1) radiator operating-temperature assumption, (2) parameter/source audit + literature renaming, (3) learning-rate rationale and sourcing, (4) data-center demand growth + ODC market-share figure. **This memo is for review *before* any notebook edits.** Every proposed change is staged with its rationale, source, and a quantified impact where I could compute one against the live model.

**Confirmed decisions (your answers):** baseline `T_rad = 70 °C`, sweep `[60, 70, 80]`; **hybrid** learning framing (keep `%/yr` time series, add Wright's-law `%/doubling` for launch + the market-share figure); two-panel demand figure; memo-first delivery.

**Headline does not move.** I re-ran the live model for the T_rad change: 2026 ODC LCOC goes from **$0.1990 → $0.1996/EFLOP** (+0.27 %) and ODC/TDC stays at **10.2×**. Launch (≈57 %) and hardware (≈43 %) still dominate; radiator/data/labor remain ≤0.1 % each. The value of this round is *defensibility for reviewers*, not changing the result.

---

## Task 1 — Radiator operating temperature

### The problem with the current assumption
The notebook currently fixes chip = 85 °C, applies a **5 °C** chip→radiator drop, and operates the radiator at **80 °C**. As you noted, a 5 °C budget only represents the **near-isothermal ΔT of the two-phase heat-pipe fluid** (evaporator↔condenser). It omits the rest of the series thermal path that any real liquid-cooled rack carries:

junction → die TIM → integrated heat spreader → TIM → cold plate → coolant film → coolant bulk → manifold/transport → heat-pipe evaporator → (small fluid ΔT) → condenser → panel conduction → radiating surface.

Each interface and convective film adds resistance. A realistic **total** junction-to-radiator drop is on the order of **~20–25 °C**, not 5 °C.

### Why fixing T_rad (rather than modeling the resistance network) is the right call
1. **The literature you already hold supports a fixed-temperature treatment.** Turyshev (2026, `Prior ODC Papers/turyshev2026.pdf`) operates the radiator at **350 K = 76.85 °C ≈ 77 °C** — and explicitly states "Radiator temperature `T_rad` is *not freely selectable*: it is upper-bounded by allowable junction temperature `T_j` and by the total thermal resistance from junction to radiator. A thermal-resistance budget is [needed]." In other words, the most rigorous ODC paper in your set treats `T_rad` as an *architecture-level design variable bounded by a resistance budget* — exactly the approach proposed here — rather than resolving every resistance.
2. **Junction temperatures bracket the choice.** Li et al. (2024, `li2024 chips.pdf`) is a chip-scale thermal-management review centered on the junction-temperature limit; data-center accelerators run junctions roughly **60–100 °C** (e.g., NVIDIA GPUs throttle in the ~83–90 °C band). With a ~20–25 °C path drop, a junction of ~90–95 °C lands the radiator near **70 °C** — consistent with Turyshev's 77 °C and comfortably inside the design space.
3. **Modeling the full network would add many poorly-constrained parameters** (TIM conductances, cold-plate HTCs, flow rates, manifold geometry) for a subsystem that is **~0.1 % of LCOC**. That is a poor accuracy-for-complexity trade for a paper whose conclusion is launch/hardware-driven.

### Recommended treatment
- **Baseline `T_rad = 70 °C`** (≈ midpoint of a defensible junction range with a realistic full-path drop; one notch below Turyshev's 77 °C, so conservative on radiator size).
- **Sensitivity sweep `T_rad ∈ {60, 70, 80} °C`**, added as a row to the tornado/one-at-a-time sensitivity.
- **One sentence of explicit scope language** in §1: "We do not resolve the junction-to-radiator thermal-resistance network; we instead fix the radiator operating temperature as a bounded design variable (cf. Turyshev 2026), and test sensitivity over 60–80 °C spanning plausible junction limits (Li 2024) and the full conduction/convection path drop."

### Quantified impact (live model run, 2026 baseline)

| T_rad | Radiator area (m²) | Radiator mass (t) | Total system mass (t) | ODC LCOC ($/EFLOP) | Δ vs 80 °C | ODC/TDC |
|---|---|---|---|---|---|---|
| 80 °C (current) | 72,450 | 398 | 6,374 | 0.1990 | — | 10.17× |
| 77 °C (Turyshev) | 75,315 | 414 | 6,390 | 0.1992 | +0.08 % | 10.17× |
| **70 °C (proposed)** | **82,567** | **454** | **6,430** | **0.1996** | **+0.27 %** | **10.19×** |
| 60 °C | 94,487 | 520 | 6,495 | 0.2002 | +0.59 % | 10.23× |

Radiator area scales ≈ `T_rad⁻⁴` so it grows ~14 % going 80→70 °C, but because radiator mass is a small slice of the launched total and radiator cost is ~0.1 % of LCOC, the headline barely moves. **Conclusion: adopt 70 °C baseline + sweep; it strengthens the physics with no material cost penalty.**

### Notebook edits this implies (staged, not yet applied)
- §1 markdown (cell 2): replace "Chip temperature 85 °C with a 5 °C chip-to-radiator drop → radiator at 80 °C" with the bounded-design-variable language + scope sentence above.
- §2b table + §2c (cell 4): change `T_rad` row to 70 °C, cite Turyshev 2026 (350 K) and Li 2024 (junction range); note the omitted resistance network as an explicit limitation.
- Code (cell 5): `T_rad_C = 70.0` (and decouple `T_chip_C`, which is now only narrative).
- Tornado (cells 22/23): add a `T_rad ∈ {60,80}` row swung around the 70 °C baseline.

---

## Task 2 — Parameter & source audit

**Method:** every fixed and learning parameter cross-checked against (a) the value in code (cell 3/5), (b) the markdown tables (cell 4), (c) `references.bib`, and (d) an actual file in `Literature 2/`. Files identified by extracting their first page.

### 2.1 Audit table

| Parameter | Value (2026) | Source file in `Literature 2/` | bib key | Status |
|---|---|---|---|---|
| Compute efficiency γ | 6.0 PFLOPS/kW | `LCOC Paper - Will_Energy-Hungry…pdf` | `noland2024energy` | ✅ in folder |
| Launch cost | 1,500 $/kg | `Cost for Space Launch…CSIS.pdf` | `roberts2022space` | ✅ in folder |
| PV BOL efficiency | 0.32 | `Space PV - NASA.pdf` | `nasa2025smallspacecraft` | ✅ in folder |
| PV areal cost | 33,000 $/m² | `PVs Alibaba.pdf` | `alibaba_pv_gaas` | ✅ in folder |
| Radiator areal cost | ~230 $/m² | **derived** (Al $/kg + coat) | — | ⚠️ no primary file (see 2.3) |
| IT cost density | 23 M$/MW | `DC Rack 3.pdf` (NVIDIA GB200) + `DC Rack 4.pdf` (SemiAnalysis TCO) | `nvidia_gb200`, `patel2025h100` | ✅ in folder |
| PV derating | 0.85 | NREL PVWatts (web tool) | `nrel_pvwatts` | ⚠️ online only |
| PV areal mass | 1.76 kg/m² | `PV GaAs TJ.pdf` | `spectrolab_pv` | ✅ in folder |
| Radiator areal mass | 5.5 kg/m² | **derived** (5.0 Al + 0.5 coat) | — | ⚠️ no primary file (see 2.3) |
| IT mass density | 55,000 kg/MW | `DC Rack.pdf` (AFCOM density) + `DC Rack 2.pdf` (Patrizio "Heavy Compute") | `kleyman2026density`, `patrizio2025heavy` | ✅ in folder |
| PV operating temp T_PV | **80 °C (md) / 31.1 °C (code)** ⚠️ | `PV operating temp.pdf` | — | ❗ inconsistent (see 2.2) |
| Radiator α_solar | 0.05 | `Advanced Materials - 2025 - Fan…pdf` | `fan2025radiative` | ✅ in folder |
| Radiator ε_IR / α_IR | 0.95 | `Advanced Materials - 2025 - Fan…pdf` | `fan2025radiative` | ✅ in folder |
| PV backside ε | 0.84 | `PV GaAs TJ.pdf` | `spectrolab_pv` | ✅ in folder |
| Discount rate r | 0.10 | `LCOC Paper…pdf` | `noland2024energy` | ✅ in folder |
| Earth view factor F_⊕ | 0.1757 | **computed** (geometry @ 2,000 km) | — | ✅ derived, no file needed |
| PV view factor F_PV | solved | `PV view factor.pdf` | `martinez2026viewfactors` | ✅ in folder |
| Radiator temp T_rad | 80→**70** | `turyshev2026.pdf` + `li2024 chips.pdf` | (add `turyshev2026`) | ✅ in folder (bib entry to add) |
| Solar constant I_⊙ | 1,361 W/m² | NASA TSI (standard) | — | ⚠️ online only (standard constant) |
| Earth albedo / T_eff | 0.306 / 254.15 K | NASA (standard) | — | ⚠️ online only (standard constants) |
| Orbital altitude / CF | 2,000 km / 0.9987 | own OREKIT work | `orekit`, `eclipse_rickman` | ✅ internal + folder |
| Annual labor | 3.75 M$/yr | staffing benchmarks (web) | — | ⚠️ online only |
| Heat-pipe basis | qualitative | `Heat Pipes…Axial Grooved.pdf`, `heat pipes 2.pdf` | `semenov_heatpipes` | ✅ in folder |

### 2.2 Inconsistencies to resolve (please confirm)
1. **`T_PV` mismatch — most important.** Cell 4 markdown lists PV operating temperature = **80 °C**; the `Scenario` dataclass (cell 5) uses **`T_PV_C = 31.1`**. These differ by ~50 °C and `T_PV` feeds the radiator heat balance (PV backside IR re-absorbed by the radiator, `∝ T_PV⁴`). One of them is stale. My read: 31.1 °C looks like a computed steady-state cell temperature, while 80 °C is a legacy worst-case. **Which is intended?** I recommend fixing the markdown to match the code (31.1 °C) and citing `PV operating temp.pdf`, or, if 80 °C is intended, updating the code — but they must agree before submission.
2. **`arm_angle_deg = 45`** is carried in the dataclass but labeled "legacy; not used in horizontal-beam model." Recommend deleting to avoid a reviewer asking what it does.
3. **`F_PV = 0.617`** is stored as a constant but recomputed by `size_radiator`; the comment already says "legacy reference." Recommend dropping the stored value or clearly marking it non-functional.
4. **`T_chip` (85 °C)** becomes narrative-only once T_rad is fixed directly (Task 1). Keep it as context, not as a driver.

### 2.3 Sources to download (gaps to close before submission)
These parameters currently rest on web links (in `checklist_resolutions_memo.md` / `radiator_embodied.md`) with **no PDF in the folder**:

- **Radiator areal mass (5.5 kg/m²) and cost (~$230/m²)** — the whole mass-grounded build-up. Recommend pulling and filing: (a) NASA NTRS *Advanced radiator concepts utilizing honeycomb panel heat pipes* (ntrs.nasa.gov), (b) Advanced Cooling Technologies *Thermally Enhanced Honeycomb Panels for Spacecraft*, (c) ISNPS *Advanced Lightweight Heat Rejection Radiators for Space* (≤3 kg/m² SOA vs ~10 kg/m² conventional), (d) the Shuttle radiator AIAA JSR paper (2024-T81 facesheets/5056 honeycomb), (e) an LME aluminum price snapshot (~$3.6/kg, Jun 2026). At least (a)–(c) should be on disk.
- **NREL PVWatts derating (0.85)** — file the PVWatts technical reference (NREL/TP-6A20-62641) rather than citing the live calculator.
- **Solar constant / Earth albedo / effective temp (1361 W/m², 0.306, 254 K)** — one NASA Earth/Sun fact sheet covers all three; currently cited as "Earth Effective Temp source" with no bib entry. Low risk (textbook constants) but add a citable reference.
- **Labor (3.75 M$/yr)** — file one Uptime/Datacenter-Knowledge staffing benchmark for the 20–30 FTE/100 MW figure.
- **Learning-rate / demand sources (Tasks 3–4)** — see those sections; none of Epoch AI, Our World in Data, Goldman, IEA, McKinsey are currently on disk.

### 2.4 Proposed literature renaming (descriptive, notebook-consistent)
Filenames → `AuthorYear_Topic_Venue.pdf`, matching the parameter names used in the notebook:

| Current | Proposed |
|---|---|
| `Advanced Materials - 2025 - Fan - …(1).pdf` | `Fan2025_RadiativeCoolingOuterSpace_AdvMater.pdf` |
| `Cost for Space Launch…Aerospace Security Project.pdf` | `Roberts2022_SpaceLaunchCostToLEO_CSIS.pdf` |
| `LCOC Paper - Will_Energy-Hungry…pdf` | `Noland2024_AIBaseloadPowerDemand_IEEEAccess.pdf` |
| `li2024 chips.pdf` | `Li2024_ChipScaleThermalManagement_ApplThermEng.pdf` |
| `Space PV - NASA.pdf` | `NASA2025_SmallSpacecraftPower_TripleJunctionPV.pdf` |
| `PVs Alibaba.pdf` | `Alibaba_GaInPGaAsGe_TripleJunctionCell_Pricing.pdf` |
| `PV GaAs TJ.pdf` | `Spectrolab_TripleJunctionGaAs_Datasheet.pdf` |
| `PV operating temp.pdf` | `PV_CellOperatingTemperature.pdf` |
| `PV view factor.pdf` | `Martinez_RadiativeViewFactors.pdf` |
| `Heat Pipes for Space…Axial Grooved…Rev4.pdf` | `Semenov2020_AxialGroovedHeatPipes_NASA.pdf` |
| `heat pipes 2.pdf` | `Semenov2020_HeatPipesSpaceApps_Part2.pdf` *(verify it is part 2)* |
| `DC Rack.pdf` | `Kleyman2026_DataCenterDensityDilemma_AFCOM.pdf` |
| `DC Rack 2.pdf` | `Patrizio2025_HeavyCompute_AIWeightProblem_DCK.pdf` |
| `DC Rack 3.pdf` | `NVIDIA_DGX_GB200_RackScaleSystems_UserGuide.pdf` |
| `DC Rack 4.pdf` | `Patel2025_H100vsGB200_TCO_SemiAnalysis.pdf` |
| `NVIDIALa.pdf` | `NVIDIA2026_SpaceComputing_Announcement.pdf` |
| `starcloud wp.pdf` | `Starcloud2024_WhyTrainAIinSpace_WhitePaper.pdf` |
| `Google Space DC.pdf` | `Google2025_SpaceBasedAIInfrastructure_TechReport.pdf` |
| `odc 45 deg arm.pdf` | `Zorpette2026_OrbitalDataCenters_IEEESpectrum.pdf` *(verify)* |
| `time to train llm.pdf` | `Manghani2024_TestTimeCompute_Medium.pdf` |
| `Eclipse Rickman…pdf` | `Rickman1995_UmbraPenumbra_NASA_TP3547.pdf` |
| WSJ / Bloomberg / NVIDIA news PDFs | keep, prefix `News_` |

I have **not** renamed anything yet (renames touch git history and any in-notebook references). On your go-ahead I'll rename the files and update the `Source` columns in cell 4 so each parameter points to its exact filename.

---

## Task 3 — Learning rates: rationale & sources

### 3.1 `%/yr` vs `%/doubling` — the framing, resolved (hybrid)
- **`%/yr` (time-based):** `cost(t) = cost₀·(1−r)ᵗ`. Transparent, what you have now. Weakness: a constant calendar rate has no causal mechanism — it implicitly assumes a fixed deployment cadence.
- **`%/doubling` (Wright's law):** `cost(C) = cost₀·(C/C₀)^b`, with learning rate `LR = 1 − 2^b`. Causal: cost falls with **cumulative production/capacity** `C`. This is how launch, PV, and compute costs are actually reported in the literature — and it is the **only** framing that lets Task 4 connect "ODC wins more market share → more cumulative ODC capacity → faster cost decline."

**Hybrid plan (your choice):**
1. Keep the existing `%/yr` trajectories as the primary time-series (continuity with prior figures).
2. For each cost parameter, **document the implied `%/doubling`** so reviewers see the mechanism, using a stated doubling cadence.
3. In Task 4, drive launch + PV + IT costs by **Wright's law on cumulative ODC capacity** as a function of market share — this is where `%/doubling` does real work.

Conversion: at an assumed doubling time `τ` (years per capacity doubling), `r_annual = 1 − (1−LR)^(1/τ)`. E.g. `LR = 15 %/doubling` at `τ = 2 yr` → `r ≈ 7.8 %/yr`; at `τ = 1.5 yr` → `≈ 10.4 %/yr`.

### 3.2 Recommended rates, per parameter

| Parameter | Recommended `%/doubling` (LR) | Implied baseline `%/yr` | Current model `%/yr` (base) | Source & rationale |
|---|---|---|---|---|
| **Launch cost** | 15 % (base), 25 % opt, 10 % pess | ~8 %/yr @ τ≈2 yr | −12 % | Conservative aerospace progress ratio ≈85 % = **15 %/doubling**; HBR/satellite-launch analyses ≈**20 %/doubling**. Anchor on CSIS (`Roberts2022…CSIS`) for the $/kg level. **The reddit "35–50 %/doubling" is not citable** — that figure is the Li-ion battery learning rate, not launch; drop it. *Note:* the current −12 %/yr is a touch aggressive vs 15 %/doubling at a 2-yr cadence; either justify a faster cadence (Starship ramp) or trim to ~−8 to −10 %/yr. |
| **PV areal cost** | 20 % (Swanson's law) | ~6–9 %/yr | −6 % | **Swanson's law = 20 %/doubling**, four decades stable (Our World in Data; you cited it). Caveat: space-grade III-V is a far smaller, less mature market than terrestrial Si, so the *realized* rate is slower → keep baseline near the lower end. Current −6 %/yr is reasonable. |
| **Radiator areal cost** | ~5 % (≈ none) | ~1–3 %/yr | −3 % | Panel is fabricated **aluminum** (commodity, mature) + commodity BaSO₄ paint — **no Wright's-law tailwind**; aluminum tracks LME, not an experience curve. Recommend explicitly stating "≈ no learning" and keeping a token −1 to −3 %/yr. Immaterial to LCOC (~0.1 %) so low stakes. |
| **Compute efficiency γ** (PFLOPS/kW = energy efficiency) | growth | **+20 %/yr base** (was +15) | +15 % | Epoch AI: **leading ML hardware energy efficiency +40 %/yr** (doubling ≈2 yr). γ is exactly FLOP/s-per-W. The model's +15 % is *conservative*; I'd raise baseline to ~+20 %/yr citing Epoch and arguing for a slowdown toward physical limits (so not the full 40 %). Keep opt +22 %, pess +7 %. |
| **IT cost** ($/MW) | see note | — | −8 % | ⚠️ **Double-count risk.** γ already captures FLOP-per-watt gains; if `it_cost_per_MW` *also* declines fast, $/FLOP improves twice. Epoch GPU **price-performance ≈ +30 %/yr (FLOP/$)** — but most of that is delivered through γ. Recommend `it_cost_per_MW` carry only the *residual* $-per-MW-capacity decline (~−3 to −5 %/yr), and state the split explicitly. Please confirm intent. |
| **PV efficiency** (pp/yr) | additive | +0.3 pp/yr base, **cap ~35 % AM0** | +0.3 pp, cap 40 % | Commercial space 3J ≈30–32 % AM0; lab record **34.2 % AM0** (Joule 2022, 39.5 % terrestrial). Practical 3J ceiling is mid-30s % AM0, not 40 %. Recommend **cap at ~35 % AM0** (40 % is a terrestrial/theoretical figure) and keep +0.3 pp/yr. |
| **IT mass density** (kg/MW) | ~none | 0 to −1 %/yr | fixed | Trend ambiguous: rising power density packs more compute per kg (downward), but AI cooling/power gear is heavier (Patrizio "Heavy Compute," upward). Recommend **keep fixed** (or token −1 %/yr) and cite AFCOM/Patrizio; flag as low-confidence. |
| **Terrestrial CAPEX** ($/kW) | — | −2 %/yr | −2 % | Mature build-out; fine as is. |

### 3.3 Two specific things to fix in the notebook narrative
- The γ note says "Historical AI hardware ~25 %/yr but slowing." Replace with the **Epoch 40 %/yr energy-efficiency** anchor + your slowdown argument, so the number is sourced.
- Add the **double-count caveat** between γ and `it_cost_per_MW` (3.2 note) — a reviewer will spot it otherwise.

---

## Task 4 — Data-center demand growth & ODC market-share figure

### 4.1 Demand projections (credible anchors)

| Source | Metric | Figure |
|---|---|---|
| **Goldman Sachs** (2025) | Global DC power vs 2023 | **+50 % by 2027, +165 % by 2030**; ~84 GW by 2027; AI → 27 % of DC power by 2027 |
| **IEA** *Energy and AI* (2025) | Global DC electricity | **~945 TWh by 2030** (≈ doubles from 2024); **~15 %/yr** 2024–2030; accelerated/AI servers **+30 %/yr** |
| **McKinsey** *AI Power* (2024) | Global DC capacity | **171–219 GW by 2030** (19–22 %/yr CAGR); **156 GW AI**; $6.7 T capex |

These three give a low/base/high envelope: convert TWh↔GW at ~8,760 h × utilization. They agree on **roughly a doubling by ~2030** then continued (decelerating) growth. For 2030→2045 I propose extrapolating with a CAGR that decays from ~15 %/yr (2026–2030) toward ~4–5 %/yr (2040s) as the build-out matures — clearly labeled as an assumption.

### 4.2 Figure design (two panels, matching the notebook's dark-space house style)

**Panel A — Total DC demand forecast with ODC-served share.**
- x: 2026–2045. y: total DC power demand (GW).
- Central line = base envelope (McKinsey/IEA midpoint anchored to ~120–130 GW in 2026 growing to ~200+ GW by 2030); shaded low↔high band from Goldman/McKinsey/IEA spread.
- Overlay ODC-served share scenarios as fractions of **incremental** demand: e.g. **0.5 %, 2 %, 5 %, 10 %** by 2045 (ramping from ~0 in 2026). Each becomes a deployed-ODC-capacity curve `C_ODC(t)`.

**Panel B — ODC LCOC vs market share (the payoff).**
- x: ODC share of DC demand (or deployed ODC GW). y: ODC LCOC ($/EFLOP), with the TDC line for reference.
- Mechanism (Wright's law on the cost parameters that are production-driven): for launch, PV, IT,
  `cost(C_ODC) = cost₀ · (C_ODC / C₀)^b`, `b = log₂(1 − LR)`.
  More ODC share → larger `C_ODC` → faster cost decline → lower ODC LCOC. The curve falls and **crosses the TDC line** at some share/year — that crossing is the headline of the figure.
- Show base/opt/pess learning bands, consistent with Task 3.

This directly delivers your goal — *cost as a function of market-share acquisition* — and ties adoption to learning through cumulative capacity rather than a bare calendar rate.

### 4.3 What I need from you for Panel B
- The **starting cumulative ODC capacity `C₀`** and which year it anchors (e.g. first ~0.1 GW pilot in ~2027). The Wright's-law curve is sensitive to `C₀`.
- Confirm the share scenarios (0.5/2/5/10 %) and whether share is of **total** or **incremental** demand.

---

## Proposed implementation order (on your approval)

1. **Task 2 housekeeping first** — resolve the `T_PV` 80 vs 31.1 °C conflict, rename literature files, repoint cell-4 `Source` columns. (Cleans the foundation.)
2. **Task 1** — set `T_rad = 70 °C`, rewrite §1/§2 scope language, add the `T_rad` sensitivity row. (Verified +0.27 % LCOC.)
3. **Task 3** — update learning-rate table + narrative (γ→Epoch anchor, launch/PV `%/doubling` documentation, `it_cost` double-count caveat, PV-eff cap 35 %).
4. **Task 4** — add the demand cells + two-panel figure; download and file the demand/learning sources.
5. **Re-run end-to-end**, regenerate figures, refresh the summary, and update `references.bib` (`turyshev2026`, Epoch AI, OWID/Swanson, Goldman, IEA, McKinsey).

## Open questions for you
1. `T_PV`: is **31.1 °C** (code) or **80 °C** (markdown) correct?
2. `it_cost_per_MW` learning — keep aggressive −8 %/yr, or split so γ carries the FLOP/W gain and `it_cost` carries only ~−3 to −5 %/yr residual?
3. Raise baseline γ growth to ~+20 %/yr (Epoch-anchored) or keep +15 %?
4. Panel B anchors: starting ODC cumulative capacity `C₀` / year, and share defined vs total or incremental demand?
