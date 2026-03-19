# Glacial Discharge Downscaling

**Repository for the code used in:**  
Downscaling daily discharge to sub-daily scales for alpine glacierized catchments  
Anne-Laure Argentin, Mattia Gianini, Bettina Schaefli, Pascal Horton, Valérie Chavez-Demoulin, Felix Pitscheider, Leona Repnik, Simone Bizzi, Stuart N. Lane, Francesco Comiti (manuscript submitted to Water Resources Research)  

Corresponding author: Anne-Laure Argentin – aargentin@bordeaux-inp.fr

---

## Overview

This repository contains the code used to downscale daily mean river discharge to sub-daily (15-minute) timescales in glacier-fed alpine catchments. The method is designed for catchments influenced by snow and ice melt, capturing seasonally-varying diel flow patterns without requiring high-resolution meteorological inputs.

The approach adapts a maximum entropy framework (POME) to the hydrological dynamics of glacierized systems. It has been calibrated on a 45-year dataset of 15-minute discharge records from seven glacier-fed catchments in the southwestern Swiss Alps.

---

## Key Features

- Converts daily mean discharge into sub-daily flow duration curves.
- Captures seasonal variations in diel discharge cycles.
- Accounts for snow and ice melt contributions using hydrological model outputs.
- Evaluates the influence of climate warming on sub-daily flow dynamics.
- Applicable to catchments with limited high-resolution discharge data.

---

## Method Summary

1. **Input:** Daily discharge time series from hydrological models or observations.
2. **Process:** Apply a maximum entropy approach calibrated for glacier-fed systems.
3. **Output:** Sub-daily discharge estimates (daily flow duration curves) at 15-minute resolution.
4. **Validation:** Comparison against observed high-resolution discharge measurements.

The method uses a sigmoid function to represent seasonally-varying flow patterns, providing robust reconstruction of sub-daily river flows.

---

## Installation & Requirements

- Python ≥ 3.8  
- Required packages (install via pip):
pip install -r requirements.txt
- Optional: Hydrological model outputs for snow depth, ice melt, or other variables to improve downscaling accuracy.

---

## Citation

If you use this code, please cite the corresponding manuscript.


## Code structure

### Core module
- \ref namespace_downscaling

### Reproducibility scripts
- \ref run_downscaling


