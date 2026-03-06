# 🌍 Semi-Synthetic Data Bases

---

## 📦 What's Inside

This directory contains **real-world time series data** that can be used as bases for semi-synthetic dataset generation.

### 📊 Available Datasets:

| Dataset | Description |
|---------|-------------|
| `gold.npy` | Gold price time series |
| `nvidia.npy` | NVIDIA stock data |
| `s_p.npy` | S&P index data |
| `rivers_ts_flood.csv` | River flood measurements (from CausalRivers) |

---

## 🎯 Purpose

---

> **💡 Tip:** Combine real-world bases with synthetic causal structures for more realistic benchmark scenarios! If you want to include other time-series for semi synthetic data generation you can do this by changing the pathing in the corresponding component. E.g. semi_synthetic_noise_path for the noise generator.
Note, we are working on making this more expressive.