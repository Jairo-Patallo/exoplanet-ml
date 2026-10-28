"""Variables físicas derivadas (Fase 2 de la hoja de ruta). Todas se calculan fila a fila: no hay fuga.

Convenciones de las vistas (comprobadas empíricamente sobre train):
  flux_local (81): tránsito centrado en el bin 40, ventana = ±4T  ->  T ≈ 81/8 ≈ 10 bins.
  flux_global / flux_odd / flux_even (201): tránsito centrado en el bin 100, fase ±0.5 en los extremos.
"""
import numpy as np
import pandas as pd

R_SUN_IN_R_EARTH = 109.1
G = 6.674e-8                       # cgs
RHO_SUN = 1.408                    # g/cm^3
KEPLER_BASELINE_D = 1460.0         # ~4 años de observación de Kepler (Q1-Q17)

LOC_IN = slice(35, 46)             # |i-40| <= 5  -> dentro del tránsito (T completo)
LOC_CORE = slice(39, 42)           # 3 bins centrales (fondo del tránsito)
LOC_VALID = slice(5, 76)           # bins 0-4 y 76-80 tienen 2-3x más dispersión (artefacto de borde): se excluyen
LOC_OUT = np.r_[5:25, 56:76]       # > 1.5T del centro: fuera del tránsito
GLB_IN = slice(95, 106)            # centro en las vistas de 201 bins
GLB_OUT = np.r_[20:70, 131:181]    # lejos del tránsito y de la fase 0.5
GLB_SEC = np.r_[0:10, 191:201]     # fase |φ| > 0.45 -> donde caería un eclipse secundario


def robust_sigma(x: np.ndarray) -> np.ndarray:
    """σ ≈ 1.4826·MAD por fila: estimador del ruido gaussiano robusto a outliers y al propio tránsito."""
    med = np.median(x, axis=1, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=1)


def catalog_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables a partir de columnas del catálogo (ecuaciones 2, 3, 5, 6 y 9 del informe)."""
    out = pd.DataFrame(index=df.index)
    delta = df["depth_ppm"] * 1e-6
    k = np.sqrt(delta.where(delta > 0))
    P_s, T_s = df["period_days"] * 86400, df["duration_hrs"] * 3600

    # Ec. 2: Rp = R*·√δ  (en radios terrestres)
    out["rp_implied"] = df["radius"] * k * R_SUN_IN_R_EARTH
    out["log_rp_ratio"] = np.log10(out["rp_implied"] / df["planet_radius_earth"])

    # Ec. 5 con b=0 -> a/R* = P(1+k)/(πT). Con b>0 la a/R* real es MENOR, así que ρ(b=0) es una
    # cota SUPERIOR de la densidad: log ρ_tr/ρ_cat << 0 no puede explicarse con b y delata un problema.
    a_over_r = P_s * (1 + k) / (np.pi * T_s)
    rho_transit = 3 * np.pi / (G * P_s ** 2) * a_over_r ** 3
    rho_catalog = RHO_SUN * df["mass"] / df["radius"] ** 3
    out["log_rho_ratio"] = np.log10(rho_transit / rho_catalog)
    out["duty_cycle"] = T_s / P_s

    # Ec. 9 (solo Kepler: cdpp_3hr es NaN en K2/TESS): SNR ≈ δ/CDPP · √(T/3h · N_tr)
    n_tr = KEPLER_BASELINE_D / df["period_days"]
    out["snr_expected"] = (df["depth_ppm"] / df["cdpp_3hr"]) * np.sqrt(df["duration_hrs"] / 3 * n_tr)
    return out


def curve_features(local: np.ndarray, odd: np.ndarray, even: np.ndarray, glob: np.ndarray) -> pd.DataFrame:
    """Variables medidas directamente sobre las vistas plegadas (sin usar el catálogo)."""
    np.seterr(divide="ignore", invalid="ignore")   # σ=0 o δ=0 -> inf/NaN, se limpian al final
    sig_loc = robust_sigma(local[:, LOC_OUT])
    depth = 1 - np.median(local[:, LOC_CORE], axis=1)
    n_in = LOC_IN.stop - LOC_IN.start
    feats = {
        "depth_local_ppm": depth * 1e6,
        # SNR del tránsito plegado: δ / (σ/√n)
        "snr_local": depth / (sig_loc / np.sqrt(n_in)),
        # Trapecio simétrico de profundidad δ e ingreso τ: déficit medio en T = δ(1-τ/T)
        # => τ/T = 1 - déficit_medio/δ.  Caja (U) -> 0 ; triángulo (V) -> 0.5 ; planeta k=0.1 -> ≈0.09
        "ingress_frac": 1 - (1 - local[:, LOC_IN]).mean(axis=1) / depth,
    }
    # Test par/impar: z = |δ_i - δ_p| / √(SE_i² + SE_p²), con SE de la mediana ≈ 1.2533·σ/√n
    n = GLB_IN.stop - GLB_IN.start
    d_odd, d_even = (1 - np.median(v[:, GLB_IN], axis=1) for v in (odd, even))
    se_odd, se_even = (1.2533 * robust_sigma(v[:, GLB_OUT]) / np.sqrt(n) for v in (odd, even))
    feats["odd_even_z"] = np.abs(d_odd - d_even) / np.sqrt(se_odd ** 2 + se_even ** 2)
    # Eclipse secundario en fase ±0.5: profundidad respecto al nivel fuera de tránsito, en unidades de SE
    base = np.median(glob[:, GLB_OUT], axis=1)
    d_sec = base - np.median(glob[:, GLB_SEC], axis=1)
    feats["secondary_z"] = d_sec / (1.2533 * robust_sigma(glob[:, GLB_OUT]) / np.sqrt(len(GLB_SEC)))
    out = pd.DataFrame(feats).replace([np.inf, -np.inf], np.nan)
    out.loc[~(depth > 0)] = np.nan     # sin caída medible las variables no tienen sentido físico
    return out
