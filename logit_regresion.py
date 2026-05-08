"""
PASO 1 — REGRESIÓN LOGÍSTICA
==============================
Pregunta central: ¿tiene mins_to_retest poder predictivo INDEPENDIENTE
después de controlar por breakout_minute, OR_size_pct, y vol_ratio?

Variables predictoras (continuas):
  - mins_to_retest    → tiempo al retest (variable clave)
  - breakout_minute   → hora del breakout (posible confundidor)
  - OR_size_pct       → tamaño del rango (predictor del Nivel 1)
  - vol_ratio         → volumen relativo (predictor del Nivel 2)
  - mfe_pre_retest_pct → excursión pre-retest (predictor del Nivel 3)

Variable dependiente:
  - is_cont = 1 si outcome == 'continuacion', 0 si falla o neutro

Modelos estimados:
  M1: Solo mins_to_retest (baseline)
  M2: mins_to_retest + breakout_minute (el test crítico)
  M3: Modelo completo con todas las variables
  M4: Modelo completo + términos cuadráticos para no linealidades

Output:
  logit_resultados.csv     → coeficientes, odds ratios, p-valores
  logit_comparacion.csv    → comparación de modelos (AIC, pseudo-R²)
  logit_efectos_marginales → efecto marginal de cada variable
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE = os.path.join("orb_results", "nivel4_raw.csv")
OUTPUT_DIR = "orb_results"
OR_MUESTRA = 15

# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y PREPARACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(filepath: str, or_dur: int) -> pd.DataFrame:
    """
    Carga nivel4_raw y prepara variables para la regresión.
    Usa solo OR duration = or_dur para el análisis principal.
    Estandariza todas las variables continuas (media=0, std=1)
    para que los coeficientes sean comparables entre sí.
    """
    df = pd.read_csv(filepath)
    df = df[df['or_duration_min'] == or_dur].copy()

    # Variable dependiente
    df['is_cont'] = (df['outcome'] == 'continuacion').astype(int)

    # Eliminar filas con NaN en variables clave
    vars_necesarias = [
        'mins_to_retest', 'breakout_minute',
        'or_size_pct', 'vol_ratio', 'mfe_pre_retest_pct'
    ]
    df = df.dropna(subset=vars_necesarias)

    print(f"  Registros para OR={or_dur} min: {len(df):,}")
    print(f"  Continuaciones: {df['is_cont'].sum():,} ({df['is_cont'].mean()*100:.1f}%)")
    print(f"  No-continuaciones: {(1-df['is_cont']).sum():,}\n")

    return df


def estandarizar(df: pd.DataFrame,
                 cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Estandariza columnas (z-score: media=0, std=1).
    Retorna el DataFrame con columnas nuevas _std y el diccionario
    de medias/stds para interpretar coeficientes después.
    """
    stats_dict = {}
    for col in cols:
        mean = df[col].mean()
        std  = df[col].std()
        df[f'{col}_std'] = (df[col] - mean) / std
        stats_dict[col] = {'mean': mean, 'std': std}
    return df, stats_dict

# ─────────────────────────────────────────────────────────────────────────────
# AJUSTE DE MODELOS
# ─────────────────────────────────────────────────────────────────────────────

def ajustar_modelo(y: pd.Series, X: pd.DataFrame,
                   nombre: str) -> dict:
    """
    Ajusta un modelo logístico y retorna resultados completos.

    Returns:
        dict con resultados del modelo
    """
    X_const = sm.add_constant(X)
    modelo  = Logit(y, X_const)

    try:
        resultado = modelo.fit(method='newton', maxiter=200, disp=False)
    except Exception:
        resultado = modelo.fit(method='bfgs', maxiter=200, disp=False)

    # Pseudo R-squared de McFadden
    pseudo_r2 = resultado.prsquared

    # AIC y BIC
    aic = resultado.aic
    bic = resultado.bic

    # Log-likelihood
    llf = resultado.llf

    return {
        'nombre':    nombre,
        'resultado': resultado,
        'pseudo_r2': pseudo_r2,
        'aic':       aic,
        'bic':       bic,
        'llf':       llf,
        'n':         int(resultado.nobs),
    }


def extraer_tabla(modelo_dict: dict,
                  stats_dict: dict) -> pd.DataFrame:
    """
    Extrae tabla de coeficientes con odds ratios e intervalos de confianza.
    Convierte coeficientes estandarizados a interpretación práctica.
    """
    res   = modelo_dict['resultado']
    params = res.params
    pvals  = res.pvalues
    conf   = res.conf_int()
    bse    = res.bse

    rows = []
    for var in params.index:
        if var == 'const':
            continue

        coef   = params[var]
        se     = bse[var]
        pval   = pvals[var]
        ci_lo  = conf.loc[var, 0]
        ci_hi  = conf.loc[var, 1]
        or_val = np.exp(coef)
        or_lo  = np.exp(ci_lo)
        or_hi  = np.exp(ci_hi)

        # Signo de significancia
        if pval < 0.001:
            sig = '***'
        elif pval < 0.01:
            sig = '**'
        elif pval < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'

        # Nombre limpio de la variable
        var_clean = var.replace('_std', '').replace('_sq', '²')

        rows.append({
            'modelo':        modelo_dict['nombre'],
            'variable':      var_clean,
            'coef':          round(coef, 4),
            'se':            round(se, 4),
            'z':             round(coef / se, 3),
            'p_valor':       round(pval, 6),
            'significancia': sig,
            'odds_ratio':    round(or_val, 4),
            'OR_IC_95_low':  round(or_lo, 4),
            'OR_IC_95_high': round(or_hi, 4),
        })

    return pd.DataFrame(rows)


def tabla_comparacion(modelos: list[dict]) -> pd.DataFrame:
    """Tabla comparativa de todos los modelos."""
    rows = []
    for m in modelos:
        rows.append({
            'modelo':    m['nombre'],
            'n':         m['n'],
            'pseudo_R2': round(m['pseudo_r2'], 4),
            'AIC':       round(m['aic'], 2),
            'BIC':       round(m['bic'], 2),
            'log_lik':   round(m['llf'], 2),
        })
    return pd.DataFrame(rows)


def efectos_marginales(modelo_dict: dict) -> pd.DataFrame:
    """
    Calcula efectos marginales promedio (Average Marginal Effects).
    Interpreta cuánto cambia la PROBABILIDAD (no el log-odds)
    cuando cada variable cambia en 1 unidad estandarizada.
    """
    res = modelo_dict['resultado']
    try:
        margeff = res.get_margeff()
        me_df   = pd.DataFrame({
            'variable':       [v.replace('_std','').replace('_sq','²')
                               for v in margeff.margeff_names],
            'efecto_marginal': margeff.margeff.round(4),
            'se':              margeff.margeff_se.round(4),
            'p_valor':         margeff.pvalues.round(6),
            'significancia':   ['***' if p < 0.001 else
                                '**'  if p < 0.01  else
                                '*'   if p < 0.05  else 'n.s.'
                                for p in margeff.pvalues],
        })
        me_df['modelo'] = modelo_dict['nombre']
        return me_df
    except Exception as e:
        print(f"  No se pudieron calcular efectos marginales: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# PRINT CONSOLA
# ─────────────────────────────────────────────────────────────────────────────

def print_consola(modelos: list[dict],
                  tablas: list[pd.DataFrame],
                  comp_df: pd.DataFrame,
                  me_df: pd.DataFrame,
                  stats_dict: dict):

    # ── Tabla comparativa de modelos ──────────────────────────────────────────
    print("\n" + "="*70)
    print("COMPARACIÓN DE MODELOS LOGÍSTICOS")
    print("="*70)
    print(f"{'Modelo':<45} | {'N':>5} | {'PseudoR²':>8} | {'AIC':>10} | {'BIC':>10}")
    print("-"*70)
    for _, r in comp_df.iterrows():
        print(f"{r['modelo']:<45} | {int(r['n']):>5} | "
              f"{r['pseudo_R2']:>8.4f} | {r['AIC']:>10.2f} | "
              f"{r['BIC']:>10.2f}")

    # ── Tabla de coeficientes por modelo ──────────────────────────────────────
    for tabla_df in tablas:
        if len(tabla_df) == 0:
            continue
        modelo_nombre = tabla_df['modelo'].iloc[0]
        print(f"\n{'='*80}")
        print(f"COEFICIENTES — {modelo_nombre}")
        print(f"(Variables estandarizadas: coeficiente = cambio en log-odds "
              f"por 1 desviación estándar)")
        print(f"{'='*80}")
        print(f"  {'Variable':<22} | {'Coef':>7} | {'SE':>6} | "
              f"{'Z':>7} | {'p-valor':>9} | {'Sig':>4} | "
              f"{'OR':>6} | {'OR_IC95':>16}")
        print(f"  {'-'*82}")
        for _, r in tabla_df.iterrows():
            ic_str = f"[{r['OR_IC_95_low']:.3f}, {r['OR_IC_95_high']:.3f}]"
            print(f"  {r['variable']:<22} | {r['coef']:>7.4f} | "
                  f"{r['se']:>6.4f} | {r['z']:>7.3f} | "
                  f"{r['p_valor']:>9.6f} | {r['significancia']:>4} | "
                  f"{r['odds_ratio']:>6.3f} | {ic_str:>16}")

    # ── Efectos marginales del modelo completo ────────────────────────────────
    if len(me_df) > 0:
        m3_me = me_df[me_df['modelo'].str.contains('M3')]
        if len(m3_me) > 0:
            print(f"\n{'='*70}")
            print("EFECTOS MARGINALES PROMEDIO — Modelo M3 (completo)")
            print("Interpretación: cambio en PROBABILIDAD de continuación")
            print("por 1 desviación estándar de cada variable")
            print(f"{'='*70}")
            print(f"  {'Variable':<22} | {'Δ Prob':>8} | "
                  f"{'SE':>6} | {'p-valor':>9} | {'Sig':>4}")
            print(f"  {'-'*58}")
            for _, r in m3_me.iterrows():
                print(f"  {r['variable']:<22} | "
                      f"{r['efecto_marginal']:>+8.4f} | "
                      f"{r['se']:>6.4f} | "
                      f"{r['p_valor']:>9.6f} | "
                      f"{r['significancia']:>4}")

    # ── La pregunta crítica ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RESPUESTA A LA PREGUNTA CRÍTICA:")
    print("¿Tiene mins_to_retest poder predictivo INDEPENDIENTE")
    print("después de controlar por breakout_minute?")
    print(f"{'='*70}")

    # Buscar p-valor de mins_to_retest en M2
    for tabla_df in tablas:
        if 'M2' in tabla_df['modelo'].iloc[0]:
            mt_row = tabla_df[tabla_df['variable'] == 'mins_to_retest']
            bm_row = tabla_df[tabla_df['variable'] == 'breakout_minute']
            if len(mt_row) > 0 and len(bm_row) > 0:
                mt_p  = mt_row.iloc[0]['p_valor']
                mt_or = mt_row.iloc[0]['odds_ratio']
                bm_p  = bm_row.iloc[0]['p_valor']
                bm_or = bm_row.iloc[0]['odds_ratio']

                print(f"\n  En M2 (mins_to_retest + breakout_minute):")
                print(f"  mins_to_retest:   OR={mt_or:.3f}  p={mt_p:.6f}  "
                      f"{('*** SIGNIFICATIVO — efecto independiente CONFIRMADO' if mt_p < 0.05 else 'n.s. — POSIBLE CONFUSIÓN CON HORA DEL BREAKOUT')}")
                print(f"  breakout_minute:  OR={bm_or:.3f}  p={bm_p:.6f}  "
                      f"{('*** SIGNIFICATIVO' if bm_p < 0.05 else 'n.s.')}")

                if mt_p < 0.05:
                    print(f"\n  ✓ ESCENARIO BUENO: mins_to_retest tiene efecto")
                    print(f"    independiente de la hora del breakout.")
                    print(f"    El hallazgo #1 del paper SE SOSTIENE.")
                else:
                    print(f"\n  ✗ ESCENARIO PROBLEMÁTICO: mins_to_retest pierde")
                    print(f"    significancia al controlar por breakout_minute.")
                    print(f"    El hallazgo #1 NECESITA REVISIÓN.")

    # ── Interpretación de unidades originales ────────────────────────────────
    print(f"\n{'─'*70}")
    print("REFERENCIA — Desviaciones estándar de las variables originales:")
    for var, s in stats_dict.items():
        print(f"  {var:<25}: media={s['mean']:.2f}, "
              f"std={s['std']:.2f}")
    print(f"{'─'*70}")
    print("Leyenda: *** p<0.001 | ** p<0.01 | * p<0.05 | n.s. no significativo")
    print("OR = Odds Ratio (>1 aumenta probabilidad, <1 la reduce)")
    print(f"{'─'*70}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Cargando datos desde: {INPUT_FILE}")
    df_raw = load_and_prepare(INPUT_FILE, OR_MUESTRA)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Variables a estandarizar
    vars_continuas = [
        'mins_to_retest',
        'breakout_minute',
        'or_size_pct',
        'vol_ratio',
        'mfe_pre_retest_pct',
    ]

    df, stats_dict = estandarizar(df_raw.copy(), vars_continuas)

    y = df['is_cont']

    # ── Definir los 4 modelos ─────────────────────────────────────────────────

    # M1: Solo mins_to_retest (baseline — referencia)
    X1 = df[['mins_to_retest_std']]

    # M2: mins_to_retest + breakout_minute (el test crítico)
    X2 = df[['mins_to_retest_std', 'breakout_minute_std']]

    # M3: Modelo completo con todas las variables
    X3 = df[[
        'mins_to_retest_std',
        'breakout_minute_std',
        'or_size_pct_std',
        'vol_ratio_std',
        'mfe_pre_retest_pct_std',
    ]]

    # M4: Modelo completo + términos cuadráticos
    # (para capturar no linealidades como el pico de vol_ratio en Q3)
    df['vol_ratio_sq']        = df['vol_ratio_std'] ** 2
    df['mfe_pre_sq']          = df['mfe_pre_retest_pct_std'] ** 2
    df['mins_to_retest_sq']   = df['mins_to_retest_std'] ** 2

    X4 = df[[
        'mins_to_retest_std',
        'mins_to_retest_sq',
        'breakout_minute_std',
        'or_size_pct_std',
        'vol_ratio_std',
        'vol_ratio_sq',
        'mfe_pre_retest_pct_std',
        'mfe_pre_sq',
    ]]

    # ── Ajustar modelos ───────────────────────────────────────────────────────
    print("Ajustando modelos logísticos...")

    nombres = [
        "M1: Solo mins_to_retest",
        "M2: mins_to_retest + breakout_minute (test crítico)",
        "M3: Modelo completo (5 variables)",
        "M4: Modelo completo + no linealidades",
    ]

    modelos = []
    for X, nombre in zip([X1, X2, X3, X4], nombres):
        print(f"  Ajustando {nombre}...")
        m = ajustar_modelo(y, X, nombre)
        modelos.append(m)

    # ── Extraer tablas ────────────────────────────────────────────────────────
    tablas = [extraer_tabla(m, stats_dict) for m in modelos]

    # ── Efectos marginales del M3 ─────────────────────────────────────────────
    print("  Calculando efectos marginales (M3)...")
    me_df = efectos_marginales(modelos[2])

    # ── Tabla comparativa ─────────────────────────────────────────────────────
    comp_df = tabla_comparacion(modelos)

    # ── Guardar resultados ────────────────────────────────────────────────────
    todas_tablas = pd.concat(tablas, ignore_index=True)
    todas_tablas.to_csv(
        os.path.join(OUTPUT_DIR, "logit_coeficientes.csv"), index=False)
    comp_df.to_csv(
        os.path.join(OUTPUT_DIR, "logit_comparacion.csv"), index=False)
    if len(me_df) > 0:
        me_df.to_csv(
            os.path.join(OUTPUT_DIR, "logit_efectos_marginales.csv"),
            index=False)

    print(f"\nArchivos guardados en {OUTPUT_DIR}/")

    # ── Print consola ─────────────────────────────────────────────────────────
    print_consola(modelos, tablas, comp_df, me_df, stats_dict)
    print("\nDone ✓")


if __name__ == "__main__":
    main()