"""
ANÁLISIS DE ROBUSTEZ TEMPORAL
==============================
Verifica que los hallazgos de los Niveles 4 y 5 son estables
a través de diferentes regímenes de mercado.

Divide el dataset en 5 subperíodos:
  P1: 2017-2019  → Bull market tranquilo
  P2: 2020       → COVID crash + recuperación
  P3: 2021       → Bull market acelerado
  P4: 2022       → Bear market (Fed tightening)
  P5: 2023-2024  → Recuperación + nuevo bull

Para cada subperíodo calcula:
  - Tasa base de continuación / falla / neutro
  - Tasa de continuación por variable predictora
  - Tasa de continuación con la combinación óptima
  - Score acumulativo (0 a 6 condiciones)

Output:
  robustez_summary.csv     → métricas base por subperíodo
  robustez_predictores.csv → poder predictivo por subperíodo
  robustez_combos.csv      → top combinaciones por subperíodo
  robustez_score.csv       → score acumulativo por subperíodo
"""

import pandas as pd
import numpy as np
import os
import itertools
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE = os.path.join("orb_results", "nivel4_raw.csv")
OUTPUT_DIR = "orb_results"
OR_MUESTRA = 15       # OR duration para el análisis detallado
MIN_CASOS  = 20       # mínimo de casos por combinación

# Umbrales de condiciones (mismos que nivel5.py)
C1_PERCENTIL  = 60
C2_MINS_MAX   = 10
C3_MFE_MIN    = 0.30
C3_MFE_MAX    = 0.50
C4_PERC_LOW   = 40
C4_PERC_HIGH  = 60
C5_BREAK_MAX  = 30

# Definición de subperíodos
SUBPERIODOS = {
    'P1_2017_2019': ('2017-01-01', '2019-12-31'),
    'P2_2020_COVID': ('2020-01-01', '2020-12-31'),
    'P3_2021_bull':  ('2021-01-01', '2021-12-31'),
    'P4_2022_bear':  ('2022-01-01', '2022-12-31'),
    'P5_2023_2024':  ('2023-01-01', '2024-12-31'),
}

CONDITIONS  = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
COND_LABELS = {
    'C1': 'OR_grande(top40%)',
    'C2': f'Retest<={C2_MINS_MAX}min',
    'C3': f'MFE_pre_{C3_MFE_MIN}-{C3_MFE_MAX}%',
    'C4': f'VolRatio_p{C4_PERC_LOW}-p{C4_PERC_HIGH}',
    'C5': f'Break<={C5_BREAK_MAX}min',
    'C6': 'Sesgo_alineado',
}

# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y PREPARACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(filepath: str) -> pd.DataFrame:
    """Carga nivel4_raw y agrega condiciones binarias."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    rows = []
    for or_dur, grp in df.groupby('or_duration_min'):
        grp = grp.copy()

        # Umbrales dinámicos por OR duration
        c1_thresh = grp['or_size_pct'].quantile(C1_PERCENTIL / 100)
        c4_low    = grp['vol_ratio'].quantile(C4_PERC_LOW  / 100)
        c4_high   = grp['vol_ratio'].quantile(C4_PERC_HIGH / 100)

        grp['C1'] = (grp['or_size_pct'] >= c1_thresh).astype(int)
        grp['C2'] = (grp['mins_to_retest'] <= C2_MINS_MAX).astype(int)
        grp['C3'] = ((grp['mfe_pre_retest_pct'] >= C3_MFE_MIN) &
                     (grp['mfe_pre_retest_pct'] <  C3_MFE_MAX)).astype(int)
        grp['C4'] = ((grp['vol_ratio'] >= c4_low) &
                     (grp['vol_ratio'] <= c4_high)).astype(int)
        grp['C5'] = (grp['breakout_minute'] <= C5_BREAK_MAX).astype(int)
        grp['C6'] = (
            ((grp['or_close_position'] > 0.67) & (grp['breakout_dir'] == 'up')) |
            ((grp['or_close_position'] < 0.33) & (grp['breakout_dir'] == 'down'))
        ).astype(int)
        grp['score'] = grp[CONDITIONS].sum(axis=1)

        rows.append(grp)

    return pd.concat(rows, ignore_index=True)


def asignar_subperiodo(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columna con el subperíodo de cada registro."""
    df = df.copy()
    df['subperiodo'] = 'fuera_rango'
    for nombre, (inicio, fin) in SUBPERIODOS.items():
        mask = (df['date'] >= inicio) & (df['date'] <= fin)
        df.loc[mask, 'subperiodo'] = nombre
    return df

# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS BASE POR SUBPERÍODO
# ─────────────────────────────────────────────────────────────────────────────

def metricas_base(df: pd.DataFrame, or_dur: int) -> pd.DataFrame:
    """
    Para cada subperíodo: tasa de continuación, falla, neutro
    y MFE_post promedio. Separado por OR duration.
    """
    sub = df[df['or_duration_min'] == or_dur].copy()
    rows = []

    # Total dataset primero
    n = len(sub)
    if n > 0:
        n_cont  = (sub['outcome'] == 'continuacion').sum()
        n_falla = (sub['outcome'] == 'falla').sum()
        n_neut  = (sub['outcome'] == 'neutro').sum()
        cont_g  = sub[sub['outcome'] == 'continuacion']
        rows.append({
            'subperiodo':       'TOTAL_2017_2024',
            'or_duration_min':  or_dur,
            'n_dias':           sub['date'].nunique(),
            'n_retests':        n,
            'pct_continuacion': round(n_cont  / n * 100, 1),
            'pct_falla':        round(n_falla / n * 100, 1),
            'pct_neutro':       round(n_neut  / n * 100, 1),
            'mfe_post_avg':     round(cont_g['mfe_post_pct'].mean(), 4),
            'mfe_post_med':     round(cont_g['mfe_post_pct'].median(), 4),
            'cont_ratio_avg':   round(cont_g['continuation_ratio'].mean(), 4),
            'mae_falla_avg':    round(sub[sub['outcome']=='falla']['mae_falla_pct'].mean(), 4),
        })

    # Por subperíodo
    for nombre in SUBPERIODOS:
        g = sub[sub['subperiodo'] == nombre]
        n = len(g)
        if n < 10:
            continue
        n_cont  = (g['outcome'] == 'continuacion').sum()
        n_falla = (g['outcome'] == 'falla').sum()
        n_neut  = (g['outcome'] == 'neutro').sum()
        cont_g  = g[g['outcome'] == 'continuacion']
        rows.append({
            'subperiodo':       nombre,
            'or_duration_min':  or_dur,
            'n_dias':           g['date'].nunique(),
            'n_retests':        n,
            'pct_continuacion': round(n_cont  / n * 100, 1),
            'pct_falla':        round(n_falla / n * 100, 1),
            'pct_neutro':       round(n_neut  / n * 100, 1),
            'mfe_post_avg':     round(cont_g['mfe_post_pct'].mean(), 4),
            'mfe_post_med':     round(cont_g['mfe_post_pct'].median(), 4),
            'cont_ratio_avg':   round(cont_g['continuation_ratio'].mean(), 4),
            'mae_falla_avg':    round(g[g['outcome']=='falla']['mae_falla_pct'].mean(), 4),
        })

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# PODER PREDICTIVO POR SUBPERÍODO
# ─────────────────────────────────────────────────────────────────────────────

def predictores_por_subperiodo(df: pd.DataFrame, or_dur: int) -> pd.DataFrame:
    """
    Para las 3 variables más importantes (mins_to_retest, or_size, mfe_pre):
    tasa de continuación por bin, separada por subperíodo.
    Verifica que el ranking de variables se mantiene estable.
    """
    sub = df[df['or_duration_min'] == or_dur].copy()
    rows = []

    periodos = ['TOTAL'] + list(SUBPERIODOS.keys())

    for periodo in periodos:
        if periodo == 'TOTAL':
            g = sub
        else:
            g = sub[sub['subperiodo'] == periodo]

        if len(g) < 20:
            continue

        # Variable 1 — mins_to_retest
        g['_mt'] = pd.cut(g['mins_to_retest'],
                          bins=[0, 10, 20, 30, 60, 999],
                          labels=['0-10min', '10-20min', '20-30min',
                                  '30-60min', '>60min'])
        for bin_label, gg in g.groupby('_mt', observed=True):
            if len(gg) < 10:
                continue
            n_cont = (gg['outcome'] == 'continuacion').sum()
            rows.append({
                'subperiodo': periodo,
                'variable':   'mins_to_retest',
                'segmento':   str(bin_label),
                'n':          len(gg),
                'pct_cont':   round(n_cont / len(gg) * 100, 1),
            })

        # Variable 2 — or_size_pct (quintiles)
        try:
            g['_qs'] = pd.qcut(g['or_size_pct'], q=5,
                               labels=['Q1','Q2','Q3','Q4','Q5'],
                               duplicates='drop')
            for q, gg in g.groupby('_qs', observed=True):
                if len(gg) < 10:
                    continue
                n_cont = (gg['outcome'] == 'continuacion').sum()
                rows.append({
                    'subperiodo': periodo,
                    'variable':   'or_size_pct',
                    'segmento':   str(q),
                    'n':          len(gg),
                    'pct_cont':   round(n_cont / len(gg) * 100, 1),
                })
        except Exception:
            pass

        # Variable 3 — mfe_pre_retest_pct
        g['_mp'] = pd.cut(g['mfe_pre_retest_pct'],
                          bins=[0, 0.30, 0.40, 0.50, 0.75, 999],
                          labels=['0.20-0.30%', '0.30-0.40%',
                                  '0.40-0.50%', '0.50-0.75%', '>0.75%'])
        for bin_label, gg in g.groupby('_mp', observed=True):
            if len(gg) < 10:
                continue
            n_cont = (gg['outcome'] == 'continuacion').sum()
            rows.append({
                'subperiodo': periodo,
                'variable':   'mfe_pre_retest_pct',
                'segmento':   str(bin_label),
                'n':          len(gg),
                'pct_cont':   round(n_cont / len(gg) * 100, 1),
            })

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# COMBINACIONES ÓPTIMAS POR SUBPERÍODO
# ─────────────────────────────────────────────────────────────────────────────

def combos_por_subperiodo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para las 5 combinaciones más importantes del Nivel 5,
    calcula la tasa de continuación por subperíodo (todos los OR juntos).
    """
    # Las 5 combinaciones clave identificadas en el Nivel 5
    COMBOS_CLAVE = [
        ('C1', 'C2', 'C4'),          # #4 — la más robusta (N=589)
        ('C2', 'C4'),                 # #7 — la más simple (N=810)
        ('C1', 'C2'),                 # #13 — mayor muestra (N=2256)
        ('C1', 'C2', 'C4', 'C6'),    # #5 — con sesgo alineado
        ('C2', 'C4', 'C6'),           # #6 — sin OR grande
    ]

    rows = []
    periodos = ['TOTAL'] + list(SUBPERIODOS.keys())

    for periodo in periodos:
        if periodo == 'TOTAL':
            g = df
        else:
            g = df[df['subperiodo'] == periodo]

        for combo in COMBOS_CLAVE:
            mask = pd.Series([True] * len(g), index=g.index)
            for c in combo:
                mask = mask & (g[c] == 1)
            sub = g[mask]
            n = len(sub)

            if n < MIN_CASOS:
                continue

            n_cont  = (sub['outcome'] == 'continuacion').sum()
            n_falla = (sub['outcome'] == 'falla').sum()
            cont_g  = sub[sub['outcome'] == 'continuacion']
            label   = ' + '.join([COND_LABELS[c] for c in combo])

            rows.append({
                'subperiodo':       periodo,
                'combinacion':      label,
                'n':                n,
                'pct_continuacion': round(n_cont  / n * 100, 1),
                'pct_falla':        round(n_falla / n * 100, 1),
                'mfe_post_avg':     round(cont_g['mfe_post_pct'].mean(), 4)
                                    if len(cont_g) > 0 else np.nan,
            })

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# SCORE ACUMULATIVO POR SUBPERÍODO
# ─────────────────────────────────────────────────────────────────────────────

def score_por_subperiodo(df: pd.DataFrame, or_dur: int) -> pd.DataFrame:
    """
    Para cada subperíodo: tasa de continuación por score (0-6).
    Verifica que la linealidad del score se mantiene en todos los regímenes.
    """
    sub = df[df['or_duration_min'] == or_dur].copy()
    rows = []

    periodos = ['TOTAL'] + list(SUBPERIODOS.keys())

    for periodo in periodos:
        if periodo == 'TOTAL':
            g = sub
        else:
            g = sub[sub['subperiodo'] == periodo]

        for score_val, gg in g.groupby('score'):
            n = len(gg)
            if n < 5:
                continue
            n_cont  = (gg['outcome'] == 'continuacion').sum()
            n_falla = (gg['outcome'] == 'falla').sum()
            rows.append({
                'subperiodo':       periodo,
                'or_duration_min':  or_dur,
                'score':            int(score_val),
                'n':                n,
                'pct_continuacion': round(n_cont  / n * 100, 1),
                'pct_falla':        round(n_falla / n * 100, 1),
                'pct_neutro':       round((n - n_cont - n_falla) / n * 100, 1),
            })

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# PRINT CONSOLA
# ─────────────────────────────────────────────────────────────────────────────

def print_consola(base_df, pred_df, combos_df, score_df):

    # ── Tabla 1: Métricas base por subperíodo ─────────────────────────────────
    print("\n" + "="*90)
    print(f"ROBUSTEZ — MÉTRICAS BASE POR SUBPERÍODO (OR {OR_MUESTRA} min)")
    print("="*90)
    print(f"{'Subperíodo':<22} | {'N_días':>6} | {'N_ret':>5} | "
          f"{'%Cont':>6} | {'%Falla':>7} | {'%Neut':>6} | "
          f"{'MFE_post':>8} | {'ContRatio':>9}")
    print("-"*90)

    sub = base_df[base_df['or_duration_min'] == OR_MUESTRA]
    for _, r in sub.iterrows():
        mfe = f"{r['mfe_post_avg']:.4f}" if not pd.isna(r['mfe_post_avg']) else "  N/A "
        cr  = f"{r['cont_ratio_avg']:.4f}" if not pd.isna(r['cont_ratio_avg']) else "  N/A "
        print(f"{r['subperiodo']:<22} | {int(r['n_dias']):>6} | "
              f"{int(r['n_retests']):>5} | "
              f"{r['pct_continuacion']:>6.1f} | "
              f"{r['pct_falla']:>7.1f} | "
              f"{r['pct_neutro']:>6.1f} | "
              f"{mfe:>8} | {cr:>9}")

    # ── Tabla 2: mins_to_retest por subperíodo ────────────────────────────────
    print("\n" + "="*75)
    print(f"ROBUSTEZ — TASA DE CONTINUACIÓN POR mins_to_retest Y SUBPERÍODO")
    print(f"(OR {OR_MUESTRA} min — ¿el retest rápido sigue prediciendo en todos los regímenes?)")
    print("="*75)

    mt_df = pred_df[pred_df['variable'] == 'mins_to_retest'].copy()
    periodos = ['TOTAL'] + list(SUBPERIODOS.keys())
    segmentos = ['0-10min', '10-20min', '20-30min', '30-60min', '>60min']

    header = f"{'Segmento':<12}"
    for p in periodos:
        p_short = p.replace('TOTAL_2017_2024', 'TOTAL').replace(
            'P1_2017_2019','P1').replace('P2_2020_COVID','P2').replace(
            'P3_2021_bull','P3').replace('P4_2022_bear','P4').replace(
            'P5_2023_2024','P5')
        header += f" | {p_short:>8}"
    print(header)
    print("-"*75)

    for seg in segmentos:
        row_str = f"{seg:<12}"
        for periodo in periodos:
            val = mt_df[(mt_df['subperiodo'] == periodo) &
                        (mt_df['segmento'] == seg)]
            if len(val) > 0:
                row_str += f" | {val.iloc[0]['pct_cont']:>7.1f}%"
            else:
                row_str += f" | {'  N/A':>8}"
        print(row_str)

    # ── Tabla 3: mfe_pre por subperíodo ──────────────────────────────────────
    print("\n" + "="*75)
    print(f"ROBUSTEZ — TASA DE CONTINUACIÓN POR mfe_pre Y SUBPERÍODO")
    print(f"(¿el hallazgo contraintuitivo se mantiene en todos los regímenes?)")
    print("="*75)

    mp_df = pred_df[pred_df['variable'] == 'mfe_pre_retest_pct'].copy()
    segmentos_mp = ['0.20-0.30%', '0.30-0.40%', '0.40-0.50%',
                    '0.50-0.75%', '>0.75%']

    header = f"{'Segmento':<13}"
    for p in periodos:
        p_short = p.replace('TOTAL_2017_2024','TOTAL').replace(
            'P1_2017_2019','P1').replace('P2_2020_COVID','P2').replace(
            'P3_2021_bull','P3').replace('P4_2022_bear','P4').replace(
            'P5_2023_2024','P5')
        header += f" | {p_short:>8}"
    print(header)
    print("-"*75)

    for seg in segmentos_mp:
        row_str = f"{seg:<13}"
        for periodo in periodos:
            val = mp_df[(mp_df['subperiodo'] == periodo) &
                        (mp_df['segmento'] == seg)]
            if len(val) > 0:
                row_str += f" | {val.iloc[0]['pct_cont']:>7.1f}%"
            else:
                row_str += f" | {'  N/A':>8}"
        print(row_str)

    # ── Tabla 4: Combinaciones óptimas por subperíodo ────────────────────────
    print("\n" + "="*90)
    print("ROBUSTEZ — COMBINACIONES ÓPTIMAS POR SUBPERÍODO (todos los OR)")
    print("="*90)
    print(f"{'Subperíodo':<22} | {'N':>5} | {'%Cont':>6} | "
          f"{'%Falla':>7} | {'MFE_post':>8} | Combinación")
    print("-"*90)

    combos_clave_labels = [
        'OR_grande(top40%) + Retest<=10min + VolRatio_p40-p60',
        'Retest<=10min + VolRatio_p40-p60',
        'OR_grande(top40%) + Retest<=10min',
    ]

    for combo_label in combos_clave_labels:
        print(f"\n  [{combo_label}]")
        sub_c = combos_df[combos_df['combinacion'] == combo_label]
        for _, r in sub_c.iterrows():
            mfe = f"{r['mfe_post_avg']:.4f}" if not pd.isna(r['mfe_post_avg']) else "  N/A "
            print(f"  {r['subperiodo']:<20} | {int(r['n']):>5} | "
                  f"{r['pct_continuacion']:>6.1f} | "
                  f"{r['pct_falla']:>7.1f} | "
                  f"{mfe:>8}")

    # ── Tabla 5: Score acumulativo por subperíodo ─────────────────────────────
    print("\n" + "="*80)
    print(f"ROBUSTEZ — SCORE ACUMULATIVO POR SUBPERÍODO (OR {OR_MUESTRA} min)")
    print(f"(¿la linealidad score→continuación se mantiene en todos los regímenes?)")
    print("="*80)

    header = f"{'Score':<7}"
    for p in periodos:
        p_short = p.replace('TOTAL_2017_2024','TOTAL').replace(
            'P1_2017_2019','P1').replace('P2_2020_COVID','P2').replace(
            'P3_2021_bull','P3').replace('P4_2022_bear','P4').replace(
            'P5_2023_2024','P5')
        header += f" | {p_short:>8}"
    print(header)
    print("-"*80)

    for score_val in range(0, 7):
        row_str = f"{score_val:<7}"
        for periodo in periodos:
            val = score_df[(score_df['subperiodo'] == periodo) &
                           (score_df['score'] == score_val) &
                           (score_df['or_duration_min'] == OR_MUESTRA)]
            if len(val) > 0:
                row_str += f" | {val.iloc[0]['pct_continuacion']:>7.1f}%"
            else:
                row_str += f" | {'  N/A':>8}"
        print(row_str)

    print(f"\n{'─'*70}")
    print("ARCHIVOS GENERADOS en orb_results/:")
    print("  robustez_summary.csv     → métricas base por subperíodo")
    print("  robustez_predictores.csv → poder predictivo por subperíodo")
    print("  robustez_combos.csv      → combinaciones óptimas por subperíodo")
    print("  robustez_score.csv       → score acumulativo por subperíodo")
    print(f"{'─'*70}")
    print("\nINTERPRETACIÓN:")
    print("  ROBUSTO     → variación <10pp entre subperíodos")
    print("  PARCIALMENTE → variación 10-20pp, ranking se mantiene")
    print("  NO ROBUSTO  → variación >20pp o ranking cambia")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Cargando nivel4_raw desde: {INPUT_FILE}")
    df = load_and_prepare(INPUT_FILE)
    df = asignar_subperiodo(df)

    print(f"Registros totales: {len(df):,}")
    print(f"\nDistribución por subperíodo:")
    for nombre in SUBPERIODOS:
        n = len(df[(df['subperiodo'] == nombre) &
                   (df['or_duration_min'] == OR_MUESTRA)])
        dias = df[(df['subperiodo'] == nombre) &
                  (df['or_duration_min'] == OR_MUESTRA)]['date'].nunique()
        print(f"  {nombre:<22}: {n:>4} retests  ({dias} días)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Calcular todo
    print(f"\nCalculando métricas base (OR {OR_MUESTRA} min)...")
    base_df = metricas_base(df, OR_MUESTRA)

    print("Calculando poder predictivo por subperíodo...")
    pred_df = predictores_por_subperiodo(df, OR_MUESTRA)

    print("Calculando combinaciones óptimas por subperíodo...")
    combos_df = combos_por_subperiodo(df)

    print(f"Calculando score acumulativo (OR {OR_MUESTRA} min)...")
    score_df = score_por_subperiodo(df, OR_MUESTRA)

    # Guardar CSVs
    base_df.to_csv(  os.path.join(OUTPUT_DIR, "robustez_summary.csv"),     index=False)
    pred_df.to_csv(  os.path.join(OUTPUT_DIR, "robustez_predictores.csv"), index=False)
    combos_df.to_csv(os.path.join(OUTPUT_DIR, "robustez_combos.csv"),      index=False)
    score_df.to_csv( os.path.join(OUTPUT_DIR, "robustez_score.csv"),       index=False)

    print("\nCSVs guardados.")

    # Imprimir consola
    print_consola(base_df, pred_df, combos_df, score_df)
    print("\nDone ✓")


if __name__ == "__main__":
    main()