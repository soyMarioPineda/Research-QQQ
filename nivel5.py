"""
MODELO COMBINADO — Cruce de condiciones predictoras
=====================================================
Evalúa todas las combinaciones posibles de 6 condiciones binarias
para todos los OR durations (5 a 120 min).

Las 6 condiciones:
  C1 — OR grande:        or_size_pct en el 40% superior del día
  C2 — Retest rápido:    mins_to_retest <= 10 minutos
  C3 — MFE_pre moderado: mfe_pre_retest_pct entre 0.30% y 0.50%
  C4 — Volumen óptimo:   vol_ratio entre percentil 40 y 60
  C5 — Breakout temprano: breakout_minute <= 30 minutos
  C6 — Sesgo alineado:   or_close_position alineado con breakout_dir

Para cada combinación de condiciones calcula:
  - N (casos que cumplen)
  - %Continuación
  - %Falla
  - %Neutro
  - MFE_post promedio
  - continuation_ratio promedio

Outputs:
  modelo_combinado_raw.csv      → todas las combinaciones × OR duration
  modelo_combinado_top.csv      → top 20 combinaciones por %continuación
                                   (mínimo 30 casos, ordenado por %cont)
  modelo_combinado_resumen.csv  → resumen por número de condiciones cumplidas
  modelo_combinado_por_or.csv   → mejor combinación por cada OR duration
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

INPUT_FILE  = os.path.join("orb_results", "nivel4_raw.csv")
OUTPUT_DIR  = "orb_results"
MIN_CASOS   = 30    # mínimo de casos para considerar una combinación válida

# Umbrales de las condiciones
C2_MINS_MAX    = 10     # retest en menos de X minutos
C3_MFE_MIN     = 0.30   # MFE_pre mínimo %
C3_MFE_MAX     = 0.50   # MFE_pre máximo %
C5_BREAK_MAX   = 30     # breakout antes de X minutos desde apertura
C1_PERCENTIL   = 60     # or_size_pct en el percentil X o superior (top 40%)
C4_PERC_LOW    = 40     # vol_ratio entre percentil X
C4_PERC_HIGH   = 60     # y percentil Y

# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y PREPARACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Calcular umbrales dinámicos por OR duration para C1 y C4
    # (percentiles calculados dentro de cada OR duration para ser justos)
    rows = []
    for or_dur, grp in df.groupby('or_duration_min'):
        grp = grp.copy()

        # C1 — OR grande: top 40% del or_size_pct para este OR duration
        c1_thresh = grp['or_size_pct'].quantile(C1_PERCENTIL / 100)

        # C4 — Volumen óptimo: entre percentil 40 y 60
        c4_low  = grp['vol_ratio'].quantile(C4_PERC_LOW  / 100)
        c4_high = grp['vol_ratio'].quantile(C4_PERC_HIGH / 100)

        # Asignar condiciones binarias
        grp['C1'] = (grp['or_size_pct'] >= c1_thresh).astype(int)
        grp['C2'] = (grp['mins_to_retest'] <= C2_MINS_MAX).astype(int)
        grp['C3'] = ((grp['mfe_pre_retest_pct'] >= C3_MFE_MIN) &
                     (grp['mfe_pre_retest_pct'] <  C3_MFE_MAX)).astype(int)
        grp['C4'] = ((grp['vol_ratio'] >= c4_low) &
                     (grp['vol_ratio'] <= c4_high)).astype(int)
        grp['C5'] = (grp['breakout_minute'] <= C5_BREAK_MAX).astype(int)

        # C6 — Sesgo alineado: OR cerró en dirección del breakout
        grp['C6'] = (
            ((grp['or_close_position'] > 0.67) & (grp['breakout_dir'] == 'up')) |
            ((grp['or_close_position'] < 0.33) & (grp['breakout_dir'] == 'down'))
        ).astype(int)

        # Score total (número de condiciones cumplidas)
        grp['score'] = grp[['C1','C2','C3','C4','C5','C6']].sum(axis=1)

        rows.append(grp)

    return pd.concat(rows, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUACIÓN DE COMBINACIONES
# ─────────────────────────────────────────────────────────────────────────────

CONDITIONS = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']

COND_LABELS = {
    'C1': f'OR_grande(top40%)',
    'C2': f'Retest<={C2_MINS_MAX}min',
    'C3': f'MFE_pre_{C3_MFE_MIN}-{C3_MFE_MAX}%',
    'C4': f'VolRatio_p{C4_PERC_LOW}-p{C4_PERC_HIGH}',
    'C5': f'Break<={C5_BREAK_MAX}min',
    'C6': 'Sesgo_alineado',
}


def evaluar_combinacion(df: pd.DataFrame, condiciones: tuple) -> dict | None:
    """
    Filtra el dataframe por las condiciones dadas y calcula métricas.
    condiciones: tupla de nombres de columnas, ej: ('C1', 'C2', 'C3')
    """
    mask = pd.Series([True] * len(df), index=df.index)
    for c in condiciones:
        mask = mask & (df[c] == 1)

    sub = df[mask]
    n = len(sub)

    if n < MIN_CASOS:
        return None

    n_cont   = (sub['outcome'] == 'continuacion').sum()
    n_falla  = (sub['outcome'] == 'falla').sum()
    n_neutro = (sub['outcome'] == 'neutro').sum()

    cont_grp  = sub[sub['outcome'] == 'continuacion']
    falla_grp = sub[sub['outcome'] == 'falla']

    label = ' + '.join([COND_LABELS[c] for c in condiciones])
    n_cond = len(condiciones)

    return {
        'condiciones':        label,
        'n_condiciones':      n_cond,
        'n':                  n,
        'pct_continuacion':   round(n_cont  / n * 100, 1),
        'pct_falla':          round(n_falla / n * 100, 1),
        'pct_neutro':         round(n_neutro / n * 100, 1),
        'mfe_post_avg':       round(cont_grp['mfe_post_pct'].mean(), 4)       if len(cont_grp) > 0 else np.nan,
        'mfe_post_med':       round(cont_grp['mfe_post_pct'].median(), 4)     if len(cont_grp) > 0 else np.nan,
        'cont_ratio_avg':     round(cont_grp['continuation_ratio'].mean(), 4) if len(cont_grp) > 0 else np.nan,
        'mae_falla_avg':      round(falla_grp['mae_falla_pct'].mean(), 4)     if len(falla_grp) > 0 else np.nan,
        'mins_cont_avg':      round(cont_grp['mins_to_continuation'].mean(), 1) if len(cont_grp) > 0 else np.nan,
    }


def evaluar_todas_combinaciones(df: pd.DataFrame,
                                 or_dur: int = None) -> pd.DataFrame:
    """
    Evalúa las 63 combinaciones posibles de las 6 condiciones.
    Si or_dur es None, usa todos los OR durations.
    """
    if or_dur is not None:
        sub = df[df['or_duration_min'] == or_dur]
    else:
        sub = df

    results = []
    for r in range(1, len(CONDITIONS) + 1):
        for combo in itertools.combinations(CONDITIONS, r):
            result = evaluar_combinacion(sub, combo)
            if result:
                result['or_duration_min'] = or_dur if or_dur else 'todos'
                results.append(result)

    return pd.DataFrame(results).sort_values('pct_continuacion', ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# RESÚMENES
# ─────────────────────────────────────────────────────────────────────────────

def resumen_por_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada OR duration: tasa de continuación según cuántas
    condiciones se cumplen simultáneamente (score 0-6).
    """
    rows = []
    for or_dur, grp in df.groupby('or_duration_min'):
        for score, g in grp.groupby('score'):
            n = len(g)
            if n < 5:
                continue
            n_cont  = (g['outcome'] == 'continuacion').sum()
            n_falla = (g['outcome'] == 'falla').sum()
            rows.append({
                'or_duration_min':  or_dur,
                'score':            int(score),
                'n':                n,
                'pct_continuacion': round(n_cont  / n * 100, 1),
                'pct_falla':        round(n_falla / n * 100, 1),
                'pct_neutro':       round((n - n_cont - n_falla) / n * 100, 1),
                'mfe_post_avg':     round(g.loc[g['outcome']=='continuacion',
                                               'mfe_post_pct'].mean(), 4),
            })
    return pd.DataFrame(rows)


def mejor_combo_por_or(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada OR duration, encuentra la mejor combinación de condiciones
    (mínimo MIN_CASOS casos, máximo %continuación).
    """
    rows = []
    or_durations = sorted(df['or_duration_min'].unique())

    for or_dur in or_durations:
        combos = evaluar_todas_combinaciones(df, or_dur=or_dur)
        if len(combos) == 0:
            continue
        # Mejor por continuación con al menos MIN_CASOS casos
        mejor = combos[combos['n'] >= MIN_CASOS].iloc[0] if len(combos) > 0 else None
        if mejor is not None:
            rows.append(mejor)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PRINT CONSOLA
# ─────────────────────────────────────────────────────────────────────────────

def print_consola(score_df, top_df, por_or_df, df):

    # ── Tabla 1: Score global (todos los OR) ─────────────────────────────────
    print("\n" + "="*80)
    print("MODELO COMBINADO — TASA DE CONTINUACIÓN POR SCORE (TODOS LOS OR)")
    print("(score = número de condiciones cumplidas simultáneamente)")
    print("="*80)

    # Agrupar por score, todos los OR juntos
    for score, grp in df.groupby('score'):
        n = len(grp)
        n_cont  = (grp['outcome'] == 'continuacion').sum()
        n_falla = (grp['outcome'] == 'falla').sum()
        n_neut  = n - n_cont - n_falla
        mfe     = grp.loc[grp['outcome']=='continuacion', 'mfe_post_pct'].mean()
        print(f"  Score {int(score)}: N={n:>6,} | "
              f"%Cont={n_cont/n*100:>5.1f}% | "
              f"%Falla={n_falla/n*100:>5.1f}% | "
              f"%Neutro={n_neut/n*100:>5.1f}% | "
              f"MFE_post={mfe:.4f}%")

    # ── Tabla 2: Top 20 combinaciones (todos los OR juntos) ──────────────────
    print("\n" + "="*100)
    print("MODELO COMBINADO — TOP 20 COMBINACIONES (TODOS LOS OR, mín 30 casos)")
    print("="*100)
    print(f"{'#':>3} | {'N':>6} | {'%Cont':>6} | {'%Falla':>7} | "
          f"{'%Neut':>6} | {'MFE_post':>8} | {'ContRatio':>9} | Condiciones")
    print("-"*100)
    for rank, (_, r) in enumerate(top_df.head(20).iterrows(), 1):
        mfe = f"{r['mfe_post_avg']:.4f}" if not pd.isna(r['mfe_post_avg']) else "  N/A "
        cr  = f"{r['cont_ratio_avg']:.4f}" if not pd.isna(r['cont_ratio_avg']) else "  N/A "
        print(f"{rank:>3} | {int(r['n']):>6,} | {r['pct_continuacion']:>6.1f} | "
              f"{r['pct_falla']:>7.1f} | {r['pct_neutro']:>6.1f} | "
              f"{mfe:>8} | {cr:>9} | {r['condiciones']}")

    # ── Tabla 3: Por número de condiciones ───────────────────────────────────
    print("\n" + "="*75)
    print("MODELO COMBINADO — EFECTO ACUMULATIVO DE CONDICIONES (OR 15 min)")
    print("="*75)
    sub15 = df[df['or_duration_min'] == 15]
    print(f"{'Score':>6} | {'N':>5} | {'%Cont':>6} | {'%Falla':>7} | "
          f"{'%Neut':>7} | {'MFE_post':>8}")
    print("-"*75)
    for score, grp in sub15.groupby('score'):
        n = len(grp)
        if n < 5:
            continue
        n_cont  = (grp['outcome'] == 'continuacion').sum()
        n_falla = (grp['outcome'] == 'falla').sum()
        n_neut  = n - n_cont - n_falla
        mfe     = grp.loc[grp['outcome']=='continuacion', 'mfe_post_pct'].mean()
        print(f"{int(score):>6} | {n:>5,} | {n_cont/n*100:>6.1f} | "
              f"{n_falla/n*100:>7.1f} | {n_neut/n*100:>7.1f} | {mfe:.4f}%")

    # ── Tabla 4: Mejor combo por OR duration ─────────────────────────────────
    print("\n" + "="*100)
    print("MODELO COMBINADO — MEJOR COMBINACIÓN POR OR DURATION")
    print("="*100)
    print(f"{'OR':>5} | {'N':>5} | {'NCond':>5} | {'%Cont':>6} | "
          f"{'%Falla':>7} | {'MFE_post':>8} | Combinación")
    print("-"*100)
    for _, r in por_or_df.iterrows():
        print(f"{int(r['or_duration_min']):>5} | "
              f"{int(r['n']):>5,} | "
              f"{int(r['n_condiciones']):>5} | "
              f"{r['pct_continuacion']:>6.1f} | "
              f"{r['pct_falla']:>7.1f} | "
              f"{r['mfe_post_avg']:>8.4f} | "
              f"{r['condiciones']}")

    print(f"\n{'─'*70}")
    print("ARCHIVOS GENERADOS en orb_results/:")
    print("  modelo_combinado_raw.csv     → todas las combinaciones × OR")
    print("  modelo_combinado_top.csv     → top 20 por %continuación")
    print("  modelo_combinado_resumen.csv → resumen por score")
    print("  modelo_combinado_por_or.csv  → mejor combo por OR duration")
    print(f"{'─'*70}")
    print("\nDEFINICIÓN DE CONDICIONES:")
    for k, v in COND_LABELS.items():
        print(f"  {k}: {v}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Cargando nivel4_raw desde: {INPUT_FILE}")
    df_raw = pd.read_csv(INPUT_FILE)
    print(f"Registros cargados: {len(df_raw):,}")
    print(f"OR durations: {sorted(df_raw['or_duration_min'].unique())}")

    print("\nCalculando condiciones binarias...")
    df = load_and_prepare(INPUT_FILE)
    print(f"Registros con condiciones: {len(df):,}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Resumen por score (todos los OR) ──────────────────────────────────────
    print("Calculando resumen por score...")
    score_df = resumen_por_score(df)
    score_df.to_csv(os.path.join(OUTPUT_DIR, "modelo_combinado_resumen.csv"), index=False)

    # ── Todas las combinaciones sobre todos los OR ────────────────────────────
    print("Evaluando 63 combinaciones × todos los OR durations...")
    all_combos = evaluar_todas_combinaciones(df, or_dur=None)
    all_combos.to_csv(os.path.join(OUTPUT_DIR, "modelo_combinado_raw.csv"), index=False)
    print(f"  Combinaciones válidas (≥{MIN_CASOS} casos): {len(all_combos):,}")

    # Top 20
    top_df = all_combos.head(20)
    top_df.to_csv(os.path.join(OUTPUT_DIR, "modelo_combinado_top.csv"), index=False)

    # ── Mejor combo por OR duration ───────────────────────────────────────────
    print("Calculando mejor combinación por OR duration...")
    por_or_df = mejor_combo_por_or(df)
    por_or_df.to_csv(os.path.join(OUTPUT_DIR, "modelo_combinado_por_or.csv"), index=False)

    # ── Print consola ─────────────────────────────────────────────────────────
    print_consola(score_df, all_combos, por_or_df, df)
    print("\nDone ✓")


if __name__ == "__main__":
    main()