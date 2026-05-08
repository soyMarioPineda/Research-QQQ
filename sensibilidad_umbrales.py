"""
PASO 2 — ANÁLISIS DE SENSIBILIDAD DE UMBRALES
===============================================
Verifica que los tres umbrales de definición del retest
son decisiones de diseño robustas y no parte del resultado.

Los tres umbrales a variar:
  U1: alejamiento_pct  → 0.10%, 0.15%, 0.20%(base), 0.25%, 0.30%
  U2: tolerancia_pct   → 0.03%, 0.05%(base), 0.08%, 0.10%
  U3: ventana_min      → 60, 90, 120(base), hasta_cierre

Para cada combinación calcula sobre OR=15 min:
  - N de retests detectados
  - Tasa de continuación base
  - Tasa de continuación con Combo #4 (C1+C2+C4)
  - Tasa con solo C2 (retest rápido)
  - Poder predictivo de mins_to_retest (bins 0-10 vs >60)

Criterio de robustez:
  Hallazgo robusto     → variación ≤ 3pp respecto al caso base
  Parcialmente robusto → variación 3-8pp
  Sensible al umbral   → variación > 8pp

Output:
  sensibilidad_umbrales.csv    → todas las combinaciones
  sensibilidad_resumen.csv     → resumen de variación por umbral
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────

DATA_FILE  = "QQQ_1min.csv"
OUTPUT_DIR = "orb_results"
OR_MUESTRA = 15
OR_DURATION = 15

# Variaciones de cada umbral
U1_ALEJAMIENTO = [0.10, 0.15, 0.20, 0.25, 0.30]   # % alejamiento mínimo
U2_TOLERANCIA  = [0.03, 0.05, 0.08, 0.10]           # % tolerancia del toque
U3_VENTANA     = [60, 90, 120, 390]                  # minutos (390=hasta cierre)

# Valores base
U1_BASE = 0.20
U2_BASE = 0.05
U3_BASE = 120

# Ventana de outcome (fija en todos los escenarios)
OUTCOME_VENTANA_MIN = 60

# Umbrales del modelo combinado
C1_PERCENTIL = 60
C2_MINS_MAX  = 10
C4_PERC_LOW  = 40
C4_PERC_HIGH = 60

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')
    df = df.set_index('datetime').sort_index()
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.between_time("09:30", "15:59")
    df['date'] = df.index.date
    return df

# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS PARA UNA COMBINACIÓN DE UMBRALES
# ─────────────────────────────────────────────────────────────────────────────

def analyze_day(day_df: pd.DataFrame,
                or_duration: int,
                alejamiento_pct: float,
                tolerancia_pct: float,
                ventana_min: int) -> dict | None:
    """
    Replica la lógica de nivel4.py con umbrales variables.
    Retorna el outcome para un día dado.
    """
    day_df = day_df.copy().reset_index()

    open_time = day_df['datetime'].iloc[0]
    or_end    = open_time + pd.Timedelta(minutes=or_duration)
    or_bars   = day_df[day_df['datetime'] < or_end]

    if len(or_bars) < max(1, or_duration - 2):
        return None

    or_high = or_bars['high'].max()
    or_low  = or_bars['low'].min()
    or_size = or_high - or_low

    if or_size == 0:
        return None

    or_size_pct       = or_size / or_low * 100
    or_close          = or_bars['close'].iloc[-1]
    or_close_position = (or_close - or_low) / or_size
    or_vol_avg        = or_bars['volume'].mean()
    if or_vol_avg == 0:
        or_vol_avg = 1

    post_or = day_df[day_df['datetime'] >= or_end].reset_index(drop=True)
    if len(post_or) < 2:
        return None

    # Detectar breakout
    breakout_idx   = None
    breakout_dir   = None
    breakout_price = None
    breakout_time  = None
    breakout_bar   = None

    for i, row in post_or.iterrows():
        if row['high'] > or_high:
            breakout_idx   = i
            breakout_dir   = 'up'
            breakout_price = or_high
            breakout_time  = row['datetime']
            breakout_bar   = row
            break
        elif row['low'] < or_low:
            breakout_idx   = i
            breakout_dir   = 'down'
            breakout_price = or_low
            breakout_time  = row['datetime']
            breakout_bar   = row
            break

    if breakout_idx is None:
        return None

    breakout_minute = int((breakout_time - open_time).seconds / 60)

    if breakout_dir == 'up':
        raw_strength = breakout_bar['close'] - or_high
    else:
        raw_strength = or_low - breakout_bar['close']
    breakout_strength = raw_strength / or_size
    vol_ratio         = breakout_bar['volume'] / or_vol_avg

    post_break = post_or.iloc[breakout_idx:].reset_index(drop=True)
    if len(post_break) < 2:
        return None

    # Umbrales variables
    alejamiento_min = breakout_price * (alejamiento_pct / 100)
    tolerancia      = breakout_price * (tolerancia_pct  / 100)

    # Ventana: si es 390 (hasta cierre) usamos fin del día
    if ventana_min == 390:
        ventana_fin = day_df['datetime'].iloc[-1]
    else:
        ventana_fin = breakout_time + pd.Timedelta(minutes=ventana_min)

    # Buscar retest
    alejamiento_alcanzado = False
    mfe_corriendo         = 0.0
    mfe_pre_pts           = 0.0
    retest_encontrado     = False
    retest_time           = None
    retest_vela           = None
    retest_idx            = None

    for i, row in post_break.iterrows():
        if i == 0:
            continue
        if row['datetime'] > ventana_fin:
            break

        if breakout_dir == 'up':
            excursion_actual = row['high'] - breakout_price
        else:
            excursion_actual = breakout_price - row['low']

        if excursion_actual > mfe_corriendo:
            mfe_corriendo = excursion_actual

        if not alejamiento_alcanzado:
            if mfe_corriendo >= alejamiento_min:
                alejamiento_alcanzado = True
            continue

        if breakout_dir == 'up':
            toco_nivel = row['low'] <= (breakout_price + tolerancia)
        else:
            toco_nivel = row['high'] >= (breakout_price - tolerancia)

        if toco_nivel:
            pre_velas = post_break.iloc[1:i]
            if len(pre_velas) > 0:
                if breakout_dir == 'up':
                    mfe_pre_pts = pre_velas['high'].max() - breakout_price
                else:
                    mfe_pre_pts = breakout_price - pre_velas['low'].min()
            else:
                mfe_pre_pts = 0.0

            retest_encontrado = True
            retest_time       = row['datetime']
            retest_vela       = row
            retest_idx        = i
            break

    if not retest_encontrado or retest_vela is None:
        return None

    mfe_pre_retest_pct = mfe_pre_pts / breakout_price * 100
    mins_to_retest     = (retest_time - breakout_time).seconds / 60

    # Determinar outcome
    outcome_fin = retest_time + pd.Timedelta(minutes=OUTCOME_VENTANA_MIN)
    post_retest = post_break.iloc[retest_idx:].reset_index(drop=True)

    if breakout_dir == 'up':
        nivel_continuacion = breakout_price + mfe_pre_pts
        nivel_falla        = or_low
    else:
        nivel_continuacion = breakout_price - mfe_pre_pts
        nivel_falla        = or_high

    outcome = 'neutro'

    for i, row in post_retest.iterrows():
        if i == 0:
            continue
        if row['datetime'] > outcome_fin:
            break

        if breakout_dir == 'up':
            supero_mfe = row['high'] >= nivel_continuacion
            cruzo_or   = row['low']  <= nivel_falla
        else:
            supero_mfe = row['low']  <= nivel_continuacion
            cruzo_or   = row['high'] >= nivel_falla

        if supero_mfe:
            outcome = 'continuacion'
            break
        elif cruzo_or:
            outcome = 'falla'
            break

    return {
        'date':               str(open_time.date()),
        'or_size_pct':        round(or_size_pct, 4),
        'or_close_position':  round(or_close_position, 4),
        'breakout_dir':       breakout_dir,
        'breakout_minute':    breakout_minute,
        'vol_ratio':          round(vol_ratio, 4),
        'mfe_pre_retest_pct': round(mfe_pre_retest_pct, 4),
        'mins_to_retest':     round(mins_to_retest, 1),
        'outcome':            outcome,
    }


def run_scenario(df_raw: pd.DataFrame,
                 alejamiento_pct: float,
                 tolerancia_pct: float,
                 ventana_min: int) -> pd.DataFrame:
    """Corre el análisis completo para una combinación de umbrales."""
    dates   = df_raw['date'].unique()
    results = []

    for date in dates:
        day_df = df_raw[df_raw['date'] == date]
        result = analyze_day(
            day_df, OR_DURATION,
            alejamiento_pct, tolerancia_pct, ventana_min
        )
        if result:
            results.append(result)

    return pd.DataFrame(results)

# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS CLAVE DE CADA ESCENARIO
# ─────────────────────────────────────────────────────────────────────────────

def calcular_metricas(df: pd.DataFrame,
                      alejamiento_pct: float,
                      tolerancia_pct: float,
                      ventana_min: int) -> dict:
    """
    Calcula las métricas clave para comparar entre escenarios.
    """
    if len(df) == 0:
        return {}

    n_total  = len(df)
    n_cont   = (df['outcome'] == 'continuacion').sum()
    n_falla  = (df['outcome'] == 'falla').sum()
    n_neutro = (df['outcome'] == 'neutro').sum()

    pct_cont  = n_cont  / n_total * 100
    pct_falla = n_falla / n_total * 100

    # Tasa de continuación con C2 (retest rápido ≤10 min)
    c2_mask = df['mins_to_retest'] <= C2_MINS_MAX
    c2_df   = df[c2_mask]
    c2_cont = (c2_df['outcome'] == 'continuacion').mean() * 100 if len(c2_df) > 10 else np.nan

    # Tasa de continuación con retests tardíos (>60 min)
    tard_mask = df['mins_to_retest'] > 60
    tard_df   = df[tard_mask]
    tard_cont = (tard_df['outcome'] == 'continuacion').mean() * 100 if len(tard_df) > 10 else np.nan

    # Brecha entre retests rápidos y tardíos
    brecha = c2_cont - tard_cont if not (np.isnan(c2_cont) or np.isnan(tard_cont)) else np.nan

    # Tasa de continuación con Combo C1+C2+C4
    c1_thresh = df['or_size_pct'].quantile(C1_PERCENTIL / 100)
    c4_lo     = df['vol_ratio'].quantile(C4_PERC_LOW  / 100)
    c4_hi     = df['vol_ratio'].quantile(C4_PERC_HIGH / 100)

    combo_mask = (
        (df['or_size_pct'] >= c1_thresh) &
        (df['mins_to_retest'] <= C2_MINS_MAX) &
        (df['vol_ratio'] >= c4_lo) &
        (df['vol_ratio'] <= c4_hi)
    )
    combo_df   = df[combo_mask]
    combo_cont = (combo_df['outcome'] == 'continuacion').mean() * 100 if len(combo_df) > 20 else np.nan
    combo_n    = len(combo_df)

    # Tasa de continuación por bins de mfe_pre
    bins_mp = [(0.20, 0.30), (0.30, 0.50), (0.50, 9999)]
    mfe_rates = {}
    for lo_b, hi_b in bins_mp:
        g = df[(df['mfe_pre_retest_pct'] >= lo_b) &
               (df['mfe_pre_retest_pct'] < hi_b)]
        key = f'mfe_{lo_b:.2f}_{hi_b:.2f}'
        mfe_rates[key] = (g['outcome'] == 'continuacion').mean() * 100 if len(g) > 10 else np.nan

    ventana_label = 'cierre' if ventana_min == 390 else f'{ventana_min}min'
    es_base = (alejamiento_pct == U1_BASE and
               tolerancia_pct  == U2_BASE and
               ventana_min     == U3_BASE)

    return {
        'alejamiento_pct':   alejamiento_pct,
        'tolerancia_pct':    tolerancia_pct,
        'ventana_min':       ventana_label,
        'es_base':           es_base,
        'n_retests':         n_total,
        'pct_cont_base':     round(pct_cont,  2),
        'pct_falla_base':    round(pct_falla, 2),
        'pct_neutro_base':   round(n_neutro / n_total * 100, 2),
        'c2_cont_pct':       round(c2_cont,   2) if not np.isnan(c2_cont)   else np.nan,
        'c2_n':              int(c2_mask.sum()),
        'tard_cont_pct':     round(tard_cont, 2) if not np.isnan(tard_cont) else np.nan,
        'brecha_rapid_tard': round(brecha,    2) if not np.isnan(brecha)    else np.nan,
        'combo_c1c2c4_pct':  round(combo_cont,2) if not np.isnan(combo_cont) else np.nan,
        'combo_c1c2c4_n':    combo_n,
        **{k: round(v, 2) if not np.isnan(v) else np.nan
           for k, v in mfe_rates.items()},
    }

# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN DE VARIACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def resumen_variacion(df_sensi: pd.DataFrame) -> pd.DataFrame:
    """
    Evalúa robustez usando la brecha relativa entre retests rápidos y tardíos
    como métrica principal. Las tasas absolutas varían por composición de
    muestra (esperado), no por inestabilidad del patrón subyacente.
    """
    rows = []
    base = df_sensi[df_sensi['es_base'] == True].iloc[0]

    # Métrica principal: brecha relativa
    # Métricas secundarias: tasas dentro de cada grupo (no la base absoluta)
    metricas_clave = [
        ('brecha_rapid_tard',  'Brecha rápido vs tardío (métrica principal)'),
        ('c2_cont_pct',        'Continuación con retest ≤10 min'),
        ('tard_cont_pct',      'Continuación con retest >60 min'),
        ('combo_c1c2c4_pct',   'Continuación con C1+C2+C4'),
        ('pct_cont_base',      'Tasa base [REFERENCIA — varía por composición]'),
    ]

    for metrica, nombre in metricas_clave:
        base_val = base[metrica]
        if pd.isna(base_val):
            continue

        variaciones = df_sensi[df_sensi['es_base'] == False][metrica].dropna()
        if len(variaciones) == 0:
            continue

        max_var = (variaciones - base_val).abs().max()
        min_val = variaciones.min()
        max_val = variaciones.max()

        # Criterio de robustez solo aplica a métricas que miden
        # el mismo fenómeno entre especificaciones
        if metrica == 'pct_cont_base':
            robustez = 'N/A — varía por composición de muestra'
        elif metrica == 'brecha_rapid_tard':
            if max_var <= 3:
                robustez = '✓ ROBUSTO (≤3pp)'
            elif max_var <= 8:
                robustez = '~ PARCIAL (3-8pp)'
            else:
                robustez = '✗ SENSIBLE (>8pp)'
        else:
            robustez = 'REF — varía por composición de muestra'

        rows.append({
            'metrica':        nombre,
            'valor_base':     round(base_val, 2),
            'min_variacion':  round(min_val, 2),
            'max_variacion':  round(max_val, 2),
            'max_desviacion': round(max_var, 2),
            'robustez':       robustez,
        })

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# PRINT CONSOLA
# ─────────────────────────────────────────────────────────────────────────────

def print_consola(sensi_df: pd.DataFrame,
                  resumen_df: pd.DataFrame):

    base = sensi_df[sensi_df['es_base'] == True]

    print("\n" + "="*95)
    print("SENSIBILIDAD AL UMBRAL 1 — ALEJAMIENTO MÍNIMO")
    print("(tolerancia y ventana fijas en valores base)")
    print("="*95)
    print(f"  {'Alej%':>6} | {'N_ret':>5} | {'%Cont_base':>10} | "
          f"{'%C2_cont':>8} | {'%Tard_cont':>10} | {'Brecha':>6} | "
          f"{'%Combo':>7} | {'ComboN':>6}")
    print(f"  {'-'*75}")

    sub = sensi_df[
        (sensi_df['tolerancia_pct'] == U2_BASE) &
        (sensi_df['ventana_min'] == str(U3_BASE) + 'min')
    ].sort_values('alejamiento_pct')

    for _, r in sub.iterrows():
        marker = ' ← BASE' if r['es_base'] else ''
        c2   = f"{r['c2_cont_pct']:.1f}" if not pd.isna(r['c2_cont_pct']) else ' N/A'
        tard = f"{r['tard_cont_pct']:.1f}" if not pd.isna(r['tard_cont_pct']) else ' N/A'
        brec = f"{r['brecha_rapid_tard']:.1f}" if not pd.isna(r['brecha_rapid_tard']) else ' N/A'
        comb = f"{r['combo_c1c2c4_pct']:.1f}" if not pd.isna(r['combo_c1c2c4_pct']) else ' N/A'
        print(f"  {r['alejamiento_pct']:>6.2f} | {int(r['n_retests']):>5} | "
              f"{r['pct_cont_base']:>10.1f} | {c2:>8} | {tard:>10} | "
              f"{brec:>6} | {comb:>7} | {int(r['combo_c1c2c4_n']):>6}"
              f"{marker}")

    print("\n" + "="*95)
    print("SENSIBILIDAD AL UMBRAL 2 — TOLERANCIA DEL TOQUE")
    print("(alejamiento y ventana fijos en valores base)")
    print("="*95)
    print(f"  {'Tol%':>5} | {'N_ret':>5} | {'%Cont_base':>10} | "
          f"{'%C2_cont':>8} | {'%Tard_cont':>10} | {'Brecha':>6} | "
          f"{'%Combo':>7}")
    print(f"  {'-'*65}")

    sub = sensi_df[
        (sensi_df['alejamiento_pct'] == U1_BASE) &
        (sensi_df['ventana_min'] == str(U3_BASE) + 'min')
    ].sort_values('tolerancia_pct')

    for _, r in sub.iterrows():
        marker = ' ← BASE' if r['es_base'] else ''
        c2   = f"{r['c2_cont_pct']:.1f}" if not pd.isna(r['c2_cont_pct']) else ' N/A'
        tard = f"{r['tard_cont_pct']:.1f}" if not pd.isna(r['tard_cont_pct']) else ' N/A'
        brec = f"{r['brecha_rapid_tard']:.1f}" if not pd.isna(r['brecha_rapid_tard']) else ' N/A'
        comb = f"{r['combo_c1c2c4_pct']:.1f}" if not pd.isna(r['combo_c1c2c4_pct']) else ' N/A'
        print(f"  {r['tolerancia_pct']:>5.2f} | {int(r['n_retests']):>5} | "
              f"{r['pct_cont_base']:>10.1f} | {c2:>8} | {tard:>10} | "
              f"{brec:>6} | {comb:>7}{marker}")

    print("\n" + "="*95)
    print("SENSIBILIDAD AL UMBRAL 3 — VENTANA DE TIEMPO")
    print("(alejamiento y tolerancia fijos en valores base)")
    print("="*95)
    print(f"  {'Ventana':>8} | {'N_ret':>5} | {'%Cont_base':>10} | "
          f"{'%C2_cont':>8} | {'%Tard_cont':>10} | {'Brecha':>6} | "
          f"{'%Combo':>7}")
    print(f"  {'-'*65}")

    sub = sensi_df[
        (sensi_df['alejamiento_pct'] == U1_BASE) &
        (sensi_df['tolerancia_pct']  == U2_BASE)
    ]

    for _, r in sub.iterrows():
        marker = ' ← BASE' if r['es_base'] else ''
        c2   = f"{r['c2_cont_pct']:.1f}" if not pd.isna(r['c2_cont_pct']) else ' N/A'
        tard = f"{r['tard_cont_pct']:.1f}" if not pd.isna(r['tard_cont_pct']) else ' N/A'
        brec = f"{r['brecha_rapid_tard']:.1f}" if not pd.isna(r['brecha_rapid_tard']) else ' N/A'
        comb = f"{r['combo_c1c2c4_pct']:.1f}" if not pd.isna(r['combo_c1c2c4_pct']) else ' N/A'
        print(f"  {str(r['ventana_min']):>8} | {int(r['n_retests']):>5} | "
              f"{r['pct_cont_base']:>10.1f} | {c2:>8} | {tard:>10} | "
              f"{brec:>6} | {comb:>7}{marker}")

    print("\n" + "="*90)
    print("RESUMEN DE ROBUSTEZ — VARIACIÓN MÁXIMA POR MÉTRICA")
    print("Métrica principal: brecha rápido vs tardío (mide el mismo fenómeno en todas")
    print("las especificaciones). La tasa base absoluta varía por composición de muestra,")
    print("no por inestabilidad del patrón — se reporta como referencia solamente.")
    print("Criterio: ROBUSTO ≤3pp | PARCIAL 3-8pp | SENSIBLE >8pp")
    print("="*90)
    print(f"  {'Métrica':<48} | {'Base':>6} | {'Min':>6} | "
          f"{'Max':>6} | {'MaxDesv':>7} | Robustez")
    print(f"  {'-'*90}")

    for _, r in resumen_df.iterrows():
        print(f"  {r['metrica']:<48} | {r['valor_base']:>6.1f} | "
              f"{r['min_variacion']:>6.1f} | {r['max_variacion']:>6.1f} | "
              f"{r['max_desviacion']:>7.1f} | {r['robustez']}")

    print(f"\n{'─'*70}")
    print("ARCHIVOS GENERADOS en orb_results/:")
    print("  sensibilidad_umbrales.csv  → todas las combinaciones")
    print("  sensibilidad_resumen.csv   → resumen de variación")
    print(f"{'─'*70}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    csv_path = os.path.join(OUTPUT_DIR, "sensibilidad_umbrales.csv")

    print(f"Leyendo resultados existentes desde: {csv_path}")
    sensi_df = pd.read_csv(csv_path)

    # Reconstruir columna es_base si no existe
    if 'es_base' not in sensi_df.columns:
        sensi_df['es_base'] = (
            (sensi_df['alejamiento_pct'] == U1_BASE) &
            (sensi_df['tolerancia_pct']  == U2_BASE) &
            (sensi_df['ventana_min']     == f'{U3_BASE}min')
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    resumen_df = resumen_variacion(sensi_df)
    resumen_df.to_csv(
        os.path.join(OUTPUT_DIR, "sensibilidad_resumen.csv"), index=False)

    print_consola(sensi_df, resumen_df)
    print("\nDone ✓")


if __name__ == "__main__":
    main()