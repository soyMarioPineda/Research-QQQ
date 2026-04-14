"""
ORB NIVEL 1 — Anatomía del Rango
=================================
Variables calculadas por día × duración de OR:

  1. OR_size_pct       = tamaño del rango en % del precio
  2. OR_close_position = dónde cerró el precio dentro del rango (0=abajo, 1=arriba)
  3. gap_pct           = gap overnight en %
  4. MFE_total_pct     = excursión máxima del breakout (para correlacionar con el rango)

Output:
  - orb_results/nivel1_raw.csv     → un registro por (día × OR_duration)
  - orb_results/nivel1_summary.csv → estadísticas agrupadas por quintil de OR_size_pct
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

# Duraciones del OR en minutos (5 a 120, paso de 5)
OR_DURATIONS = list(range(5, 125, 5))

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    print(f"Cargando {filepath} ...")
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')
    df = df.set_index('datetime').sort_index()
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.between_time('09:30', '15:59')
    df['date'] = df.index.date
    print(f"  Filas totales:  {len(df):,}")
    print(f"  Días únicos:    {df['date'].nunique():,}")
    print(f"  Rango de fechas: {df.index[0].date()} → {df.index[-1].date()}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# NIVEL 1 — ANÁLISIS POR DÍA × OR_DURATION
# ─────────────────────────────────────────────────────────────────────────────

def analyze_level1(day_df: pd.DataFrame, or_duration: int,
                   prev_close: float | None) -> dict | None:
    """
    Para un día y una duración de OR, calcula todas las variables del Nivel 1.

    Args:
        day_df     : DataFrame filtrado para ese día (horario regular)
        or_duration: minutos del Opening Range
        prev_close : precio de cierre del día anterior (para gap)

    Returns:
        dict con variables del Nivel 1, o None si no hay suficientes datos
    """
    day_df = day_df.copy().reset_index()
    date   = str(day_df['datetime'].iloc[0].date())

    # ── Velas del OR ──────────────────────────────────────────────────────────
    or_mask = (
        (day_df['datetime'].dt.hour == 9) &
        (day_df['datetime'].dt.minute >= 30) &
        (day_df['datetime'].dt.minute < 30 + or_duration)
    )
    or_bars = day_df[or_mask]

    if len(or_bars) < max(1, or_duration - 2):   # tolerancia de 2 velas faltantes
        return None

    or_high  = or_bars['high'].max()
    or_low   = or_bars['low'].min()
    or_size  = or_high - or_low
    or_close = or_bars['close'].iloc[-1]   # último close dentro del OR

    if or_size <= 0:
        return None

    # ── Variables del Nivel 1 ─────────────────────────────────────────────────

    # 1.1 Tamaño relativo del rango
    or_size_pct = (or_size / or_low) * 100

    # 1.2 Posición del cierre dentro del rango (0=fondo, 1=techo)
    or_close_position = (or_close - or_low) / or_size

    # 1.3 Gap overnight
    open_price = day_df['open'].iloc[0]
    if prev_close and prev_close > 0:
        gap_pct = ((open_price - prev_close) / prev_close) * 100
    else:
        gap_pct = np.nan

    # ── Velas post-OR ─────────────────────────────────────────────────────────
    or_end_time = or_bars['datetime'].iloc[-1]
    post_or     = day_df[day_df['datetime'] > or_end_time].reset_index(drop=True)

    if len(post_or) < 2:
        return None

    # ── Detectar primer breakout ──────────────────────────────────────────────
    breakout_idx   = None
    breakout_dir   = None
    breakout_price = None

    for i, row in post_or.iterrows():
        if row['high'] > or_high:
            breakout_idx   = i
            breakout_dir   = 'up'
            breakout_price = or_high
            break
        elif row['low'] < or_low:
            breakout_idx   = i
            breakout_dir   = 'down'
            breakout_price = or_low
            break

    if breakout_idx is None:
        # No hubo breakout → igual guardamos las variables del rango
        return {
            'date':               date,
            'or_duration_min':    or_duration,
            'or_high':            round(or_high, 4),
            'or_low':             round(or_low, 4),
            'or_size_pts':        round(or_size, 4),
            'or_size_pct':        round(or_size_pct, 4),
            'or_close_position':  round(or_close_position, 4),
            'gap_pct':            round(gap_pct, 4) if not np.isnan(gap_pct) else np.nan,
            'breakout_dir':       'none',
            'breakout_minute':    np.nan,
            'breakout_strength':  np.nan,
            'mfe_total_pts':      np.nan,
            'mfe_total_pct':      np.nan,
        }

    # ── MFE total del breakout ────────────────────────────────────────────────
    post_break     = post_or.iloc[breakout_idx:].reset_index(drop=True)
    breakout_time  = post_break['datetime'].iloc[0]
    open_time      = day_df['datetime'].iloc[0]
    breakout_minute = int((breakout_time - open_time).seconds / 60)

    if breakout_dir == 'up':
        mfe_total_pts = post_break['high'].max() - breakout_price
        # Fuerza: cuánto cerró la vela de break por encima del nivel
        breakout_strength = (post_break['close'].iloc[0] - or_high) / or_size
    else:
        mfe_total_pts = breakout_price - post_break['low'].min()
        breakout_strength = (or_low - post_break['close'].iloc[0]) / or_size

    mfe_total_pct = (mfe_total_pts / breakout_price) * 100

    return {
        'date':               date,
        'or_duration_min':    or_duration,
        'or_high':            round(or_high, 4),
        'or_low':             round(or_low, 4),
        'or_size_pts':        round(or_size, 4),
        'or_size_pct':        round(or_size_pct, 4),
        'or_close_position':  round(or_close_position, 4),
        'gap_pct':            round(gap_pct, 4) if not np.isnan(gap_pct) else np.nan,
        'breakout_dir':       breakout_dir,
        'breakout_minute':    breakout_minute,
        'breakout_strength':  round(breakout_strength, 4),
        'mfe_total_pts':      round(mfe_total_pts, 4),
        'mfe_total_pct':      round(mfe_total_pct, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN POR QUINTILES DE OR_SIZE_PCT
# ─────────────────────────────────────────────────────────────────────────────

def summarize_level1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada duración de OR, divide los días en 5 quintiles según OR_size_pct
    y calcula estadísticas del MFE en cada quintil.
    Responde: ¿rangos más grandes o más pequeños producen mejores breakouts?
    """
    results = []
    df_valid = df[df['breakout_dir'] != 'none'].copy()

    for or_dur, grp in df_valid.groupby('or_duration_min'):
        if len(grp) < 50:
            continue

        # Quintiles de tamaño de rango
        grp['quintil'] = pd.qcut(grp['or_size_pct'], q=5,
                                  labels=['Q1_mini', 'Q2_pequeño',
                                          'Q3_medio', 'Q4_grande', 'Q5_enorme'])

        for q, qgrp in grp.groupby('quintil', observed=True):
            results.append({
                'or_duration_min':      or_dur,
                'quintil_or_size':      str(q),
                'or_size_pct_min':      round(qgrp['or_size_pct'].min(), 4),
                'or_size_pct_max':      round(qgrp['or_size_pct'].max(), 4),
                'or_size_pct_median':   round(qgrp['or_size_pct'].median(), 4),
                'n_dias':               len(qgrp),
                'pct_up':               round((qgrp['breakout_dir']=='up').mean()*100, 1),
                'pct_down':             round((qgrp['breakout_dir']=='down').mean()*100, 1),
                'avg_mfe_pct':          round(qgrp['mfe_total_pct'].mean(), 4),
                'median_mfe_pct':       round(qgrp['mfe_total_pct'].median(), 4),
                'avg_mfe_pts':          round(qgrp['mfe_total_pts'].mean(), 4),
                'avg_breakout_minute':  round(qgrp['breakout_minute'].mean(), 1),
                'avg_breakout_strength':round(qgrp['breakout_strength'].mean(), 4),
                # OR_close_position: predice dirección?
                'avg_close_pos_en_ups': round(qgrp[qgrp['breakout_dir']=='up']['or_close_position'].mean(), 3),
                'avg_close_pos_en_downs': round(qgrp[qgrp['breakout_dir']=='down']['or_close_position'].mean(), 3),
            })

    return pd.DataFrame(results)


def summarize_gap_effect(df: pd.DataFrame) -> pd.DataFrame:
    """
    ¿El gap overnight modera la dirección y fuerza del breakout?
    """
    df_valid = df[df['breakout_dir'] != 'none'].copy()
    df_valid = df_valid.dropna(subset=['gap_pct'])

    results = []
    for or_dur, grp in df_valid.groupby('or_duration_min'):
        if len(grp) < 50:
            continue

        gap_up   = grp[grp['gap_pct'] > 0.1]
        gap_down = grp[grp['gap_pct'] < -0.1]
        gap_flat = grp[(grp['gap_pct'] >= -0.1) & (grp['gap_pct'] <= 0.1)]

        for label, sub in [('gap_alcista', gap_up),
                            ('gap_bajista', gap_down),
                            ('gap_plano',   gap_flat)]:
            if len(sub) < 10:
                continue
            results.append({
                'or_duration_min': or_dur,
                'gap_tipo':        label,
                'n_dias':          len(sub),
                'pct_break_up':    round((sub['breakout_dir']=='up').mean()*100, 1),
                'pct_break_down':  round((sub['breakout_dir']=='down').mean()*100, 1),
                'avg_mfe_pct':     round(sub['mfe_total_pct'].mean(), 4),
                'median_mfe_pct':  round(sub['mfe_total_pct'].median(), 4),
            })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    df = load_data(DATA_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dates      = sorted(df['date'].unique())
    all_results = []
    total      = len(dates) * len(OR_DURATIONS)
    counter    = 0

    # Precalcular cierres del día anterior
    # (necesitamos el close de ayer para calcular el gap de hoy)
    daily_closes = {}
    for date in dates:
        day_df = df[df['date'] == date]
        daily_closes[date] = day_df['close'].iloc[-1]

    dates_list = list(dates)

    print(f"Analizando {len(dates_list):,} días × {len(OR_DURATIONS)} duraciones de OR...")
    print(f"Total combinaciones: {total:,}\n")

    for i, date in enumerate(dates_list):
        day_df     = df[df['date'] == date]
        prev_close = daily_closes.get(dates_list[i-1]) if i > 0 else None

        for or_dur in OR_DURATIONS:
            result = analyze_level1(day_df, or_dur, prev_close)
            if result:
                all_results.append(result)
            counter += 1
            if counter % 1000 == 0:
                pct = counter / total * 100
                print(f"  {counter:,}/{total:,} ({pct:.1f}%) — registros válidos: {len(all_results):,}")

    raw_df = pd.DataFrame(all_results)

    # ── Guardar raw ───────────────────────────────────────────────────────────
    raw_path = os.path.join(OUTPUT_DIR, "nivel1_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"\nRaw guardado: {raw_path}  ({len(raw_df):,} filas)")

    # ── Resumen por quintiles ─────────────────────────────────────────────────
    summary_df = summarize_level1(raw_df)
    summary_path = os.path.join(OUTPUT_DIR, "nivel1_summary_quintiles.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Resumen quintiles: {summary_path}")

    # ── Efecto del gap ────────────────────────────────────────────────────────
    gap_df = summarize_gap_effect(raw_df)
    gap_path = os.path.join(OUTPUT_DIR, "nivel1_gap_effect.csv")
    gap_df.to_csv(gap_path, index=False)
    print(f"Efecto gap:        {gap_path}")

    # ── Imprimir resumen en consola (OR de 15 min como ejemplo) ──────────────
    print("\n" + "="*85)
    print("NIVEL 1 — MUESTRA: OR de 15 minutos — MFE por quintil de tamaño de rango")
    print("="*85)

    sample = summary_df[summary_df['or_duration_min'] == 15]
    if len(sample) > 0:
        print(f"\n{'Quintil':<14} | {'OR_size%':>8} | {'N':>5} | "
              f"{'%Up':>5} | {'%Down':>5} | {'MFE_avg%':>8} | {'MFE_med%':>8} | "
              f"{'ClosePos_Up':>11} | {'ClosePos_Dn':>11}")
        print("-"*95)
        for _, row in sample.iterrows():
            print(
                f"{row['quintil_or_size']:<14} | "
                f"{row['or_size_pct_median']:>8.4f} | "
                f"{int(row['n_dias']):>5} | "
                f"{row['pct_up']:>5.1f} | "
                f"{row['pct_down']:>5.1f} | "
                f"{row['avg_mfe_pct']:>8.4f} | "
                f"{row['median_mfe_pct']:>8.4f} | "
                f"{row['avg_close_pos_en_ups']:>11.3f} | "
                f"{row['avg_close_pos_en_downs']:>11.3f}"
            )

    print("\n" + "="*85)
    print("NIVEL 1 — EFECTO DEL GAP (OR 15 min)")
    print("="*85)
    gap_sample = gap_df[gap_df['or_duration_min'] == 15]
    if len(gap_sample) > 0:
        print(f"\n{'Gap tipo':<14} | {'N':>5} | {'%BreakUp':>8} | "
              f"{'%BreakDn':>8} | {'MFE_avg%':>8} | {'MFE_med%':>8}")
        print("-"*60)
        for _, row in gap_sample.iterrows():
            print(
                f"{row['gap_tipo']:<14} | "
                f"{int(row['n_dias']):>5} | "
                f"{row['pct_break_up']:>8.1f} | "
                f"{row['pct_break_down']:>8.1f} | "
                f"{row['avg_mfe_pct']:>8.4f} | "
                f"{row['median_mfe_pct']:>8.4f}"
            )

    print("""
─────────────────────────────────────────────────────
COLUMNAS CLAVE DEL nivel1_raw.csv:
  or_size_pct        → tamaño del rango en % del precio
  or_close_position  → 0=cerró en el fondo, 1=cerró en el techo del OR
  gap_pct            → gap overnight en %
  breakout_dir       → 'up', 'down', o 'none' (no hubo breakout)
  breakout_minute    → minutos desde apertura hasta el breakout
  breakout_strength  → qué tan fuerte penetró el nivel (normalizado por OR_size)
  mfe_total_pct      → excursión máxima en dirección del break en %
─────────────────────────────────────────────────────
Done ✓
""")


if __name__ == "__main__":
    main()