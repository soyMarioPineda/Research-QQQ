"""
NIVEL 3 — Anatomía del Retest
==============================
Para cada (día × OR_duration) que tuvo breakout, detecta el PRIMER retest
válido según las 3 condiciones definidas:

  CONDICIÓN 1 — Alejamiento mínimo: 0.20% del precio del breakout
  CONDICIÓN 2 — Tolerancia del toque: ±0.05% del precio del breakout
  CONDICIÓN 3 — Ventana de tiempo: 120 minutos post-breakout

Variables medidas:
  3.1  mfe_pre_retest_pct  → máximo alcanzado ANTES de volver al nivel (%)
  3.2  mins_to_retest      → minutos desde breakout hasta el retest
  3.3  retest_speed        → mfe_pre_retest_pct / mins_to_retest
  3.4  retest_depth_pct    → cuánto penetró el nivel en la vela del retest (%)

Outputs:
  nivel3_raw.csv           → un registro por (día × OR_duration)
  nivel3_summary.csv       → resumen estadístico por OR_duration
  nivel3_mfe_pre_bins.csv  → tasa de retest por bin de MFE_pre
  nivel3_tiempo_bins.csv   → distribución del tiempo al retest
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────

DATA_FILE    = "QQQ_1min.csv"
OUTPUT_DIR   = "orb_results"
OR_DURATIONS = list(range(5, 125, 5))

# Condiciones del retest
ALEJAMIENTO_PCT  = 0.20   # Condición 1: mínimo 0.20% de alejamiento
TOLERANCIA_PCT   = 0.05   # Condición 2: zona ±0.05% alrededor del breakout
VENTANA_MIN      = 120    # Condición 3: máximo 120 minutos post-breakout

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
# ANÁLISIS POR DÍA Y POR DURACIÓN DE OR
# ─────────────────────────────────────────────────────────────────────────────

def analyze_day(day_df: pd.DataFrame, or_duration: int) -> dict | None:
    day_df = day_df.copy().reset_index()

    # ── Opening Range ─────────────────────────────────────────────────────────
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

    # ── Velas post-OR ─────────────────────────────────────────────────────────
    post_or = day_df[day_df['datetime'] >= or_end].reset_index(drop=True)

    if len(post_or) < 2:
        return None

    # ── Detectar primer breakout ──────────────────────────────────────────────
    breakout_idx   = None
    breakout_dir   = None
    breakout_price = None
    breakout_time  = None

    for i, row in post_or.iterrows():
        if row['high'] > or_high:
            breakout_idx   = i
            breakout_dir   = 'up'
            breakout_price = or_high
            breakout_time  = row['datetime']
            break
        elif row['low'] < or_low:
            breakout_idx   = i
            breakout_dir   = 'down'
            breakout_price = or_low
            breakout_time  = row['datetime']
            break

    if breakout_idx is None:
        return None

    breakout_minute = int((breakout_time - open_time).seconds / 60)

    # ── Velas desde el breakout ───────────────────────────────────────────────
    post_break = post_or.iloc[breakout_idx:].reset_index(drop=True)

    if len(post_break) < 2:
        return None

    # ── Umbrales del retest ───────────────────────────────────────────────────
    alejamiento_min = breakout_price * (ALEJAMIENTO_PCT / 100)
    tolerancia      = breakout_price * (TOLERANCIA_PCT  / 100)
    ventana_fin     = breakout_time + pd.Timedelta(minutes=VENTANA_MIN)

    # ── Máximo alcanzado en toda la sesión post-breakout (MFE total) ──────────
    if breakout_dir == 'up':
        mfe_total_price = post_break['high'].max()
        mfe_total_pts   = mfe_total_price - breakout_price
    else:
        mfe_total_price = post_break['low'].min()
        mfe_total_pts   = breakout_price - mfe_total_price

    mfe_total_pct = mfe_total_pts / breakout_price * 100

    # ─────────────────────────────────────────────────────────────────────────
    # LÓGICA DEL RETEST — vela por vela
    # ─────────────────────────────────────────────────────────────────────────
    # Estado de la búsqueda
    alejamiento_alcanzado = False   # ¿ya se alejó 0.20%?
    mfe_pre_retest_pts    = 0.0     # máximo alcanzado antes del retest
    mfe_corriendo         = 0.0     # tracking del máximo corriente

    retest_encontrado     = False
    retest_time           = None
    retest_vela           = None

    for i, row in post_break.iterrows():
        if i == 0:
            continue  # saltamos la vela del breakout mismo

        # Verificar ventana de tiempo
        if row['datetime'] > ventana_fin:
            break

        # Calcular excursión actual en dirección del break
        if breakout_dir == 'up':
            excursion_actual = row['high'] - breakout_price
        else:
            excursion_actual = breakout_price - row['low']

        # Actualizar máximo corriente
        if excursion_actual > mfe_corriendo:
            mfe_corriendo = excursion_actual

        # ── Condición 1: ¿ya se alejó suficiente? ────────────────────────────
        if not alejamiento_alcanzado:
            if mfe_corriendo >= alejamiento_min:
                alejamiento_alcanzado = True
                mfe_pre_retest_pts    = mfe_corriendo  # guardamos el MFE hasta este punto
            continue  # si aún no se alejó, no buscamos retest todavía

        # ── Una vez alejado, buscar Condición 2: ¿tocó el nivel? ─────────────
        # Para breakout alcista: el LOW de la vela toca zona ±tolerancia del breakout_price
        # Para breakout bajista: el HIGH de la vela toca zona ±tolerancia del breakout_price
        if breakout_dir == 'up':
            toco_nivel = row['low'] <= (breakout_price + tolerancia)
        else:
            toco_nivel = row['high'] >= (breakout_price - tolerancia)

        if toco_nivel:
            # ── Actualizar MFE_pre_retest al máximo real antes de esta vela ──
            # Recorrer desde el inicio hasta esta vela para el máximo real
            pre_velas = post_break.iloc[1:i]
            if len(pre_velas) > 0:
                if breakout_dir == 'up':
                    mfe_pre_retest_pts = pre_velas['high'].max() - breakout_price
                else:
                    mfe_pre_retest_pts = breakout_price - pre_velas['low'].min()
            else:
                mfe_pre_retest_pts = 0.0

            retest_encontrado = True
            retest_time       = row['datetime']
            retest_vela       = row
            break

    # ─────────────────────────────────────────────────────────────────────────
    # CONSTRUIR RESULTADO
    # ─────────────────────────────────────────────────────────────────────────

    # MFE pre-retest en %
    mfe_pre_retest_pct = mfe_pre_retest_pts / breakout_price * 100 if mfe_pre_retest_pts > 0 else 0.0

    result = {
        'date':               str(open_time.date()),
        'or_duration_min':    or_duration,
        'or_size_pct':        round(or_size_pct, 4),
        'or_close_position':  round(or_close_position, 4),
        'breakout_dir':       breakout_dir,
        'breakout_minute':    breakout_minute,
        'breakout_price':     round(breakout_price, 4),
        'mfe_total_pct':      round(mfe_total_pct, 4),

        # Variables del retest
        'alejamiento_ok':     alejamiento_alcanzado,
        'retest':             retest_encontrado,
        'mfe_pre_retest_pct': round(mfe_pre_retest_pct, 4),
    }

    if retest_encontrado and retest_vela is not None:
        mins_to_retest = (retest_time - breakout_time).seconds / 60

        # Profundidad del retest: cuánto penetró el nivel (normalizado por OR_size)
        if breakout_dir == 'up':
            penetracion = breakout_price - retest_vela['low']
        else:
            penetracion = retest_vela['high'] - breakout_price
        retest_depth_pct = penetracion / or_size * 100 if or_size > 0 else 0.0

        # Velocidad: cuánto % se alejó por minuto antes de volver
        retest_speed = mfe_pre_retest_pct / mins_to_retest if mins_to_retest > 0 else 0.0

        result.update({
            'mins_to_retest':    round(mins_to_retest, 1),
            'retest_speed':      round(retest_speed, 6),
            'retest_depth_pct':  round(retest_depth_pct, 4),
            'retest_time':       str(retest_time.time()),
        })
    else:
        result.update({
            'mins_to_retest':    np.nan,
            'retest_speed':      np.nan,
            'retest_depth_pct':  np.nan,
            'retest_time':       np.nan,
        })

    return result

# ─────────────────────────────────────────────────────────────────────────────
# RESÚMENES
# ─────────────────────────────────────────────────────────────────────────────

def resumen_por_or(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen estadístico por duración de OR."""
    rows = []
    for or_dur, grp in df.groupby('or_duration_min'):
        n_total        = len(grp)
        n_alejados     = grp['alejamiento_ok'].sum()
        n_retest       = grp['retest'].sum()
        pct_alejado    = n_alejados / n_total * 100
        pct_retest_del_total    = n_retest / n_total * 100
        pct_retest_del_alejado  = n_retest / n_alejados * 100 if n_alejados > 0 else 0

        sub_retest = grp[grp['retest'] == True]

        rows.append({
            'or_duration_min':           or_dur,
            'n_breakouts':               n_total,
            'n_alejados':                int(n_alejados),
            'n_retest':                  int(n_retest),
            'pct_alejado':               round(pct_alejado, 1),
            'pct_retest_del_total':      round(pct_retest_del_total, 1),
            'pct_retest_del_alejado':    round(pct_retest_del_alejado, 1),
            'mfe_pre_avg_pct':           round(sub_retest['mfe_pre_retest_pct'].mean(), 4),
            'mfe_pre_med_pct':           round(sub_retest['mfe_pre_retest_pct'].median(), 4),
            'mins_to_retest_avg':        round(sub_retest['mins_to_retest'].mean(), 1),
            'mins_to_retest_med':        round(sub_retest['mins_to_retest'].median(), 1),
            'retest_speed_avg':          round(sub_retest['retest_speed'].mean(), 6),
            'retest_depth_avg_pct':      round(sub_retest['retest_depth_pct'].mean(), 4),
        })
    return pd.DataFrame(rows)


def resumen_mfe_pre_bins(df: pd.DataFrame, or_dur: int = 15) -> pd.DataFrame:
    """
    Para OR de 15 min: tasa de retest por bin de MFE_pre_retest.
    Solo considera los días donde hubo alejamiento (condición 1 cumplida).
    """
    sub = df[(df['or_duration_min'] == or_dur) & (df['alejamiento_ok'] == True)].copy()

    bins   = [0, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00, 1.50, 2.00, 999]
    labels = ['0.00-0.20','0.20-0.30','0.30-0.40','0.40-0.50',
              '0.50-0.75','0.75-1.00','1.00-1.50','1.50-2.00','>2.00']

    sub['mfe_bin'] = pd.cut(sub['mfe_pre_retest_pct'], bins=bins, labels=labels, right=False)

    rows = []
    for bin_label, g in sub.groupby('mfe_bin', observed=True):
        rows.append({
            'mfe_pre_bin':       str(bin_label),
            'n':                 len(g),
            'n_retest':          int(g['retest'].sum()),
            'pct_retest':        round(g['retest'].mean() * 100, 1),
            'mins_retest_avg':   round(g.loc[g['retest']==True, 'mins_to_retest'].mean(), 1),
            'depth_avg_pct':     round(g.loc[g['retest']==True, 'retest_depth_pct'].mean(), 4),
        })
    return pd.DataFrame(rows)


def resumen_tiempo_bins(df: pd.DataFrame, or_dur: int = 15) -> pd.DataFrame:
    """Distribución del tiempo al retest para OR de 15 min."""
    sub = df[(df['or_duration_min'] == or_dur) & (df['retest'] == True)].copy()

    bins   = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120]
    labels = ['0-5','5-10','10-15','15-20','20-30','30-45','45-60','60-90','90-120']

    sub['tiempo_bin'] = pd.cut(sub['mins_to_retest'], bins=bins, labels=labels, right=False)

    rows = []
    for bin_label, g in sub.groupby('tiempo_bin', observed=True):
        rows.append({
            'mins_bin':          str(bin_label),
            'n':                 len(g),
            'pct_del_total':     round(len(g) / len(sub) * 100, 1),
            'mfe_pre_avg_pct':   round(g['mfe_pre_retest_pct'].mean(), 4),
            'depth_avg_pct':     round(g['retest_depth_pct'].mean(), 4),
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# PRINT CONSOLA
# ─────────────────────────────────────────────────────────────────────────────

def print_consola(summary_df, mfe_bins_df, tiempo_bins_df, raw_df):
    OR_MUESTRA = 15

    print("\n" + "="*95)
    print("NIVEL 3 — RESUMEN GENERAL por DURACIÓN DEL OR")
    print("="*95)
    print(f"{'OR(min)':>7} | {'N_break':>7} | {'%Alejado':>8} | {'%Retest/Total':>13} | "
          f"{'%Retest/Alejado':>15} | {'MFE_pre_avg%':>12} | {'MinRetest_avg':>13} | {'Depth_avg%':>10}")
    print("-"*95)
    for _, r in summary_df.iterrows():
        print(f"{int(r['or_duration_min']):>7} | "
              f"{int(r['n_breakouts']):>7} | "
              f"{r['pct_alejado']:>8.1f} | "
              f"{r['pct_retest_del_total']:>13.1f} | "
              f"{r['pct_retest_del_alejado']:>15.1f} | "
              f"{r['mfe_pre_avg_pct']:>12.4f} | "
              f"{r['mins_to_retest_avg']:>13.1f} | "
              f"{r['retest_depth_avg_pct']:>10.4f}")

    print("\n" + "="*70)
    print(f"NIVEL 3 — OR {OR_MUESTRA} min — TASA DE RETEST por BIN de MFE_PRE")
    print(f"(solo días donde el precio se alejó ≥{ALEJAMIENTO_PCT}%)")
    print("="*70)
    print(f"{'MFE_pre bin':>12} | {'N':>5} | {'N_retest':>8} | "
          f"{'%Retest':>7} | {'MinRetest':>9} | {'Depth%':>6}")
    print("-"*70)
    for _, r in mfe_bins_df.iterrows():
        min_rt = f"{r['mins_retest_avg']:.1f}" if not pd.isna(r['mins_retest_avg']) else "N/A"
        dep    = f"{r['depth_avg_pct']:.4f}"   if not pd.isna(r['depth_avg_pct'])   else "N/A"
        print(f"{r['mfe_pre_bin']:>12} | {int(r['n']):>5} | {int(r['n_retest']):>8} | "
              f"{r['pct_retest']:>7.1f} | {min_rt:>9} | {dep:>6}")

    print("\n" + "="*65)
    print(f"NIVEL 3 — OR {OR_MUESTRA} min — DISTRIBUCIÓN TIEMPO AL RETEST")
    print("="*65)
    print(f"{'Minutos':>10} | {'N':>5} | {'% del total':>10} | "
          f"{'MFE_pre_avg%':>12} | {'Depth_avg%':>10}")
    print("-"*65)
    for _, r in tiempo_bins_df.iterrows():
        print(f"{r['mins_bin']:>10} | {int(r['n']):>5} | {r['pct_del_total']:>10.1f} | "
              f"{r['mfe_pre_avg_pct']:>12.4f} | {r['depth_avg_pct']:>10.4f}")

    # Stats generales OR 15
    sub15 = raw_df[raw_df['or_duration_min'] == OR_MUESTRA]
    n_total     = len(sub15)
    n_alejados  = sub15['alejamiento_ok'].sum()
    n_retest    = sub15['retest'].sum()

    print(f"\n{'─'*65}")
    print(f"NIVEL 3 — OR {OR_MUESTRA} min — ESTADÍSTICAS GENERALES")
    print(f"{'─'*65}")
    print(f"  Total breakouts analizados:          {n_total:,}")
    print(f"  Se alejaron ≥{ALEJAMIENTO_PCT}%:              "
          f"{int(n_alejados):,} ({n_alejados/n_total*100:.1f}%)")
    print(f"  Hicieron retest (de los alejados):   "
          f"{int(n_retest):,} ({n_retest/n_alejados*100:.1f}%)")
    print(f"  NO hicieron retest:                  "
          f"{int(n_alejados-n_retest):,} ({(n_alejados-n_retest)/n_alejados*100:.1f}%)")

    sub_rt = sub15[sub15['retest'] == True]
    print(f"\n  --- Cuando SÍ hubo retest ---")
    print(f"  MFE_pre promedio:   {sub_rt['mfe_pre_retest_pct'].mean():.4f}%")
    print(f"  MFE_pre mediana:    {sub_rt['mfe_pre_retest_pct'].median():.4f}%")
    print(f"  Tiempo promedio:    {sub_rt['mins_to_retest'].mean():.1f} minutos")
    print(f"  Tiempo mediana:     {sub_rt['mins_to_retest'].median():.1f} minutos")
    print(f"  Profundidad prom:   {sub_rt['retest_depth_pct'].mean():.4f}% del OR_size")
    print(f"  Velocidad promedio: {sub_rt['retest_speed'].mean():.6f} %/minuto")

    print(f"\n{'─'*65}")
    print("ARCHIVOS GENERADOS en orb_results/:")
    print("  nivel3_raw.csv          → registro completo (día × OR_duration)")
    print("  nivel3_summary.csv      → resumen por OR_duration")
    print("  nivel3_mfe_pre_bins.csv → tasa de retest por bin de MFE_pre")
    print("  nivel3_tiempo_bins.csv  → distribución del tiempo al retest")
    print(f"{'─'*65}")
    print("\nCOLUMNAS CLAVE nivel3_raw.csv:")
    print("  alejamiento_ok      → True si el precio se alejó ≥0.20%")
    print("  retest              → True si hubo retest válido")
    print("  mfe_pre_retest_pct  → máximo alcanzado antes de volver (%)")
    print("  mins_to_retest      → minutos desde breakout hasta el retest")
    print("  retest_speed        → mfe_pre_retest_pct / mins_to_retest")
    print("  retest_depth_pct    → penetración del nivel / OR_size (%)")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Cargando datos desde: {DATA_FILE}")
    df = load_data(DATA_FILE)

    dates = df['date'].unique()
    print(f"Días encontrados:  {len(dates)}")
    print(f"OR durations:      {OR_DURATIONS}")
    total = len(dates) * len(OR_DURATIONS)
    print(f"Total combinaciones: {total:,}")
    print(f"\nCondiciones del retest:")
    print(f"  Alejamiento mínimo : {ALEJAMIENTO_PCT}% del precio de breakout")
    print(f"  Tolerancia del toque: ±{TOLERANCIA_PCT}% del precio de breakout")
    print(f"  Ventana de tiempo  : {VENTANA_MIN} minutos post-breakout\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []
    counter     = 0

    for date in dates:
        day_df = df[df['date'] == date]
        for or_dur in OR_DURATIONS:
            result = analyze_day(day_df, or_dur)
            if result:
                all_results.append(result)
            counter += 1
            if counter % 1000 == 0:
                pct = counter / total * 100
                print(f"  {counter:,}/{total:,} ({pct:.1f}%) — "
                      f"registros: {len(all_results):,}")

    print(f"\nTotal registros: {len(all_results):,}")

    raw_df = pd.DataFrame(all_results)

    # Guardar raw
    raw_path = os.path.join(OUTPUT_DIR, "nivel3_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"Raw guardado: {raw_path}  ({len(raw_df):,} filas)")

    # Resúmenes
    summary_df    = resumen_por_or(raw_df)
    mfe_bins_df   = resumen_mfe_pre_bins(raw_df, or_dur=15)
    tiempo_bins_df = resumen_tiempo_bins(raw_df, or_dur=15)

    summary_df.to_csv(   os.path.join(OUTPUT_DIR, "nivel3_summary.csv"),      index=False)
    mfe_bins_df.to_csv(  os.path.join(OUTPUT_DIR, "nivel3_mfe_pre_bins.csv"), index=False)
    tiempo_bins_df.to_csv(os.path.join(OUTPUT_DIR,"nivel3_tiempo_bins.csv"),  index=False)

    print_consola(summary_df, mfe_bins_df, tiempo_bins_df, raw_df)
    print("\nDone ✓")


if __name__ == "__main__":
    main()