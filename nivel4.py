"""
NIVEL 4 — El Outcome Final
===========================
Para cada (día × OR_duration) que tuvo retest válido, determina el outcome:

  OUTCOME A — CONTINUACIÓN:
    El precio supera el MFE_pre_retest dentro de 60 min post-retest.

  OUTCOME B — FALLA:
    El precio cruza el lado opuesto del OR dentro de 60 min post-retest,
    antes de superar el MFE_pre_retest.

  OUTCOME C — NEUTRO:
    Ninguno de los dos anteriores en 60 minutos.

Variables medidas:
  Para CONTINUACIÓN:
    mfe_post_pct        → máximo alcanzado post-retest desde breakout_price (%)
    continuation_ratio  → mfe_post_pct / mfe_pre_retest_pct
    mins_to_continuation → minutos desde retest hasta superar MFE_pre

  Para FALLA:
    mae_falla_pct       → movimiento en contra desde breakout_price (%)
    mins_to_falla       → minutos desde retest hasta cruzar OR opuesto

Outputs:
  nivel4_raw.csv              → registro completo (día × OR_duration)
  nivel4_summary.csv          → resumen de outcomes por OR_duration
  nivel4_predictores.csv      → tasas de continuación por variable predictora
  nivel4_continuation_ratio.csv → distribución del continuation_ratio
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

# Condiciones del retest (mismas que Nivel 3)
ALEJAMIENTO_PCT = 0.20
TOLERANCIA_PCT  = 0.05
VENTANA_MIN     = 120

# Ventana para determinar el outcome post-retest
OUTCOME_VENTANA_MIN = 60

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
# ANÁLISIS PRINCIPAL
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

    # Volumen promedio OR
    or_vol_avg = or_bars['volume'].mean()
    if or_vol_avg == 0:
        or_vol_avg = 1

    # ── Velas post-OR ─────────────────────────────────────────────────────────
    post_or = day_df[day_df['datetime'] >= or_end].reset_index(drop=True)
    if len(post_or) < 2:
        return None

    # ── Detectar primer breakout ──────────────────────────────────────────────
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

    breakout_minute  = int((breakout_time - open_time).seconds / 60)

    # Variables Nivel 2
    if breakout_dir == 'up':
        raw_strength = breakout_bar['close'] - or_high
    else:
        raw_strength = or_low - breakout_bar['close']
    breakout_strength = raw_strength / or_size
    vol_ratio         = breakout_bar['volume'] / or_vol_avg

    # ── Velas desde el breakout ───────────────────────────────────────────────
    post_break = post_or.iloc[breakout_idx:].reset_index(drop=True)
    if len(post_break) < 2:
        return None

    # ── Umbrales del retest ───────────────────────────────────────────────────
    alejamiento_min = breakout_price * (ALEJAMIENTO_PCT / 100)
    tolerancia      = breakout_price * (TOLERANCIA_PCT  / 100)
    ventana_fin     = breakout_time + pd.Timedelta(minutes=VENTANA_MIN)

    # ── Buscar retest (misma lógica que Nivel 3) ──────────────────────────────
    alejamiento_alcanzado = False
    mfe_pre_pts           = 0.0
    mfe_corriendo         = 0.0
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
            # MFE_pre = máximo real antes de esta vela
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

    # ── Si no hubo retest válido, no hay Nivel 4 ─────────────────────────────
    if not retest_encontrado or retest_vela is None:
        return None

    mfe_pre_retest_pct = mfe_pre_pts / breakout_price * 100

    # Profundidad y velocidad del retest
    mins_to_retest = (retest_time - breakout_time).seconds / 60
    if breakout_dir == 'up':
        penetracion = breakout_price - retest_vela['low']
    else:
        penetracion = retest_vela['high'] - breakout_price
    retest_depth_pct = penetracion / or_size * 100 if or_size > 0 else 0.0
    retest_speed     = mfe_pre_retest_pct / mins_to_retest if mins_to_retest > 0 else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # NIVEL 4 — DETERMINAR OUTCOME POST-RETEST
    # ─────────────────────────────────────────────────────────────────────────

    # Velas después del retest hasta fin de ventana outcome
    outcome_fin = retest_time + pd.Timedelta(minutes=OUTCOME_VENTANA_MIN)
    post_retest = post_break.iloc[retest_idx:].reset_index(drop=True)

    # Nivel que hay que superar para confirmar continuación
    if breakout_dir == 'up':
        nivel_continuacion = breakout_price + mfe_pre_pts   # superar MFE_pre
        nivel_falla        = or_low                          # cruzar al otro lado del OR
    else:
        nivel_continuacion = breakout_price - mfe_pre_pts   # superar MFE_pre (bajando)
        nivel_falla        = or_high                         # cruzar al otro lado del OR

    outcome             = 'neutro'
    mfe_post_pct        = np.nan
    continuation_ratio  = np.nan
    mins_to_continuation = np.nan
    mae_falla_pct       = np.nan
    mins_to_falla       = np.nan

    for i, row in post_retest.iterrows():
        if i == 0:
            continue  # saltamos la vela del retest mismo

        if row['datetime'] > outcome_fin:
            break

        mins_desde_retest = (row['datetime'] - retest_time).seconds / 60

        # ── ¿Continuación? ────────────────────────────────────────────────────
        if breakout_dir == 'up':
            supero_mfe = row['high'] >= nivel_continuacion
        else:
            supero_mfe = row['low'] <= nivel_continuacion

        # ── ¿Falla? ───────────────────────────────────────────────────────────
        if breakout_dir == 'up':
            cruzo_or = row['low'] <= nivel_falla
        else:
            cruzo_or = row['high'] >= nivel_falla

        # Evaluar en orden: primero continuación, luego falla
        if supero_mfe:
            outcome = 'continuacion'
            mins_to_continuation = mins_desde_retest

            # MFE post: máximo alcanzado desde el breakout_price post-retest
            post_cont = post_retest.iloc[:i+1]
            if breakout_dir == 'up':
                mfe_post_price = post_cont['high'].max()
                mfe_post_pts   = mfe_post_price - breakout_price
            else:
                mfe_post_price = post_cont['low'].min()
                mfe_post_pts   = breakout_price - mfe_post_price

            mfe_post_pct       = mfe_post_pts / breakout_price * 100
            continuation_ratio = mfe_post_pct / mfe_pre_retest_pct if mfe_pre_retest_pct > 0 else np.nan
            break

        elif cruzo_or:
            outcome = 'falla'
            mins_to_falla = mins_desde_retest

            # MAE: cuánto se movió en contra desde el breakout_price
            post_falla = post_retest.iloc[:i+1]
            if breakout_dir == 'up':
                mae_price    = post_falla['low'].min()
                mae_falla_pts = breakout_price - mae_price
            else:
                mae_price    = post_falla['high'].max()
                mae_falla_pts = mae_price - breakout_price

            mae_falla_pct = mae_falla_pts / breakout_price * 100
            break

    return {
        # Identificación
        'date':               str(open_time.date()),
        'or_duration_min':    or_duration,

        # Nivel 1 — contexto del rango
        'or_size_pct':        round(or_size_pct, 4),
        'or_close_position':  round(or_close_position, 4),

        # Nivel 2 — breakout
        'breakout_dir':       breakout_dir,
        'breakout_minute':    breakout_minute,
        'breakout_strength':  round(breakout_strength, 4),
        'vol_ratio':          round(vol_ratio, 4),

        # Nivel 3 — retest
        'mfe_pre_retest_pct': round(mfe_pre_retest_pct, 4),
        'mins_to_retest':     round(mins_to_retest, 1),
        'retest_speed':       round(retest_speed, 6),
        'retest_depth_pct':   round(retest_depth_pct, 4),

        # Nivel 4 — outcome
        'outcome':            outcome,

        # Si continuación
        'mfe_post_pct':           round(mfe_post_pct, 4)        if not np.isnan(mfe_post_pct)        else np.nan,
        'continuation_ratio':     round(continuation_ratio, 4)  if not np.isnan(continuation_ratio)  else np.nan,
        'mins_to_continuation':   round(mins_to_continuation, 1) if not np.isnan(mins_to_continuation) else np.nan,

        # Si falla
        'mae_falla_pct':          round(mae_falla_pct, 4)       if not np.isnan(mae_falla_pct)       else np.nan,
        'mins_to_falla':          round(mins_to_falla, 1)       if not np.isnan(mins_to_falla)       else np.nan,
    }

# ─────────────────────────────────────────────────────────────────────────────
# RESÚMENES
# ─────────────────────────────────────────────────────────────────────────────

def resumen_por_or(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for or_dur, grp in df.groupby('or_duration_min'):
        n = len(grp)
        n_cont   = (grp['outcome'] == 'continuacion').sum()
        n_falla  = (grp['outcome'] == 'falla').sum()
        n_neutro = (grp['outcome'] == 'neutro').sum()

        cont_grp  = grp[grp['outcome'] == 'continuacion']
        falla_grp = grp[grp['outcome'] == 'falla']

        rows.append({
            'or_duration_min':        or_dur,
            'n_retests':              n,
            'n_continuacion':         int(n_cont),
            'n_falla':                int(n_falla),
            'n_neutro':               int(n_neutro),
            'pct_continuacion':       round(n_cont  / n * 100, 1),
            'pct_falla':              round(n_falla / n * 100, 1),
            'pct_neutro':             round(n_neutro / n * 100, 1),

            # Continuación
            'mfe_post_avg_pct':       round(cont_grp['mfe_post_pct'].mean(), 4),
            'mfe_post_med_pct':       round(cont_grp['mfe_post_pct'].median(), 4),
            'cont_ratio_avg':         round(cont_grp['continuation_ratio'].mean(), 4),
            'cont_ratio_med':         round(cont_grp['continuation_ratio'].median(), 4),
            'mins_to_cont_avg':       round(cont_grp['mins_to_continuation'].mean(), 1),

            # Falla
            'mae_falla_avg_pct':      round(falla_grp['mae_falla_pct'].mean(), 4),
            'mins_to_falla_avg':      round(falla_grp['mins_to_falla'].mean(), 1),
        })
    return pd.DataFrame(rows)


def resumen_predictores(df: pd.DataFrame, or_dur: int = 15) -> pd.DataFrame:
    """
    Para OR de 15 min: tasa de continuación segmentada por cada variable predictora.
    Muestra el poder predictivo de cada variable de los Niveles 1-3.
    """
    sub = df[df['or_duration_min'] == or_dur].copy()
    rows = []

    def agregar(nombre_var, serie, etiquetas):
        sub['_bin'] = serie
        for etiqueta, g in sub.groupby('_bin', observed=True):
            if len(g) < 10:
                continue
            n_cont = (g['outcome'] == 'continuacion').sum()
            n_fall = (g['outcome'] == 'falla').sum()
            rows.append({
                'variable':        nombre_var,
                'segmento':        str(etiqueta),
                'n':               len(g),
                'pct_continuacion': round(n_cont / len(g) * 100, 1),
                'pct_falla':        round(n_fall / len(g) * 100, 1),
                'pct_neutro':       round((len(g) - n_cont - n_fall) / len(g) * 100, 1),
                'mfe_post_avg':    round(g.loc[g['outcome']=='continuacion','mfe_post_pct'].mean(), 4),
                'cont_ratio_avg':  round(g.loc[g['outcome']=='continuacion','continuation_ratio'].mean(), 4),
            })

    # OR_size_pct por quintiles
    try:
        sub['_q'] = pd.qcut(sub['or_size_pct'], q=5,
                             labels=['Q1_mini','Q2_peq','Q3_med','Q4_gde','Q5_eno'],
                             duplicates='drop')
        agregar('or_size_pct (quintiles)', sub['_q'], None)
    except Exception:
        pass

    # OR_close_position en tercios
    sub['_cp'] = pd.cut(sub['or_close_position'],
                        bins=[0, 0.33, 0.67, 1.01],
                        labels=['bajo(<0.33)','medio(0.33-0.67)','alto(>0.67)'])
    agregar('or_close_position', sub['_cp'], None)

    # breakout_minute en franjas
    sub['_bm'] = pd.cut(sub['breakout_minute'],
                        bins=[0, 15, 30, 60, 90, 999],
                        labels=['0-15min','15-30min','30-60min','60-90min','>90min'])
    agregar('breakout_minute', sub['_bm'], None)

    # vol_ratio por quintiles
    try:
        sub['_vr'] = pd.qcut(sub['vol_ratio'], q=5,
                              labels=['Q1','Q2','Q3','Q4','Q5'],
                              duplicates='drop')
        agregar('vol_ratio (quintiles)', sub['_vr'], None)
    except Exception:
        pass

    # breakout_strength por quintiles
    try:
        sub['_bs'] = pd.qcut(sub['breakout_strength'], q=5,
                              labels=['Q1','Q2','Q3','Q4','Q5'],
                              duplicates='drop')
        agregar('breakout_strength (quintiles)', sub['_bs'], None)
    except Exception:
        pass

    # mfe_pre_retest_pct en bins
    sub['_mp'] = pd.cut(sub['mfe_pre_retest_pct'],
                        bins=[0, 0.20, 0.30, 0.40, 0.50, 0.75, 999],
                        labels=['0.20-0.30','0.30-0.40','0.40-0.50','0.50-0.75','0.75-1.00','>1.00'])
    agregar('mfe_pre_retest_pct', sub['_mp'], None)

    # mins_to_retest en bins
    sub['_mt'] = pd.cut(sub['mins_to_retest'],
                        bins=[0, 10, 20, 30, 45, 60, 999],
                        labels=['0-10min','10-20min','20-30min','30-45min','45-60min','>60min'])
    agregar('mins_to_retest', sub['_mt'], None)

    # retest_depth_pct en tercios
    try:
        sub['_rd'] = pd.qcut(sub['retest_depth_pct'], q=3,
                              labels=['superficial','medio','profundo'],
                              duplicates='drop')
        agregar('retest_depth_pct', sub['_rd'], None)
    except Exception:
        pass

    # Día de la semana
    sub['_dow'] = pd.to_datetime(sub['date']).dt.day_name()
    agregar('dia_semana', sub['_dow'], None)

    return pd.DataFrame(rows)


def resumen_continuation_ratio(df: pd.DataFrame, or_dur: int = 15) -> pd.DataFrame:
    """Distribución del continuation_ratio para OR de 15 min."""
    sub = df[(df['or_duration_min'] == or_dur) & (df['outcome'] == 'continuacion')].copy()

    bins   = [0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 999]
    labels = ['<0.5','0.5-0.75','0.75-1.0','1.0-1.25','1.25-1.5','1.5-2.0','2.0-3.0','>3.0']

    sub['ratio_bin'] = pd.cut(sub['continuation_ratio'], bins=bins, labels=labels, right=False)

    rows = []
    for bin_label, g in sub.groupby('ratio_bin', observed=True):
        rows.append({
            'ratio_bin':       str(bin_label),
            'n':               len(g),
            'pct_del_total':   round(len(g) / len(sub) * 100, 1),
            'mfe_post_avg':    round(g['mfe_post_pct'].mean(), 4),
            'mfe_pre_avg':     round(g['mfe_pre_retest_pct'].mean(), 4),
            'mins_cont_avg':   round(g['mins_to_continuation'].mean(), 1),
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# PRINT CONSOLA
# ─────────────────────────────────────────────────────────────────────────────

def print_consola(summary_df, predictores_df, ratio_df, raw_df):
    OR_MUESTRA = 15

    print("\n" + "="*100)
    print("NIVEL 4 — RESUMEN DE OUTCOMES por DURACIÓN DEL OR")
    print("="*100)
    print(f"{'OR':>5} | {'N':>5} | {'%Cont':>6} | {'%Falla':>7} | {'%Neutro':>7} | "
          f"{'MFE_post_avg%':>13} | {'ContRatio_avg':>13} | {'MinsCont':>8} | "
          f"{'MAE_falla%':>10} | {'MinsFalla':>9}")
    print("-"*100)
    for _, r in summary_df.iterrows():
        print(f"{int(r['or_duration_min']):>5} | "
              f"{int(r['n_retests']):>5} | "
              f"{r['pct_continuacion']:>6.1f} | "
              f"{r['pct_falla']:>7.1f} | "
              f"{r['pct_neutro']:>7.1f} | "
              f"{r['mfe_post_avg_pct']:>13.4f} | "
              f"{r['cont_ratio_avg']:>13.4f} | "
              f"{r['mins_to_cont_avg']:>8.1f} | "
              f"{r['mae_falla_avg_pct']:>10.4f} | "
              f"{r['mins_to_falla_avg']:>9.1f}")

    print("\n" + "="*80)
    print(f"NIVEL 4 — OR {OR_MUESTRA} min — PODER PREDICTIVO DE CADA VARIABLE")
    print("="*80)

    variables = predictores_df['variable'].unique()
    for var in variables:
        sub = predictores_df[predictores_df['variable'] == var]
        print(f"\n  [{var}]")
        print(f"  {'Segmento':>22} | {'N':>5} | {'%Cont':>6} | {'%Falla':>7} | "
              f"{'%Neutro':>7} | {'MFE_post_avg':>12} | {'ContRatio':>9}")
        print(f"  {'-'*78}")
        for _, r in sub.iterrows():
            mfe = f"{r['mfe_post_avg']:.4f}" if not pd.isna(r['mfe_post_avg']) else "  N/A "
            cr  = f"{r['cont_ratio_avg']:.4f}" if not pd.isna(r['cont_ratio_avg']) else "  N/A "
            print(f"  {r['segmento']:>22} | {int(r['n']):>5} | "
                  f"{r['pct_continuacion']:>6.1f} | "
                  f"{r['pct_falla']:>7.1f} | "
                  f"{r['pct_neutro']:>7.1f} | "
                  f"{mfe:>12} | {cr:>9}")

    print("\n" + "="*70)
    print(f"NIVEL 4 — OR {OR_MUESTRA} min — DISTRIBUCIÓN DEL CONTINUATION RATIO")
    print("(ratio > 1.0 significa que el post-retest fue mayor que el pre-retest)")
    print("="*70)
    print(f"{'Ratio bin':>10} | {'N':>5} | {'% total':>7} | "
          f"{'MFE_post%':>9} | {'MFE_pre%':>8} | {'MinsCont':>8}")
    print("-"*70)
    for _, r in ratio_df.iterrows():
        print(f"{r['ratio_bin']:>10} | {int(r['n']):>5} | "
              f"{r['pct_del_total']:>7.1f} | "
              f"{r['mfe_post_avg']:>9.4f} | "
              f"{r['mfe_pre_avg']:>8.4f} | "
              f"{r['mins_cont_avg']:>8.1f}")

    # Estadísticas generales OR 15
    sub15 = raw_df[raw_df['or_duration_min'] == OR_MUESTRA]
    n     = len(sub15)
    n_c   = (sub15['outcome'] == 'continuacion').sum()
    n_f   = (sub15['outcome'] == 'falla').sum()
    n_n   = (sub15['outcome'] == 'neutro').sum()

    cont_grp  = sub15[sub15['outcome'] == 'continuacion']
    falla_grp = sub15[sub15['outcome'] == 'falla']

    print(f"\n{'─'*70}")
    print(f"NIVEL 4 — OR {OR_MUESTRA} min — ESTADÍSTICAS GENERALES")
    print(f"{'─'*70}")
    print(f"  Total retests analizados: {n:,}")
    print(f"  CONTINUACIÓN:  {int(n_c):,} ({n_c/n*100:.1f}%)")
    print(f"  FALLA:         {int(n_f):,} ({n_f/n*100:.1f}%)")
    print(f"  NEUTRO:        {int(n_n):,} ({n_n/n*100:.1f}%)")

    print(f"\n  --- Cuando hubo CONTINUACIÓN ---")
    print(f"  MFE_post promedio:          {cont_grp['mfe_post_pct'].mean():.4f}%")
    print(f"  MFE_post mediana:           {cont_grp['mfe_post_pct'].median():.4f}%")
    print(f"  Continuation ratio prom:    {cont_grp['continuation_ratio'].mean():.4f}x")
    print(f"  Continuation ratio mediana: {cont_grp['continuation_ratio'].median():.4f}x")
    print(f"  Minutos hasta confirmar:    {cont_grp['mins_to_continuation'].mean():.1f} min promedio")
    print(f"  Ratio > 1.0 (post > pre):   "
          f"{(cont_grp['continuation_ratio'] > 1.0).mean()*100:.1f}% de las continuaciones")

    print(f"\n  --- Cuando hubo FALLA ---")
    print(f"  MAE falla promedio:         {falla_grp['mae_falla_pct'].mean():.4f}%")
    print(f"  MAE falla mediana:          {falla_grp['mae_falla_pct'].median():.4f}%")
    print(f"  Minutos hasta falla:        {falla_grp['mins_to_falla'].mean():.1f} min promedio")

    print(f"\n{'─'*70}")
    print("ARCHIVOS GENERADOS en orb_results/:")
    print("  nivel4_raw.csv                → registro completo")
    print("  nivel4_summary.csv            → resumen por OR_duration")
    print("  nivel4_predictores.csv        → poder predictivo de cada variable")
    print("  nivel4_continuation_ratio.csv → distribución del continuation_ratio")
    print(f"{'─'*70}")
    print("\nCOLUMNAS CLAVE nivel4_raw.csv:")
    print("  outcome              → 'continuacion', 'falla', 'neutro'")
    print("  mfe_post_pct         → máximo post-retest desde breakout_price (%)")
    print("  continuation_ratio   → mfe_post_pct / mfe_pre_retest_pct")
    print("  mins_to_continuation → minutos hasta superar MFE_pre")
    print("  mae_falla_pct        → movimiento en contra si hubo falla (%)")
    print("  mins_to_falla        → minutos hasta cruzar OR opuesto si falla")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Cargando datos desde: {DATA_FILE}")
    df = load_data(DATA_FILE)

    dates = df['date'].unique()
    print(f"Días encontrados:    {len(dates)}")
    print(f"OR durations:        {OR_DURATIONS}")
    total = len(dates) * len(OR_DURATIONS)
    print(f"Total combinaciones: {total:,}")
    print(f"\nDefinición de outcomes:")
    print(f"  CONTINUACIÓN: supera MFE_pre dentro de {OUTCOME_VENTANA_MIN} min post-retest")
    print(f"  FALLA:        cruza OR opuesto antes de superar MFE_pre")
    print(f"  NEUTRO:       ninguno de los dos en {OUTCOME_VENTANA_MIN} min\n")

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
                      f"retests: {len(all_results):,}")

    print(f"\nTotal retests con outcome: {len(all_results):,}")

    raw_df = pd.DataFrame(all_results)

    # Guardar raw
    raw_path = os.path.join(OUTPUT_DIR, "nivel4_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"Raw guardado: {raw_path}  ({len(raw_df):,} filas)")

    # Resúmenes
    summary_df    = resumen_por_or(raw_df)
    predictores_df = resumen_predictores(raw_df, or_dur=15)
    ratio_df      = resumen_continuation_ratio(raw_df, or_dur=15)

    summary_df.to_csv(    os.path.join(OUTPUT_DIR, "nivel4_summary.csv"),            index=False)
    predictores_df.to_csv(os.path.join(OUTPUT_DIR, "nivel4_predictores.csv"),        index=False)
    ratio_df.to_csv(      os.path.join(OUTPUT_DIR, "nivel4_continuation_ratio.csv"), index=False)

    print_consola(summary_df, predictores_df, ratio_df, raw_df)
    print("\nDone ✓")


if __name__ == "__main__":
    main()