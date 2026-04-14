"""
NIVEL 2 — Anatomía del Breakout
================================
Variables que se miden por cada día y duración de OR:

  2.1  breakout_minute    → minutos desde apertura hasta el breakout
  2.2  breakout_strength  → cuánto penetró la vela de break el nivel (/ OR_size)
  2.3  vol_ratio          → volumen vela breakout / volumen promedio velas OR
  2.4  breakout_con_gap   → ¿el breakout va en la misma dirección que el gap?

Outputs:
  nivel2_raw.csv              → un registro por (día × OR_duration)
  nivel2_hora_mfe.csv         → MFE promedio por franja horaria y OR duration
  nivel2_volratio_mfe.csv     → MFE promedio por quintil de vol_ratio
  nivel2_strength_mfe.csv     → MFE promedio por quintil de breakout_strength
  nivel2_gap_dir.csv          → efecto de alineación gap × dirección breakout
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS  — ajusta solo estas líneas si es necesario
# ─────────────────────────────────────────────────────────────────────────────

DATA_FILE   = "QQQ_1min.csv"       # archivo original de 1 minuto
OUTPUT_DIR  = "orb_results"        # misma carpeta que el Nivel 1
OR_DURATIONS = list(range(5, 125, 5))  # 5, 10, 15 … 120 minutos

# Franjas horarias para el análisis de hora del breakout (en minutos desde apertura)
# 0=9:30, 30=10:00, 60=10:30, 90=11:00, 120=11:30, 150=12:00,
# 180=12:30, 210=13:00, 240=13:30, 270=14:00, 300=14:30, 330=15:00
FRANJAS = [
    (0,   30,  "09:30-10:00"),
    (30,  60,  "10:00-10:30"),
    (60,  90,  "10:30-11:00"),
    (90,  120, "11:00-11:30"),
    (120, 150, "11:30-12:00"),
    (150, 180, "12:00-12:30"),
    (180, 210, "12:30-13:00"),
    (210, 240, "13:00-13:30"),
    (240, 270, "13:30-14:00"),
    (270, 300, "14:00-14:30"),
    (300, 330, "14:30-15:00"),
    (330, 390, "15:00-15:30"),
]

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

    # ── Velas del Opening Range ───────────────────────────────────────────────
    open_time = day_df['datetime'].iloc[0]
    or_end    = open_time + pd.Timedelta(minutes=or_duration)
    or_bars   = day_df[day_df['datetime'] < or_end]

    if len(or_bars) < max(1, or_duration - 2):
        return None

    or_high     = or_bars['high'].max()
    or_low      = or_bars['low'].min()
    or_size     = or_high - or_low
    or_size_pct = or_size / or_low * 100

    if or_size == 0:
        return None

    # Posición del cierre dentro del OR
    or_close          = or_bars['close'].iloc[-1]
    or_close_position = (or_close - or_low) / or_size

    # Volumen promedio dentro del OR (para vol_ratio)
    or_vol_avg = or_bars['volume'].mean()
    if or_vol_avg == 0:
        or_vol_avg = 1

    # ── Gap overnight ─────────────────────────────────────────────────────────
    # (open del día vs close del día anterior — aproximado con open primera vela)
    day_open = day_df['open'].iloc[0]

    # ── Velas posteriores al OR ───────────────────────────────────────────────
    post_or = day_df[day_df['datetime'] >= or_end].reset_index(drop=True)

    if len(post_or) < 2:
        return None

    # ── Detectar primer breakout ──────────────────────────────────────────────
    breakout_idx    = None
    breakout_dir    = None
    breakout_price  = None
    breakout_bar    = None

    for i, row in post_or.iterrows():
        if row['high'] > or_high:
            breakout_idx   = i
            breakout_dir   = 'up'
            breakout_price = or_high
            breakout_bar   = row
            break
        elif row['low'] < or_low:
            breakout_idx   = i
            breakout_dir   = 'down'
            breakout_price = or_low
            breakout_bar   = row
            break

    if breakout_idx is None:
        return None

    breakout_time   = breakout_bar['datetime']
    breakout_minute = int((breakout_time - open_time).seconds / 60)

    # ── Variable 2.2 — Fuerza del breakout ───────────────────────────────────
    # Qué tan lejos cerró la vela de breakout más allá del nivel, normalizado
    if breakout_dir == 'up':
        raw_strength = breakout_bar['close'] - or_high
    else:
        raw_strength = or_low - breakout_bar['close']

    breakout_strength = raw_strength / or_size  # puede ser negativo si cerró dentro del OR

    # ── Variable 2.3 — Vol ratio ──────────────────────────────────────────────
    vol_ratio = breakout_bar['volume'] / or_vol_avg

    # ── Variable 2.4 — Alineación con gap ────────────────────────────────────
    # Usamos la posición del close del OR como proxy del gap acumulado
    # gap "alcista" si or_close_position > 0.5 (precio subió en el OR)
    # gap "bajista" si or_close_position < 0.5
    if or_close_position > 0.55:
        gap_tipo = 'alcista'
    elif or_close_position < 0.45:
        gap_tipo = 'bajista'
    else:
        gap_tipo = 'plano'

    if breakout_dir == 'up' and gap_tipo == 'alcista':
        breakout_con_gap = 1   # breakout alineado con presión interna
    elif breakout_dir == 'down' and gap_tipo == 'bajista':
        breakout_con_gap = 1
    else:
        breakout_con_gap = 0   # breakout contra la presión interna

    # ── MFE total desde el breakout hasta cierre ──────────────────────────────
    post_break = post_or.iloc[breakout_idx:].reset_index(drop=True)

    if breakout_dir == 'up':
        mfe_price = post_break['high'].max()
        mfe_pts   = mfe_price - breakout_price
    else:
        mfe_price = post_break['low'].min()
        mfe_pts   = breakout_price - mfe_price

    mfe_pct = mfe_pts / breakout_price * 100

    # ── Franja horaria del breakout ───────────────────────────────────────────
    franja = "fuera_rango"
    for (start, end, label) in FRANJAS:
        if start <= breakout_minute < end:
            franja = label
            break

    # ── Día de la semana ──────────────────────────────────────────────────────
    dia_semana = breakout_time.day_name()

    return {
        'date':               str(open_time.date()),
        'dia_semana':         dia_semana,
        'or_duration_min':    or_duration,

        # Nivel 1 (contexto)
        'or_size_pct':        round(or_size_pct, 4),
        'or_close_position':  round(or_close_position, 4),
        'gap_tipo':           gap_tipo,

        # Nivel 2 — variables del breakout
        'breakout_dir':       breakout_dir,
        'breakout_minute':    breakout_minute,
        'breakout_time':      str(breakout_time.time()),
        'franja_horaria':     franja,
        'breakout_strength':  round(breakout_strength, 4),
        'vol_ratio':          round(vol_ratio, 4),
        'breakout_con_gap':   breakout_con_gap,

        # MFE total
        'mfe_pct':            round(mfe_pct, 4),
        'mfe_pts':            round(mfe_pts, 4),
    }

# ─────────────────────────────────────────────────────────────────────────────
# RESÚMENES
# ─────────────────────────────────────────────────────────────────────────────

def resumen_por_franja(df: pd.DataFrame) -> pd.DataFrame:
    """MFE promedio por franja horaria, separado por OR duration."""
    rows = []
    for or_dur in sorted(df['or_duration_min'].unique()):
        sub = df[df['or_duration_min'] == or_dur]
        for franja in [f[2] for f in FRANJAS]:
            g = sub[sub['franja_horaria'] == franja]
            if len(g) < 10:
                continue
            rows.append({
                'or_duration_min': or_dur,
                'franja_horaria':  franja,
                'n':               len(g),
                'pct_up':          round((g['breakout_dir'] == 'up').mean() * 100, 1),
                'mfe_avg_pct':     round(g['mfe_pct'].mean(), 4),
                'mfe_med_pct':     round(g['mfe_pct'].median(), 4),
                'vol_ratio_avg':   round(g['vol_ratio'].mean(), 2),
                'strength_avg':    round(g['breakout_strength'].mean(), 4),
            })
    return pd.DataFrame(rows)


def resumen_por_quintil(df: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    """MFE promedio por quintil de una variable continua."""
    rows = []
    for or_dur in sorted(df['or_duration_min'].unique()):
        sub = df[df['or_duration_min'] == or_dur].copy()
        if len(sub) < 50:
            continue
        try:
            sub['quintil'] = pd.qcut(sub[col], q=5,
                                     labels=['Q1','Q2','Q3','Q4','Q5'],
                                     duplicates='drop')
        except Exception:
            continue
        for q, g in sub.groupby('quintil', observed=True):
            rows.append({
                'or_duration_min': or_dur,
                'variable':        label,
                'quintil':         str(q),
                f'{col}_avg':      round(g[col].mean(), 4),
                'n':               len(g),
                'mfe_avg_pct':     round(g['mfe_pct'].mean(), 4),
                'mfe_med_pct':     round(g['mfe_pct'].median(), 4),
                'pct_up':          round((g['breakout_dir'] == 'up').mean() * 100, 1),
            })
    return pd.DataFrame(rows)


def resumen_gap_dir(df: pd.DataFrame) -> pd.DataFrame:
    """Efecto de alineación breakout con gap."""
    rows = []
    for or_dur in sorted(df['or_duration_min'].unique()):
        sub = df[df['or_duration_min'] == or_dur]
        for alineado, g in sub.groupby('breakout_con_gap'):
            label = 'alineado_con_gap' if alineado == 1 else 'contra_gap'
            rows.append({
                'or_duration_min':  or_dur,
                'alineacion':       label,
                'n':                len(g),
                'mfe_avg_pct':      round(g['mfe_pct'].mean(), 4),
                'mfe_med_pct':      round(g['mfe_pct'].median(), 4),
                'pct_up':           round((g['breakout_dir'] == 'up').mean() * 100, 1),
            })
    return pd.DataFrame(rows)


def resumen_dia_semana(df: pd.DataFrame) -> pd.DataFrame:
    """MFE promedio por día de la semana."""
    orden = ['Monday','Tuesday','Wednesday','Thursday','Friday']
    rows = []
    for or_dur in sorted(df['or_duration_min'].unique()):
        sub = df[df['or_duration_min'] == or_dur]
        for dia in orden:
            g = sub[sub['dia_semana'] == dia]
            if len(g) < 10:
                continue
            rows.append({
                'or_duration_min': or_dur,
                'dia_semana':      dia,
                'n':               len(g),
                'mfe_avg_pct':     round(g['mfe_pct'].mean(), 4),
                'mfe_med_pct':     round(g['mfe_pct'].median(), 4),
                'vol_ratio_avg':   round(g['vol_ratio'].mean(), 2),
                'pct_up':          round((g['breakout_dir'] == 'up').mean() * 100, 1),
            })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# PRINT CONSOLA — OR de 15 min como muestra representativa
# ─────────────────────────────────────────────────────────────────────────────

def print_resumen_consola(raw_df, franja_df, quintil_vol, quintil_str, gap_df, dia_df):
    OR_MUESTRA = 15

    print("\n" + "="*85)
    print(f"NIVEL 2 — OR {OR_MUESTRA} min — MFE por FRANJA HORARIA del breakout")
    print("="*85)
    sub = franja_df[franja_df['or_duration_min'] == OR_MUESTRA]
    print(f"{'Franja':>17} | {'N':>5} | {'%Up':>5} | {'MFE_avg%':>8} | {'MFE_med%':>8} | {'VolRatio':>8} | {'Strength':>8}")
    print("-"*85)
    for _, r in sub.iterrows():
        print(f"{r['franja_horaria']:>17} | {int(r['n']):>5} | {r['pct_up']:>5.1f} | "
              f"{r['mfe_avg_pct']:>8.4f} | {r['mfe_med_pct']:>8.4f} | "
              f"{r['vol_ratio_avg']:>8.2f} | {r['strength_avg']:>8.4f}")

    print("\n" + "="*75)
    print(f"NIVEL 2 — OR {OR_MUESTRA} min — MFE por QUINTIL de VOL_RATIO")
    print("="*75)
    sub = quintil_vol[quintil_vol['or_duration_min'] == OR_MUESTRA]
    print(f"{'Quintil':>6} | {'vol_ratio_avg':>13} | {'N':>5} | {'MFE_avg%':>8} | {'MFE_med%':>8}")
    print("-"*75)
    for _, r in sub.iterrows():
        print(f"{r['quintil']:>6} | {r['vol_ratio_avg']:>13.2f} | {int(r['n']):>5} | "
              f"{r['mfe_avg_pct']:>8.4f} | {r['mfe_med_pct']:>8.4f}")

    print("\n" + "="*75)
    print(f"NIVEL 2 — OR {OR_MUESTRA} min — MFE por QUINTIL de BREAKOUT_STRENGTH")
    print("="*75)
    sub = quintil_str[quintil_str['or_duration_min'] == OR_MUESTRA]
    print(f"{'Quintil':>6} | {'strength_avg':>12} | {'N':>5} | {'MFE_avg%':>8} | {'MFE_med%':>8}")
    print("-"*75)
    for _, r in sub.iterrows():
        print(f"{r['quintil']:>6} | {r['breakout_strength_avg']:>12.4f} | {int(r['n']):>5} | "
              f"{r['mfe_avg_pct']:>8.4f} | {r['mfe_med_pct']:>8.4f}")

    print("\n" + "="*65)
    print(f"NIVEL 2 — OR {OR_MUESTRA} min — ALINEACIÓN con GAP")
    print("="*65)
    sub = gap_df[gap_df['or_duration_min'] == OR_MUESTRA]
    print(f"{'Alineación':>22} | {'N':>5} | {'MFE_avg%':>8} | {'MFE_med%':>8}")
    print("-"*65)
    for _, r in sub.iterrows():
        print(f"{r['alineacion']:>22} | {int(r['n']):>5} | "
              f"{r['mfe_avg_pct']:>8.4f} | {r['mfe_med_pct']:>8.4f}")

    print("\n" + "="*65)
    print(f"NIVEL 2 — OR {OR_MUESTRA} min — MFE por DÍA DE LA SEMANA")
    print("="*65)
    sub = dia_df[dia_df['or_duration_min'] == OR_MUESTRA]
    print(f"{'Día':>12} | {'N':>5} | {'%Up':>5} | {'MFE_avg%':>8} | {'MFE_med%':>8} | {'VolRatio':>8}")
    print("-"*65)
    for _, r in sub.iterrows():
        print(f"{r['dia_semana']:>12} | {int(r['n']):>5} | {r['pct_up']:>5.1f} | "
              f"{r['mfe_avg_pct']:>8.4f} | {r['mfe_med_pct']:>8.4f} | {r['vol_ratio_avg']:>8.2f}")

    print("\n" + "─"*65)
    print("ARCHIVOS GENERADOS en orb_results/:")
    print("  nivel2_raw.csv             → registro completo por (día × OR)")
    print("  nivel2_hora_mfe.csv        → MFE por franja horaria")
    print("  nivel2_volratio_mfe.csv    → MFE por quintil de vol_ratio")
    print("  nivel2_strength_mfe.csv    → MFE por quintil de breakout_strength")
    print("  nivel2_gap_dir.csv         → efecto alineación gap × breakout")
    print("  nivel2_dia_semana.csv      → MFE por día de la semana")
    print("\nCOLUMNAS CLAVE nivel2_raw.csv:")
    print("  breakout_minute    → minutos desde apertura hasta el breakout")
    print("  franja_horaria     → franja de 30 min donde ocurrió el breakout")
    print("  breakout_strength  → penetración normalizada por OR_size")
    print("  vol_ratio          → volumen vela break / promedio velas OR")
    print("  breakout_con_gap   → 1=alineado con presión interna, 0=contra")
    print("  dia_semana         → día de la semana del breakout")
    print("  mfe_pct            → excursión máxima en dirección del break (%)")
    print("─"*65)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Cargando datos desde: {DATA_FILE}")
    df = load_data(DATA_FILE)

    dates = df['date'].unique()
    print(f"Días encontrados: {len(dates)}")
    print(f"OR durations: {OR_DURATIONS}")
    total = len(dates) * len(OR_DURATIONS)
    print(f"Total combinaciones: {total:,}\n")

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
                validos = len(all_results)
                print(f"  {counter:,}/{total:,} ({pct:.1f}%) — registros válidos: {validos:,}")

    print(f"\nTotal registros válidos: {len(all_results):,}")

    raw_df = pd.DataFrame(all_results)

    # ── Guardar raw ───────────────────────────────────────────────────────────
    raw_path = os.path.join(OUTPUT_DIR, "nivel2_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"Raw guardado: {raw_path}  ({len(raw_df):,} filas)")

    # ── Resúmenes ─────────────────────────────────────────────────────────────
    franja_df    = resumen_por_franja(raw_df)
    quintil_vol  = resumen_por_quintil(raw_df, 'vol_ratio',          'vol_ratio')
    quintil_str  = resumen_por_quintil(raw_df, 'breakout_strength',  'breakout_strength')
    gap_df       = resumen_gap_dir(raw_df)
    dia_df       = resumen_dia_semana(raw_df)

    franja_df.to_csv(os.path.join(OUTPUT_DIR, "nivel2_hora_mfe.csv"),      index=False)
    quintil_vol.to_csv(os.path.join(OUTPUT_DIR, "nivel2_volratio_mfe.csv"),index=False)
    quintil_str.to_csv(os.path.join(OUTPUT_DIR, "nivel2_strength_mfe.csv"),index=False)
    gap_df.to_csv(os.path.join(OUTPUT_DIR, "nivel2_gap_dir.csv"),          index=False)
    dia_df.to_csv(os.path.join(OUTPUT_DIR, "nivel2_dia_semana.csv"),       index=False)

    # ── Print consola ─────────────────────────────────────────────────────────
    print_resumen_consola(raw_df, franja_df, quintil_vol, quintil_str, gap_df, dia_df)
    print("\nDone ✓")


if __name__ == "__main__":
    main()