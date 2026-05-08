"""
PASOS 3 Y 4 — ANÁLISIS DEL OUTCOME NEUTRO Y CORRELACIONES
===========================================================

PASO 3: Para los casos Neutros (no resueltos en 60 min),
        extiende la ventana a 120 minutos y al cierre de sesión.
        Responde: ¿el ratio Continuación/Falla de 9.6x se mantiene?

PASO 4: Tabla de correlaciones Phi entre las 6 condiciones C1-C6.
        Justifica que el score es aditivo porque las condiciones
        son genuinamente independientes entre sí.
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
INPUT_N4   = os.path.join("orb_results", "nivel4_raw.csv")
OUTPUT_DIR = "orb_results"
OR_MUESTRA = 15

# Condiciones del retest
ALEJAMIENTO_PCT = 0.20
TOLERANCIA_PCT  = 0.05
VENTANA_MIN     = 120
OUTCOME_VENTANA = 60   # ventana original del outcome

# Modelo combinado
C1_PERCENTIL = 60
C2_MINS_MAX  = 10
C3_MFE_MIN   = 0.30
C3_MFE_MAX   = 0.50
C4_PERC_LOW  = 40
C4_PERC_HIGH = 60
C5_BREAK_MAX = 30

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def load_intraday(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')
    df = df.set_index('datetime').sort_index()
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.between_time("09:30", "15:59")
    df['date'] = df.index.date
    return df


def load_nivel4(filepath: str, or_dur: int = None) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    if or_dur is not None:
        df = df[df['or_duration_min'] == or_dur].copy()
    return df.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# PASO 3 — RECONSTRUIR RETEST Y ANALIZAR NEUTROS
# ─────────────────────────────────────────────────────────────────────────────

def reconstruir_retest_y_analizar(day_df_raw: pd.DataFrame,
                                   or_duration: int,
                                   alejamiento_pct: float,
                                   tolerancia_pct: float,
                                   ventana_min: int) -> dict | None:
    """
    Para un día dado, detecta el breakout y el retest con los
    umbrales base, luego analiza el outcome en ventanas extendidas.
    Retorna None si no hay retest válido.
    """
    day_df = day_df_raw.copy().reset_index()

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

    or_vol_avg = or_bars['volume'].mean()
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
            breakout_idx, breakout_dir   = i, 'up'
            breakout_price, breakout_time = or_high, row['datetime']
            breakout_bar = row
            break
        elif row['low'] < or_low:
            breakout_idx, breakout_dir   = i, 'down'
            breakout_price, breakout_time = or_low, row['datetime']
            breakout_bar = row
            break

    if breakout_idx is None:
        return None

    post_break = post_or.iloc[breakout_idx:].reset_index(drop=True)
    if len(post_break) < 2:
        return None

    vol_ratio        = breakout_bar['volume'] / or_vol_avg
    breakout_minute  = int((breakout_time - open_time).seconds / 60)
    or_close         = or_bars['close'].iloc[-1]
    or_close_position = (or_close - or_low) / or_size
    or_size_pct      = or_size / or_low * 100

    # Umbrales del retest
    alejamiento_min = breakout_price * (alejamiento_pct / 100)
    tolerancia      = breakout_price * (tolerancia_pct  / 100)
    ventana_fin     = breakout_time + pd.Timedelta(minutes=ventana_min)

    # Buscar retest
    alejamiento_alcanzado = False
    mfe_corriendo = 0.0
    mfe_pre_pts   = 0.0
    retest_encontrado = False
    retest_time   = None
    retest_idx    = None

    for i, row in post_break.iterrows():
        if i == 0:
            continue
        if row['datetime'] > ventana_fin:
            break

        excursion = (row['high'] - breakout_price if breakout_dir == 'up'
                     else breakout_price - row['low'])
        if excursion > mfe_corriendo:
            mfe_corriendo = excursion

        if not alejamiento_alcanzado:
            if mfe_corriendo >= alejamiento_min:
                alejamiento_alcanzado = True
            continue

        toco = (row['low'] <= breakout_price + tolerancia if breakout_dir == 'up'
                else row['high'] >= breakout_price - tolerancia)

        if toco:
            pre_velas = post_break.iloc[1:i]
            if len(pre_velas) > 0:
                mfe_pre_pts = (pre_velas['high'].max() - breakout_price
                               if breakout_dir == 'up'
                               else breakout_price - pre_velas['low'].min())
            retest_encontrado = True
            retest_time  = row['datetime']
            retest_idx   = i
            break

    if not retest_encontrado:
        return None

    mfe_pre_retest_pct = mfe_pre_pts / breakout_price * 100
    mins_to_retest     = (retest_time - breakout_time).seconds / 60

    # Definir niveles de outcome
    if breakout_dir == 'up':
        nivel_cont  = breakout_price + mfe_pre_pts
        nivel_falla = or_low
    else:
        nivel_cont  = breakout_price - mfe_pre_pts
        nivel_falla = or_high

    post_retest = post_break.iloc[retest_idx:].reset_index(drop=True)

    # Calcular outcome en distintas ventanas
    def get_outcome(max_mins):
        fin = retest_time + pd.Timedelta(minutes=max_mins)
        for i, row in post_retest.iterrows():
            if i == 0:
                continue
            if row['datetime'] > fin:
                break
            if breakout_dir == 'up':
                if row['high'] >= nivel_cont:
                    return 'continuacion'
                if row['low']  <= nivel_falla:
                    return 'falla'
            else:
                if row['low']  <= nivel_cont:
                    return 'continuacion'
                if row['high'] >= nivel_falla:
                    return 'falla'
        return 'neutro'

    def get_outcome_cierre():
        for i, row in post_retest.iterrows():
            if i == 0:
                continue
            if breakout_dir == 'up':
                if row['high'] >= nivel_cont:
                    return 'continuacion', (row['datetime'] - retest_time).seconds / 60
                if row['low']  <= nivel_falla:
                    return 'falla', (row['datetime'] - retest_time).seconds / 60
            else:
                if row['low']  <= nivel_cont:
                    return 'continuacion', (row['datetime'] - retest_time).seconds / 60
                if row['high'] >= nivel_falla:
                    return 'falla', (row['datetime'] - retest_time).seconds / 60
        return 'neutro', np.nan

    outcome_60  = get_outcome(OUTCOME_VENTANA)
    outcome_120 = get_outcome(120)
    outcome_end, mins_res = get_outcome_cierre()

    return {
        'date':               str(open_time.date()),
        'or_size_pct':        round(or_size_pct, 4),
        'or_close_position':  round(or_close_position, 4),
        'breakout_dir':       breakout_dir,
        'breakout_minute':    breakout_minute,
        'vol_ratio':          round(vol_ratio, 4),
        'mfe_pre_retest_pct': round(mfe_pre_retest_pct, 4),
        'mins_to_retest':     round(mins_to_retest, 1),
        'outcome_60min':      outcome_60,
        'outcome_120min':     outcome_120,
        'outcome_cierre':     outcome_end,
        'mins_resolution':    round(mins_res, 1) if not np.isnan(mins_res) else np.nan,
    }


def calcular_ratios(resultados_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el ratio Cont/Falla para las tres definiciones de ventana.
    """
    rows = []
    for col, label in [
        ('outcome_60min',  '60 min (original)'),
        ('outcome_120min', '120 min (extendida)'),
        ('outcome_cierre', 'Hasta cierre de sesión'),
    ]:
        n_total = len(resultados_df)
        n_cont  = (resultados_df[col] == 'continuacion').sum()
        n_falla = (resultados_df[col] == 'falla').sum()
        n_neut  = (resultados_df[col] == 'neutro').sum()
        ratio   = n_cont / n_falla if n_falla > 0 else np.nan
        rows.append({
            'ventana':          label,
            'n_total':          n_total,
            'n_continuacion':   int(n_cont),
            'n_falla':          int(n_falla),
            'n_neutro':         int(n_neut),
            'pct_continuacion': round(n_cont  / n_total * 100, 1),
            'pct_falla':        round(n_falla / n_total * 100, 1),
            'pct_neutro':       round(n_neut  / n_total * 100, 1),
            'ratio_cont_falla': round(ratio, 2) if not np.isnan(ratio) else np.nan,
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# PASO 4 — CORRELACIONES
# ─────────────────────────────────────────────────────────────────────────────

def paso4_correlaciones(df_n4_full: pd.DataFrame) -> tuple:
    """
    Calcula correlación Phi entre las 6 condiciones binarias.
    Usa solo OR=15 min para consistencia con el resto del análisis.
    """
    df = df_n4_full[df_n4_full['or_duration_min'] == OR_MUESTRA].copy()

    if len(df) == 0:
        raise ValueError(f"No hay datos para OR={OR_MUESTRA} min")

    # Calcular umbrales
    c1_thresh = df['or_size_pct'].quantile(C1_PERCENTIL / 100)
    c4_lo     = df['vol_ratio'].quantile(C4_PERC_LOW  / 100)
    c4_hi     = df['vol_ratio'].quantile(C4_PERC_HIGH / 100)

    # Asignar condiciones
    df['C1'] = (df['or_size_pct'] >= c1_thresh).astype(int)
    df['C2'] = (df['mins_to_retest'] <= C2_MINS_MAX).astype(int)
    df['C3'] = ((df['mfe_pre_retest_pct'] >= C3_MFE_MIN) &
                (df['mfe_pre_retest_pct'] <  C3_MFE_MAX)).astype(int)
    df['C4'] = ((df['vol_ratio'] >= c4_lo) &
                (df['vol_ratio'] <= c4_hi)).astype(int)
    df['C5'] = (df['breakout_minute'] <= C5_BREAK_MAX).astype(int)
    df['C6'] = (
        ((df['or_close_position'] > 0.67) & (df['breakout_dir'] == 'up')) |
        ((df['or_close_position'] < 0.33) & (df['breakout_dir'] == 'down'))
    ).astype(int)

    conds = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    descripciones = {
        'C1': 'OR_grande (top 40%)',
        'C2': 'Retest ≤10 min',
        'C3': 'MFE_pre 0.30-0.50%',
        'C4': 'Vol_ratio p40-p60',
        'C5': 'Breakout ≤30 min',
        'C6': 'Sesgo alineado',
    }

    # Tasa de activación
    tasa_rows = []
    for c in conds:
        tasa_rows.append({
            'condicion':           c,
            'descripcion':         descripciones[c],
            'tasa_activacion_pct': round(df[c].mean() * 100, 1),
            'n_activa':            int(df[c].sum()),
            'n_total':             len(df),
        })
    tasa_df = pd.DataFrame(tasa_rows)

    # Matriz de correlación Phi
    corr_matrix = df[conds].corr(method='pearson')

    # Pares ordenados
    corr_long = []
    for i in range(len(conds)):
        for j in range(i+1, len(conds)):
            c_a, c_b = conds[i], conds[j]
            phi      = corr_matrix.loc[c_a, c_b]
            abs_phi  = abs(phi)
            if abs_phi < 0.10:
                nivel = 'Negligible (<0.10)'
            elif abs_phi < 0.20:
                nivel = 'Débil (0.10-0.20)'
            elif abs_phi < 0.30:
                nivel = 'Moderada (0.20-0.30)'
            else:
                nivel = '⚠ Sustancial (>0.30)'
            corr_long.append({
                'C_A': c_a, 'C_B': c_b,
                'desc_A': descripciones[c_a],
                'desc_B': descripciones[c_b],
                'phi': round(phi, 4),
                'abs_phi': round(abs_phi, 4),
                'nivel': nivel,
            })

    corr_long_df = pd.DataFrame(corr_long).sort_values('abs_phi', ascending=False)
    return corr_matrix, corr_long_df, tasa_df

# ─────────────────────────────────────────────────────────────────────────────
# PRINT CONSOLA
# ─────────────────────────────────────────────────────────────────────────────

def print_consola(all_results: pd.DataFrame,
                  ratios_df: pd.DataFrame,
                  corr_matrix: pd.DataFrame,
                  corr_long_df: pd.DataFrame,
                  tasa_df: pd.DataFrame):

    conds = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']

    # ── PASO 3: Resolución de Neutros ─────────────────────────────────────────
    print("\n" + "="*75)
    print("PASO 3 — RESOLUCIÓN DE LOS CASOS NEUTROS (OR 15 min)")
    print("="*75)

    neutros = all_results[all_results['outcome_60min'] == 'neutro']
    n_neutros = len(neutros)
    print(f"\n  Total casos Neutros en ventana de 60 min: {n_neutros:,}")

    if n_neutros > 0:
        # Entre 60 y 120 minutos
        n_cont_120 = (neutros['outcome_120min'] == 'continuacion').sum()
        n_fall_120 = (neutros['outcome_120min'] == 'falla').sum()
        n_neut_120 = (neutros['outcome_120min'] == 'neutro').sum()

        print(f"\n  ¿Cómo se resuelven entre 60 y 120 minutos?")
        print(f"    → Continuación:          {n_cont_120:>3} ({n_cont_120/n_neutros*100:.1f}%)")
        print(f"    → Falla:                 {n_fall_120:>3} ({n_fall_120/n_neutros*100:.1f}%)")
        print(f"    → Siguen sin resolver:   {n_neut_120:>3} ({n_neut_120/n_neutros*100:.1f}%)")

        # Hasta cierre
        n_cont_end = (neutros['outcome_cierre'] == 'continuacion').sum()
        n_fall_end = (neutros['outcome_cierre'] == 'falla').sum()
        n_genuino  = (neutros['outcome_cierre'] == 'neutro').sum()

        print(f"\n  ¿Cómo se resuelven hasta el cierre de sesión?")
        print(f"    → Continuación:          {n_cont_end:>3} ({n_cont_end/n_neutros*100:.1f}%)")
        print(f"    → Falla:                 {n_fall_end:>3} ({n_fall_end/n_neutros*100:.1f}%)")
        print(f"    → Genuinamente neutros:  {n_genuino:>3} ({n_genuino/n_neutros*100:.1f}%)")

        resueltos = neutros[neutros['outcome_cierre'] != 'neutro']
        if len(resueltos) > 0:
            print(f"\n  Tiempo de resolución tardía:")
            print(f"    → Promedio: {resueltos['mins_resolution'].mean():.1f} min desde el retest")
            print(f"    → Mediana:  {resueltos['mins_resolution'].median():.1f} min")

    # ── Tabla de ratios ───────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("PASO 3 — RATIO CONTINUACIÓN/FALLA BAJO DISTINTAS VENTANAS")
    print("¿El ratio de 9.6x del modelo combinado se deteriora")
    print("cuando los Neutros eventualmente se resuelven?")
    print(f"{'='*80}")
    print(f"  {'Ventana':<26} | {'N_Cont':>6} | {'N_Falla':>7} | "
          f"{'%Cont':>6} | {'%Falla':>7} | {'%Neut':>6} | {'Ratio C/F':>9}")
    print(f"  {'-'*75}")
    for _, r in ratios_df.iterrows():
        ratio_str = f"{r['ratio_cont_falla']:.2f}x" if not pd.isna(r['ratio_cont_falla']) else '  N/A'
        print(f"  {r['ventana']:<26} | {int(r['n_continuacion']):>6} | "
              f"{int(r['n_falla']):>7} | {r['pct_continuacion']:>6.1f} | "
              f"{r['pct_falla']:>7.1f} | {r['pct_neutro']:>6.1f} | {ratio_str:>9}")

    # ── PASO 4: Tasas de activación ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PASO 4 — TASA DE ACTIVACIÓN DE CADA CONDICIÓN (OR 15 min)")
    print("="*70)
    print(f"  {'Cond':>5} | {'Descripción':<25} | {'%Activa':>7} | {'N_activa':>8}")
    print(f"  {'-'*55}")
    for _, r in tasa_df.iterrows():
        print(f"  {r['condicion']:>5} | {r['descripcion']:<25} | "
              f"{r['tasa_activacion_pct']:>7.1f} | {int(r['n_activa']):>8}")

    # ── PASO 4: Matriz de correlación ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PASO 4 — MATRIZ DE CORRELACIÓN PHI ENTRE CONDICIONES (OR 15 min)")
    print("φ≈0 → independientes | φ>0.30 → correlación sustancial (*)")
    print(f"{'='*70}")

    header = f"  {'':>4}"
    for c in conds:
        header += f" | {c:>7}"
    print(header)
    print(f"  {'-'*52}")
    for c_row in conds:
        row_str = f"  {c_row:>4}"
        for c_col in conds:
            val = corr_matrix.loc[c_row, c_col]
            if c_row == c_col:
                row_str += f" | {'1.000':>7}"
            else:
                marker = '*' if abs(val) > 0.30 else ' '
                row_str += f" | {val:>6.3f}{marker}"
        print(row_str)

    # ── Pares ordenados ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PASO 4 — PARES ORDENADOS POR CORRELACIÓN (mayor a menor)")
    print(f"{'='*70}")
    print(f"  {'Par':>5} | {'φ':>7} | {'|φ|':>5} | Nivel")
    print(f"  {'-'*55}")
    for _, r in corr_long_df.iterrows():
        print(f"  {r['C_A']}–{r['C_B']:>3} | {r['phi']:>7.4f} | "
              f"{r['abs_phi']:>5.4f} | {r['nivel']}")

    max_corr     = corr_long_df['abs_phi'].max()
    n_sustancial = (corr_long_df['abs_phi'] > 0.30).sum()

    print(f"\n{'─'*70}")
    print(f"CONCLUSIÓN SOBRE ADITIVIDAD DEL SCORE:")
    print(f"  Correlación máxima entre condiciones: φ = {max_corr:.4f}")
    print(f"  Pares con correlación sustancial (>0.30): {n_sustancial}")
    if n_sustancial == 0:
        print(f"  ✓ Las 6 condiciones son sustancialmente independientes.")
        print(f"    La aditividad del score está justificada.")
    else:
        print(f"  ⚠ Hay correlación sustancial en {n_sustancial} par(es).")
        print(f"    La aditividad del score debe reportarse con cautela.")

    print(f"\n{'─'*70}")
    print("ARCHIVOS GENERADOS en orb_results/:")
    print("  neutro_resolucion.csv      → todos los casos con outcomes extendidos")
    print("  neutro_ratio_final.csv     → ratio Cont/Falla por ventana")
    print("  correlaciones_matriz.csv   → matriz Phi completa")
    print("  correlaciones_pares.csv    → pares ordenados por correlación")
    print("  correlaciones_tasas.csv    → tasa de activación por condición")
    print(f"{'─'*70}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Cargando datos intradía desde: {DATA_FILE}")
    df_raw = load_intraday(DATA_FILE)
    print(f"Días disponibles: {df_raw['date'].nunique():,}")

    print(f"\nCargando nivel4_raw (OR {OR_MUESTRA} min)...")
    df_n4     = load_nivel4(INPUT_N4, OR_MUESTRA)
    df_n4_all = load_nivel4(INPUT_N4)          # todos los OR
    print(f"Registros OR {OR_MUESTRA} min: {len(df_n4):,}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 3: Reconstruir todos los casos desde cero con ventanas extendidas
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("PASO 3: Reconstruyendo breakouts y retests con ventana extendida...")
    print("(Esto puede tardar 3-5 minutos)")

    dates   = df_raw['date'].unique()
    all_res = []
    counter = 0

    for date in dates:
        day_df = df_raw[df_raw['date'] == date]
        result = reconstruir_retest_y_analizar(
            day_df, OR_MUESTRA,
            ALEJAMIENTO_PCT, TOLERANCIA_PCT, VENTANA_MIN
        )
        if result:
            all_res.append(result)
        counter += 1
        if counter % 300 == 0:
            print(f"  {counter}/{len(dates)} días procesados — "
                  f"retests: {len(all_res):,}")

    all_results_df = pd.DataFrame(all_res)
    print(f"  Total días procesados: {len(dates):,}")
    print(f"  Total retests detectados: {len(all_results_df):,}")

    # Verificar consistencia con nivel4_raw
    n_original = len(df_n4)
    n_nuevo    = len(all_results_df)
    print(f"\n  Verificación: nivel4_raw tiene {n_original:,} retests, "
          f"reconstrucción tiene {n_nuevo:,}")
    if abs(n_original - n_nuevo) > 10:
        print("  ⚠ Diferencia > 10 registros — verificar umbrales")
    else:
        print("  ✓ Consistente")

    # Guardar resultados
    all_results_df.to_csv(
        os.path.join(OUTPUT_DIR, "neutro_resolucion.csv"), index=False)

    # Calcular ratios
    ratios_df = calcular_ratios(all_results_df)
    ratios_df.to_csv(
        os.path.join(OUTPUT_DIR, "neutro_ratio_final.csv"), index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 4: Correlaciones
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("PASO 4: Calculando correlaciones entre condiciones...")

    corr_matrix, corr_long_df, tasa_df = paso4_correlaciones(df_n4_all)

    corr_matrix.to_csv(
        os.path.join(OUTPUT_DIR, "correlaciones_matriz.csv"))
    corr_long_df.to_csv(
        os.path.join(OUTPUT_DIR, "correlaciones_pares.csv"), index=False)
    tasa_df.to_csv(
        os.path.join(OUTPUT_DIR, "correlaciones_tasas.csv"), index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Print consola
    # ─────────────────────────────────────────────────────────────────────────
    print_consola(all_results_df, ratios_df, corr_matrix, corr_long_df, tasa_df)
    print("\nDone ✓")


if __name__ == "__main__":
    main()