"""
TESTS ESTADÍSTICOS FORMALES
============================
Implementa 5 tipos de tests sobre los resultados de los Niveles 4 y 5:

  TEST 1: Intervalos de confianza de Wilson para tasas de continuación
  TEST 2: Chi-cuadrado de independencia para tablas de contingencia
  TEST 3: Cochran-Armitage para tendencias monotónicas
  TEST 4: Bootstrap para validación de intervalos
  TEST 5: Diferencia de proporciones entre subperíodos

Outputs:
  tests_wilson_ic.csv         → IC 95% para todas las combinaciones clave
  tests_chi2_contingencia.csv → chi-cuadrado para variables predictoras
  tests_tendencia.csv         → Cochran-Armitage para MFE_pre y mins_retest
  tests_bootstrap.csv         → IC bootstrap para combinaciones principales
  tests_subperiodos.csv       → comparación formal entre regímenes
  tests_resumen.csv           → tabla maestra con todos los resultados
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE  = os.path.join("orb_results", "nivel4_raw.csv")
OUTPUT_DIR  = "orb_results"
OR_MUESTRA  = 15
N_BOOTSTRAP = 10000
ALPHA       = 0.05
RANDOM_SEED = 42

# Umbrales del modelo combinado (mismos que nivel5.py)
C1_PERCENTIL = 60
C2_MINS_MAX  = 10
C3_MFE_MIN   = 0.30
C3_MFE_MAX   = 0.50
C4_PERC_LOW  = 40
C4_PERC_HIGH = 60
C5_BREAK_MAX = 30

SUBPERIODOS = {
    'P1_2017_2019':  ('2017-01-01', '2019-12-31'),
    'P2_2020_COVID': ('2020-01-01', '2020-12-31'),
    'P3_2021_bull':  ('2021-01-01', '2021-12-31'),
    'P4_2022_bear':  ('2022-01-01', '2022-12-31'),
    'P5_2023_2024':  ('2023-01-01', '2024-12-31'),
}

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES ESTADÍSTICAS BASE
# ─────────────────────────────────────────────────────────────────────────────

def wilson_ic(n_success: int, n_total: int,
              alpha: float = 0.05) -> tuple[float, float]:
    """
    Intervalo de confianza de Wilson para una proporción.
    Más preciso que el intervalo normal, especialmente en extremos.

    Returns:
        (lower, upper) bounds del IC al nivel (1-alpha)
    """
    if n_total == 0:
        return (np.nan, np.nan)

    p = n_success / n_total
    z = stats.norm.ppf(1 - alpha / 2)
    z2 = z ** 2

    center = (p + z2 / (2 * n_total)) / (1 + z2 / n_total)
    margin = (z / (1 + z2 / n_total)) * np.sqrt(
        p * (1 - p) / n_total + z2 / (4 * n_total ** 2)
    )

    return (max(0.0, center - margin), min(1.0, center + margin))


def chi2_test(contingency_table: np.ndarray) -> tuple[float, float, int]:
    """
    Test chi-cuadrado de independencia sobre tabla de contingencia.

    Returns:
        (chi2_stat, p_value, degrees_of_freedom)
    """
    chi2, p, dof, _ = chi2_contingency(contingency_table)
    return (chi2, p, dof)


def cochran_armitage_test(counts_success: list[int],
                           counts_total: list[int]) -> tuple[float, float]:
    """
    Test de Cochran-Armitage para tendencia en proporciones ordenadas.
    Detecta si hay una tendencia estadísticamente significativa
    (creciente o decreciente) en las tasas a través de grupos ordenados.

    Returns:
        (z_statistic, p_value_two_tailed)
    """
    k = len(counts_success)
    scores = list(range(k))  # scores ordinales 0, 1, 2, ...

    n = np.array(counts_total)
    x = np.array(counts_success)
    t = np.array(scores)

    N = n.sum()
    if N == 0:
        return (np.nan, np.nan)

    p_total = x.sum() / N

    # Estadístico de Cochran-Armitage
    numerator   = (t * (x - n * p_total)).sum()
    denominator = np.sqrt(
        p_total * (1 - p_total) * (N * (t ** 2 * n).sum() - ((t * n).sum()) ** 2) / N
    )

    if denominator == 0:
        return (np.nan, np.nan)

    z = numerator / denominator
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return (z, p_value)


def bootstrap_proportion(successes: np.ndarray, n_bootstrap: int = 10000,
                          alpha: float = 0.05,
                          seed: int = 42) -> tuple[float, float, float]:
    """
    Bootstrap para intervalo de confianza de una proporción.
    Remuestrea con reemplazo n_bootstrap veces.

    Args:
        successes: array binario (1=continuación, 0=otro)

    Returns:
        (observed_proportion, lower_ic, upper_ic)
    """
    rng = np.random.default_rng(seed)
    n   = len(successes)
    obs = successes.mean()

    boot_means = np.array([
        rng.choice(successes, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    lower = np.percentile(boot_means, alpha / 2 * 100)
    upper = np.percentile(boot_means, (1 - alpha / 2) * 100)

    return (obs, lower, upper)


def test_diferencia_proporciones(n1: int, p1: float,
                                  n2: int, p2: float) -> tuple[float, float]:
    """
    Test z para diferencia entre dos proporciones independientes.

    Returns:
        (z_statistic, p_value_two_tailed)
    """
    if n1 == 0 or n2 == 0:
        return (np.nan, np.nan)

    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se     = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

    if se == 0:
        return (np.nan, np.nan)

    z       = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return (z, p_value)


def significancia(p_value: float) -> str:
    """Convierte p-valor en símbolo estándar de significancia."""
    if pd.isna(p_value):
        return "N/A"
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "n.s."

# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y PREPARACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df['is_cont'] = (df['outcome'] == 'continuacion').astype(int)

    # Agregar subperíodo
    df['subperiodo'] = 'fuera_rango'
    for nombre, (inicio, fin) in SUBPERIODOS.items():
        mask = (df['date'] >= inicio) & (df['date'] <= fin)
        df.loc[mask, 'subperiodo'] = nombre

    # Agregar condiciones binarias (misma lógica que nivel5.py)
    rows = []
    for or_dur, grp in df.groupby('or_duration_min'):
        grp = grp.copy()
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
        rows.append(grp)

    return pd.concat(rows, ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — INTERVALOS DE CONFIANZA DE WILSON
# ─────────────────────────────────────────────────────────────────────────────

def test1_wilson(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula IC de Wilson para:
    - Tasa base de continuación (total y por OR duration)
    - Combinaciones clave del modelo combinado
    - Tasa de continuación por bins de las variables predictoras clave
    """
    rows = []
    sub15 = df[df['or_duration_min'] == OR_MUESTRA]

    # ── Tasa base por OR duration ─────────────────────────────────────────────
    for or_dur, grp in df.groupby('or_duration_min'):
        n       = len(grp)
        n_cont  = grp['is_cont'].sum()
        lo, hi  = wilson_ic(n_cont, n)
        rows.append({
            'categoria':   f'Base_OR_{or_dur}min',
            'descripcion': f'Tasa base de continuación OR={or_dur} min',
            'n':           n,
            'n_cont':      int(n_cont),
            'pct_cont':    round(n_cont / n * 100, 2),
            'ic_lower':    round(lo * 100, 2),
            'ic_upper':    round(hi * 100, 2),
            'ic_width':    round((hi - lo) * 100, 2),
        })

    # ── Combinaciones clave ───────────────────────────────────────────────────
    combos = {
        'C1+C2+C4 (N~589)':  ['C1', 'C2', 'C4'],
        'C2+C4 (N~810)':     ['C2', 'C4'],
        'C1+C2 (N~2256)':    ['C1', 'C2'],
        'C1+C2+C4+C6':       ['C1', 'C2', 'C4', 'C6'],
        'C2+C4+C6':          ['C2', 'C4', 'C6'],
        'Solo C2':            ['C2'],
    }

    for label, conds in combos.items():
        mask = pd.Series([True] * len(df), index=df.index)
        for c in conds:
            mask = mask & (df[c] == 1)
        grp    = df[mask]
        n      = len(grp)
        n_cont = grp['is_cont'].sum()
        if n < 10:
            continue
        lo, hi = wilson_ic(n_cont, n)
        rows.append({
            'categoria':   f'Combo_{label}',
            'descripcion': f'Combinación: {label}',
            'n':           n,
            'n_cont':      int(n_cont),
            'pct_cont':    round(n_cont / n * 100, 2),
            'ic_lower':    round(lo * 100, 2),
            'ic_upper':    round(hi * 100, 2),
            'ic_width':    round((hi - lo) * 100, 2),
        })

    # ── Bins de mins_to_retest (OR 15 min) ───────────────────────────────────
    bins_mt = [(0, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 9999)]
    labels_mt = ['0-10min', '10-20min', '20-30min', '30-45min', '45-60min', '>60min']

    for (lo_b, hi_b), lbl in zip(bins_mt, labels_mt):
        grp    = sub15[(sub15['mins_to_retest'] >= lo_b) &
                       (sub15['mins_to_retest'] < hi_b)]
        n      = len(grp)
        n_cont = grp['is_cont'].sum()
        if n < 10:
            continue
        lo, hi = wilson_ic(n_cont, n)
        rows.append({
            'categoria':   f'MinsRetest_{lbl}',
            'descripcion': f'mins_to_retest = {lbl} (OR 15 min)',
            'n':           n,
            'n_cont':      int(n_cont),
            'pct_cont':    round(n_cont / n * 100, 2),
            'ic_lower':    round(lo * 100, 2),
            'ic_upper':    round(hi * 100, 2),
            'ic_width':    round((hi - lo) * 100, 2),
        })

    # ── Bins de mfe_pre_retest (OR 15 min) ───────────────────────────────────
    bins_mp = [(0.20, 0.30), (0.30, 0.40), (0.40, 0.50),
               (0.50, 0.75), (0.75, 1.00), (1.00, 999)]
    labels_mp = ['0.20-0.30%', '0.30-0.40%', '0.40-0.50%',
                 '0.50-0.75%', '0.75-1.00%', '>1.00%']

    for (lo_b, hi_b), lbl in zip(bins_mp, labels_mp):
        grp    = sub15[(sub15['mfe_pre_retest_pct'] >= lo_b) &
                       (sub15['mfe_pre_retest_pct'] < hi_b)]
        n      = len(grp)
        n_cont = grp['is_cont'].sum()
        if n < 10:
            continue
        lo, hi = wilson_ic(n_cont, n)
        rows.append({
            'categoria':   f'MFEpre_{lbl}',
            'descripcion': f'mfe_pre_retest = {lbl} (OR 15 min)',
            'n':           n,
            'n_cont':      int(n_cont),
            'pct_cont':    round(n_cont / n * 100, 2),
            'ic_lower':    round(lo * 100, 2),
            'ic_upper':    round(hi * 100, 2),
            'ic_width':    round((hi - lo) * 100, 2),
        })

    # ── Scores 0-5 (OR 15 min) ───────────────────────────────────────────────
    for score_val in range(6):
        grp    = sub15[sub15['score'] == score_val] if 'score' in sub15.columns else pd.DataFrame()
        if len(grp) < 5:
            continue
        n      = len(grp)
        n_cont = grp['is_cont'].sum()
        lo, hi = wilson_ic(n_cont, n)
        rows.append({
            'categoria':   f'Score_{score_val}',
            'descripcion': f'Score combinado = {score_val} (OR 15 min)',
            'n':           n,
            'n_cont':      int(n_cont),
            'pct_cont':    round(n_cont / n * 100, 2),
            'ic_lower':    round(lo * 100, 2),
            'ic_upper':    round(hi * 100, 2),
            'ic_width':    round((hi - lo) * 100, 2),
        })

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — CHI-CUADRADO
# ─────────────────────────────────────────────────────────────────────────────

def test2_chi2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test chi-cuadrado para independencia entre variables predictoras y outcome.
    Para cada variable, construye tabla de contingencia outcome × grupo.
    """
    rows = []
    sub15 = df[df['or_duration_min'] == OR_MUESTRA].copy()

    def run_chi2(sub: pd.DataFrame, var_name: str,
                 grupos: list, labels: list):
        table = []
        for (lo_b, hi_b) in grupos:
            if hi_b == 9999:
                g = sub[sub[var_name] >= lo_b]
            else:
                g = sub[(sub[var_name] >= lo_b) & (sub[var_name] < hi_b)]
            if len(g) < 5:
                continue
            n_cont   = (g['outcome'] == 'continuacion').sum()
            n_falla  = (g['outcome'] == 'falla').sum()
            n_neutro = (g['outcome'] == 'neutro').sum()
            table.append([n_cont, n_falla, n_neutro])

        if len(table) < 2:
            return None

        table_arr = np.array(table)
        # Eliminar columnas con suma cero
        col_sums = table_arr.sum(axis=0)
        table_arr = table_arr[:, col_sums > 0]

        if table_arr.shape[1] < 2:
            return None

        chi2_val, p_val, dof, _ = chi2_contingency(table_arr)
        return {
            'variable':    var_name,
            'n_grupos':    len(table),
            'chi2':        round(chi2_val, 3),
            'dof':         dof,
            'p_value':     round(p_val, 6),
            'significancia': significancia(p_val),
            'interpretacion': 'Asociación significativa' if p_val < 0.05
                              else 'Sin evidencia de asociación',
        }

    # mins_to_retest
    result = run_chi2(sub15, 'mins_to_retest',
                      [(0,10),(10,20),(20,30),(30,45),(45,60),(60,9999)],
                      ['0-10','10-20','20-30','30-45','45-60','>60'])
    if result:
        rows.append(result)

    # mfe_pre_retest_pct
    result = run_chi2(sub15, 'mfe_pre_retest_pct',
                      [(0.20,0.30),(0.30,0.40),(0.40,0.50),
                       (0.50,0.75),(0.75,1.00),(1.00,9999)],
                      ['0.20-0.30','0.30-0.40','0.40-0.50',
                       '0.50-0.75','0.75-1.00','>1.00'])
    if result:
        rows.append(result)

    # or_size_pct (quintiles)
    sub15_copy = sub15.copy()
    try:
        sub15_copy['or_size_q'] = pd.qcut(sub15_copy['or_size_pct'],
                                           q=5, labels=False,
                                           duplicates='drop')
        table = []
        for q in range(5):
            g = sub15_copy[sub15_copy['or_size_q'] == q]
            if len(g) < 5:
                continue
            table.append([
                (g['outcome'] == 'continuacion').sum(),
                (g['outcome'] == 'falla').sum(),
                (g['outcome'] == 'neutro').sum(),
            ])
        if len(table) >= 2:
            table_arr = np.array(table)
            col_sums  = table_arr.sum(axis=0)
            table_arr = table_arr[:, col_sums > 0]
            chi2_val, p_val, dof, _ = chi2_contingency(table_arr)
            rows.append({
                'variable':       'or_size_pct (quintiles)',
                'n_grupos':       len(table),
                'chi2':           round(chi2_val, 3),
                'dof':            dof,
                'p_value':        round(p_val, 6),
                'significancia':  significancia(p_val),
                'interpretacion': 'Asociación significativa' if p_val < 0.05
                                  else 'Sin evidencia de asociación',
            })
    except Exception:
        pass

    # vol_ratio (quintiles)
    sub15_copy2 = sub15.copy()
    try:
        sub15_copy2['vol_q'] = pd.qcut(sub15_copy2['vol_ratio'],
                                        q=5, labels=False,
                                        duplicates='drop')
        table = []
        for q in range(5):
            g = sub15_copy2[sub15_copy2['vol_q'] == q]
            if len(g) < 5:
                continue
            table.append([
                (g['outcome'] == 'continuacion').sum(),
                (g['outcome'] == 'falla').sum(),
                (g['outcome'] == 'neutro').sum(),
            ])
        if len(table) >= 2:
            table_arr = np.array(table)
            col_sums  = table_arr.sum(axis=0)
            table_arr = table_arr[:, col_sums > 0]
            chi2_val, p_val, dof, _ = chi2_contingency(table_arr)
            rows.append({
                'variable':       'vol_ratio (quintiles)',
                'n_grupos':       len(table),
                'chi2':           round(chi2_val, 3),
                'dof':            dof,
                'p_value':        round(p_val, 6),
                'significancia':  significancia(p_val),
                'interpretacion': 'Asociación significativa' if p_val < 0.05
                                  else 'Sin evidencia de asociación',
            })
    except Exception:
        pass

    # dia_semana
    sub15['dow'] = pd.to_datetime(sub15['date']).dt.day_name()
    table = []
    for dia in ['Monday','Tuesday','Wednesday','Thursday','Friday']:
        g = sub15[sub15['dow'] == dia]
        if len(g) < 5:
            continue
        table.append([
            (g['outcome'] == 'continuacion').sum(),
            (g['outcome'] == 'falla').sum(),
            (g['outcome'] == 'neutro').sum(),
        ])
    if len(table) >= 2:
        table_arr = np.array(table)
        col_sums  = table_arr.sum(axis=0)
        table_arr = table_arr[:, col_sums > 0]
        chi2_val, p_val, dof, _ = chi2_contingency(table_arr)
        rows.append({
            'variable':       'dia_semana',
            'n_grupos':       len(table),
            'chi2':           round(chi2_val, 3),
            'dof':            dof,
            'p_value':        round(p_val, 6),
            'significancia':  significancia(p_val),
            'interpretacion': 'Asociación significativa' if p_val < 0.05
                              else 'Sin evidencia de asociación',
        })

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — COCHRAN-ARMITAGE (TENDENCIAS)
# ─────────────────────────────────────────────────────────────────────────────

def test3_tendencias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test de Cochran-Armitage para tendencias monotónicas en:
    1. mins_to_retest → tasa de continuación (esperamos tendencia decreciente)
    2. mfe_pre_retest → tasa de continuación (esperamos tendencia decreciente)
    3. or_size_pct (quintiles) → tasa de continuación (esperamos tendencia creciente)
    4. score → tasa de continuación (esperamos tendencia creciente)
    """
    rows = []
    sub15 = df[df['or_duration_min'] == OR_MUESTRA].copy()

    # ── Variable 1: mins_to_retest ────────────────────────────────────────────
    bins_mt = [(0,10),(10,20),(20,30),(30,45),(45,60),(60,9999)]
    successes, totals = [], []
    for (lo_b, hi_b) in bins_mt:
        if hi_b == 9999:
            g = sub15[sub15['mins_to_retest'] >= lo_b]
        else:
            g = sub15[(sub15['mins_to_retest'] >= lo_b) &
                      (sub15['mins_to_retest'] < hi_b)]
        if len(g) >= 5:
            successes.append(int(g['is_cont'].sum()))
            totals.append(len(g))

    z, p = cochran_armitage_test(successes, totals)
    rows.append({
        'variable':       'mins_to_retest',
        'hipotesis':      'Tendencia decreciente: más tiempo → menos continuación',
        'n_grupos':       len(successes),
        'z_statistic':    round(z, 3) if not np.isnan(z) else np.nan,
        'p_value':        round(p, 8) if not np.isnan(p) else np.nan,
        'significancia':  significancia(p),
        'direccion':      'Decreciente (z<0)' if z < 0 else 'Creciente (z>0)',
        'interpretacion': 'Tendencia significativa' if p < 0.05
                          else 'Sin tendencia significativa',
    })

    # ── Variable 2: mfe_pre_retest_pct ────────────────────────────────────────
    bins_mp = [(0.20,0.30),(0.30,0.40),(0.40,0.50),
               (0.50,0.75),(0.75,1.00),(1.00,9999)]
    successes, totals = [], []
    for (lo_b, hi_b) in bins_mp:
        if hi_b == 9999:
            g = sub15[sub15['mfe_pre_retest_pct'] >= lo_b]
        else:
            g = sub15[(sub15['mfe_pre_retest_pct'] >= lo_b) &
                      (sub15['mfe_pre_retest_pct'] < hi_b)]
        if len(g) >= 5:
            successes.append(int(g['is_cont'].sum()))
            totals.append(len(g))

    z, p = cochran_armitage_test(successes, totals)
    rows.append({
        'variable':       'mfe_pre_retest_pct',
        'hipotesis':      'Tendencia decreciente: mayor MFE_pre → menos continuación',
        'n_grupos':       len(successes),
        'z_statistic':    round(z, 3) if not np.isnan(z) else np.nan,
        'p_value':        round(p, 8) if not np.isnan(p) else np.nan,
        'significancia':  significancia(p),
        'direccion':      'Decreciente (z<0)' if z < 0 else 'Creciente (z>0)',
        'interpretacion': 'Tendencia significativa' if p < 0.05
                          else 'Sin tendencia significativa',
    })

    # ── Variable 3: or_size_pct (quintiles) ───────────────────────────────────
    sub15_copy = sub15.copy()
    try:
        sub15_copy['or_q'] = pd.qcut(sub15_copy['or_size_pct'],
                                      q=5, labels=False, duplicates='drop')
        successes, totals = [], []
        for q in range(5):
            g = sub15_copy[sub15_copy['or_q'] == q]
            if len(g) >= 5:
                successes.append(int(g['is_cont'].sum()))
                totals.append(len(g))

        z, p = cochran_armitage_test(successes, totals)
        rows.append({
            'variable':       'or_size_pct (quintiles)',
            'hipotesis':      'Tendencia creciente: mayor OR → más continuación',
            'n_grupos':       len(successes),
            'z_statistic':    round(z, 3) if not np.isnan(z) else np.nan,
            'p_value':        round(p, 8) if not np.isnan(p) else np.nan,
            'significancia':  significancia(p),
            'direccion':      'Decreciente (z<0)' if z < 0 else 'Creciente (z>0)',
            'interpretacion': 'Tendencia significativa' if p < 0.05
                              else 'Sin tendencia significativa',
        })
    except Exception:
        pass

    # ── Variable 4: score combinado ───────────────────────────────────────────
    if 'score' in sub15.columns:
        successes, totals = [], []
        for score_val in sorted(sub15['score'].unique()):
            g = sub15[sub15['score'] == score_val]
            if len(g) >= 5:
                successes.append(int(g['is_cont'].sum()))
                totals.append(len(g))

        z, p = cochran_armitage_test(successes, totals)
        rows.append({
            'variable':       'score_combinado',
            'hipotesis':      'Tendencia creciente: mayor score → más continuación',
            'n_grupos':       len(successes),
            'z_statistic':    round(z, 3) if not np.isnan(z) else np.nan,
            'p_value':        round(p, 8) if not np.isnan(p) else np.nan,
            'significancia':  significancia(p),
            'direccion':      'Decreciente (z<0)' if z < 0 else 'Creciente (z>0)',
            'interpretacion': 'Tendencia significativa' if p < 0.05
                              else 'Sin tendencia significativa',
        })

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────

def test4_bootstrap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bootstrap IC para las combinaciones principales.
    Compara con Wilson IC — si coinciden, el resultado es doblemente robusto.
    """
    rows = []
    np.random.seed(RANDOM_SEED)

    combos = {
        'C1+C2+C4': ['C1', 'C2', 'C4'],
        'C2+C4':    ['C2', 'C4'],
        'C1+C2':    ['C1', 'C2'],
        'Solo_C2':  ['C2'],
    }

    print("  Calculando bootstrap (esto puede tomar ~30 segundos)...")

    for label, conds in combos.items():
        mask = pd.Series([True] * len(df), index=df.index)
        for c in conds:
            mask = mask & (df[c] == 1)
        grp = df[mask]
        n   = len(grp)

        if n < 20:
            continue

        successes = grp['is_cont'].values
        obs, boot_lo, boot_hi = bootstrap_proportion(
            successes, N_BOOTSTRAP, ALPHA, RANDOM_SEED
        )
        wilson_lo, wilson_hi = wilson_ic(int(successes.sum()), n, ALPHA)

        rows.append({
            'combinacion':      label,
            'n':                n,
            'pct_cont_obs':     round(obs * 100, 2),
            'wilson_lower':     round(wilson_lo * 100, 2),
            'wilson_upper':     round(wilson_hi * 100, 2),
            'bootstrap_lower':  round(boot_lo * 100, 2),
            'bootstrap_upper':  round(boot_hi * 100, 2),
            'coinciden':        abs(boot_lo - wilson_lo) < 0.02 and
                                abs(boot_hi - wilson_hi) < 0.02,
        })

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — DIFERENCIA ENTRE SUBPERÍODOS
# ─────────────────────────────────────────────────────────────────────────────

def test5_subperiodos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test de diferencia de proporciones entre todos los pares de subperíodos.
    Foco en las comparaciones más importantes:
    - P4 (bear) vs cada otro subperíodo
    - P3 (bull 2021) vs cada otro subperíodo
    """
    rows = []
    sub15 = df[df['or_duration_min'] == OR_MUESTRA]

    # Calcular tasa por subperíodo
    stats_por_periodo = {}
    for nombre in SUBPERIODOS:
        g = sub15[sub15['subperiodo'] == nombre]
        if len(g) >= 20:
            stats_por_periodo[nombre] = {
                'n':    len(g),
                'p':    g['is_cont'].mean(),
                'n_c':  int(g['is_cont'].sum()),
            }

    # Agregar total
    stats_por_periodo['TOTAL'] = {
        'n': len(sub15),
        'p': sub15['is_cont'].mean(),
        'n_c': int(sub15['is_cont'].sum()),
    }

    # Todos los pares
    periodos = list(stats_por_periodo.keys())
    for i in range(len(periodos)):
        for j in range(i+1, len(periodos)):
            p1_name = periodos[i]
            p2_name = periodos[j]
            s1 = stats_por_periodo[p1_name]
            s2 = stats_por_periodo[p2_name]

            z, p_val = test_diferencia_proporciones(
                s1['n'], s1['p'],
                s2['n'], s2['p']
            )

            rows.append({
                'periodo_1':    p1_name,
                'n1':           s1['n'],
                'pct_cont_1':   round(s1['p'] * 100, 1),
                'periodo_2':    p2_name,
                'n2':           s2['n'],
                'pct_cont_2':   round(s2['p'] * 100, 1),
                'diff_pp':      round((s1['p'] - s2['p']) * 100, 1),
                'z_statistic':  round(z, 3) if not np.isnan(z) else np.nan,
                'p_value':      round(p_val, 6) if not np.isnan(p_val) else np.nan,
                'significancia': significancia(p_val),
            })

    return pd.DataFrame(rows).sort_values('p_value')

# ─────────────────────────────────────────────────────────────────────────────
# PRINT CONSOLA
# ─────────────────────────────────────────────────────────────────────────────

def print_consola(wilson_df, chi2_df, tend_df, boot_df, subp_df):

    # ── TEST 1: Wilson IC ─────────────────────────────────────────────────────
    print("\n" + "="*90)
    print("TEST 1 — INTERVALOS DE CONFIANZA DE WILSON (95%)")
    print("="*90)

    # Tasa base OR 15 min
    base = wilson_df[wilson_df['categoria'] == f'Base_OR_{OR_MUESTRA}min']
    if len(base) > 0:
        r = base.iloc[0]
        print(f"\n  Tasa base OR {OR_MUESTRA} min:")
        print(f"  N={r['n']:,}  |  {r['pct_cont']:.1f}%  "
              f"[{r['ic_lower']:.1f}%, {r['ic_upper']:.1f}%]  "
              f"(ancho IC: {r['ic_width']:.1f}pp)")

    # Combinaciones
    print(f"\n  {'Combinación':<30} | {'N':>5} | {'%Cont':>6} | "
          f"{'IC_low':>7} | {'IC_high':>7} | {'Ancho':>5}")
    print(f"  {'-'*72}")
    combos = wilson_df[wilson_df['categoria'].str.startswith('Combo_')]
    for _, r in combos.iterrows():
        print(f"  {r['descripcion'][:28]:<30} | {int(r['n']):>5} | "
              f"{r['pct_cont']:>6.1f} | {r['ic_lower']:>7.1f} | "
              f"{r['ic_upper']:>7.1f} | {r['ic_width']:>5.1f}")

    # mins_to_retest
    print(f"\n  mins_to_retest → continuación (OR {OR_MUESTRA} min):")
    print(f"  {'Bin':<12} | {'N':>5} | {'%Cont':>6} | {'IC_low':>7} | {'IC_high':>7}")
    print(f"  {'-'*48}")
    mt_rows = wilson_df[wilson_df['categoria'].str.startswith('MinsRetest_')]
    for _, r in mt_rows.iterrows():
        bin_name = r['categoria'].replace('MinsRetest_', '')
        print(f"  {bin_name:<12} | {int(r['n']):>5} | {r['pct_cont']:>6.1f} | "
              f"{r['ic_lower']:>7.1f} | {r['ic_upper']:>7.1f}")

    # mfe_pre
    print(f"\n  mfe_pre_retest_pct → continuación (OR {OR_MUESTRA} min):")
    print(f"  {'Bin':<12} | {'N':>5} | {'%Cont':>6} | {'IC_low':>7} | {'IC_high':>7}")
    print(f"  {'-'*48}")
    mp_rows = wilson_df[wilson_df['categoria'].str.startswith('MFEpre_')]
    for _, r in mp_rows.iterrows():
        bin_name = r['categoria'].replace('MFEpre_', '')
        print(f"  {bin_name:<12} | {int(r['n']):>5} | {r['pct_cont']:>6.1f} | "
              f"{r['ic_lower']:>7.1f} | {r['ic_upper']:>7.1f}")

    # ── TEST 2: Chi-cuadrado ──────────────────────────────────────────────────
    print("\n" + "="*80)
    print("TEST 2 — CHI-CUADRADO DE INDEPENDENCIA (OR 15 min)")
    print("="*80)
    print(f"  {'Variable':<30} | {'χ²':>8} | {'dof':>3} | "
          f"{'p-valor':>10} | {'Sig':>4} | Interpretación")
    print(f"  {'-'*80}")
    for _, r in chi2_df.iterrows():
        print(f"  {r['variable']:<30} | {r['chi2']:>8.3f} | {r['dof']:>3} | "
              f"{r['p_value']:>10.6f} | {r['significancia']:>4} | "
              f"{r['interpretacion']}")

    # ── TEST 3: Cochran-Armitage ──────────────────────────────────────────────
    print("\n" + "="*80)
    print("TEST 3 — COCHRAN-ARMITAGE (TENDENCIAS MONOTÓNICAS, OR 15 min)")
    print("Hipótesis: ¿existe tendencia estadísticamente significativa?")
    print("="*80)
    print(f"  {'Variable':<25} | {'Z':>7} | {'p-valor':>10} | "
          f"{'Sig':>4} | {'Dirección':>15}")
    print(f"  {'-'*72}")
    for _, r in tend_df.iterrows():
        z_str = f"{r['z_statistic']:.3f}" if not pd.isna(r['z_statistic']) else "  N/A"
        p_str = f"{r['p_value']:.8f}"     if not pd.isna(r['p_value'])     else "     N/A"
        print(f"  {r['variable']:<25} | {z_str:>7} | {p_str:>10} | "
              f"{r['significancia']:>4} | {r['direccion']:>15}")

    # ── TEST 4: Bootstrap ─────────────────────────────────────────────────────
    print("\n" + "="*85)
    print("TEST 4 — BOOTSTRAP IC vs WILSON IC (N=10,000 iteraciones)")
    print("Si coinciden → resultado doblemente robusto")
    print("="*85)
    print(f"  {'Combinación':<15} | {'N':>5} | {'%Obs':>5} | "
          f"{'Wilson IC':>16} | {'Bootstrap IC':>16} | {'¿Coinciden?':>11}")
    print(f"  {'-'*78}")
    for _, r in boot_df.iterrows():
        coin = "✓ Sí" if r['coinciden'] else "✗ No"
        print(f"  {r['combinacion']:<15} | {int(r['n']):>5} | "
              f"{r['pct_cont_obs']:>5.1f} | "
              f"[{r['wilson_lower']:.1f}, {r['wilson_upper']:.1f}]  "
              f"  | [{r['bootstrap_lower']:.1f}, "
              f"{r['bootstrap_upper']:.1f}]   | {coin:>11}")

    # ── TEST 5: Subperíodos ───────────────────────────────────────────────────
    print("\n" + "="*85)
    print("TEST 5 — DIFERENCIA DE PROPORCIONES ENTRE SUBPERÍODOS (OR 15 min)")
    print("Pares más importantes ordenados por p-valor")
    print("="*85)
    print(f"  {'Período 1':<20} | {'%1':>5} | {'Período 2':<20} | "
          f"{'%2':>5} | {'Δpp':>5} | {'Z':>6} | {'p-valor':>9} | {'Sig':>4}")
    print(f"  {'-'*90}")
    top_pairs = subp_df.head(15)
    for _, r in top_pairs.iterrows():
        z_str = f"{r['z_statistic']:.2f}" if not pd.isna(r['z_statistic']) else "N/A"
        p_str = f"{r['p_value']:.4f}"     if not pd.isna(r['p_value'])     else "N/A"
        print(f"  {r['periodo_1']:<20} | {r['pct_cont_1']:>5.1f} | "
              f"{r['periodo_2']:<20} | {r['pct_cont_2']:>5.1f} | "
              f"{r['diff_pp']:>+5.1f} | {z_str:>6} | {p_str:>9} | "
              f"{r['significancia']:>4}")

    print(f"\n  Leyenda: *** p<0.001 | ** p<0.01 | * p<0.05 | n.s. no significativo")

    print(f"\n{'─'*70}")
    print("ARCHIVOS GENERADOS en orb_results/:")
    print("  tests_wilson_ic.csv          → IC Wilson para todas las métricas")
    print("  tests_chi2_contingencia.csv  → Chi-cuadrado por variable")
    print("  tests_tendencia.csv          → Cochran-Armitage por variable")
    print("  tests_bootstrap.csv          → IC Bootstrap vs Wilson")
    print("  tests_subperiodos.csv        → Diferencia entre regímenes")
    print(f"{'─'*70}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Cargando datos desde: {INPUT_FILE}")
    df = load_and_prepare(INPUT_FILE)
    print(f"Registros cargados: {len(df):,}")
    print(f"OR {OR_MUESTRA} min — registros: "
          f"{len(df[df['or_duration_min']==OR_MUESTRA]):,}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("TEST 1: Calculando intervalos de confianza Wilson...")
    wilson_df = test1_wilson(df)
    wilson_df.to_csv(
        os.path.join(OUTPUT_DIR, "tests_wilson_ic.csv"), index=False)

    print("TEST 2: Calculando chi-cuadrado...")
    chi2_df = test2_chi2(df)
    chi2_df.to_csv(
        os.path.join(OUTPUT_DIR, "tests_chi2_contingencia.csv"), index=False)

    print("TEST 3: Calculando Cochran-Armitage...")
    tend_df = test3_tendencias(df)
    tend_df.to_csv(
        os.path.join(OUTPUT_DIR, "tests_tendencia.csv"), index=False)

    print("TEST 4: Calculando bootstrap (paciencia ~30 seg)...")
    boot_df = test4_bootstrap(df)
    boot_df.to_csv(
        os.path.join(OUTPUT_DIR, "tests_bootstrap.csv"), index=False)

    print("TEST 5: Calculando diferencias entre subperíodos...")
    subp_df = test5_subperiodos(df)
    subp_df.to_csv(
        os.path.join(OUTPUT_DIR, "tests_subperiodos.csv"), index=False)

    print_consola(wilson_df, chi2_df, tend_df, boot_df, subp_df)
    print("\nDone ✓")


if __name__ == "__main__":
    main()