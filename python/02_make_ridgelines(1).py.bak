#!/usr/bin/env python3

"""
02_make_ridgelines.py
---------------------

Generate ridgeline density plots of total phoneme inventory size by
language family and compute per‑family summary statistics.  The
script reads the raw PHOIBLE and Glottolog languoid tables, cleans
and deduplicates the data, summarises inventory sizes, and writes a
CSV of counts, medians and IQRs to `out/summary_tables.csv`.  The
ridgeline plot is saved as PDF, PNG and SVG in the `figs/` directory.

Because the environment lacks the `joypy` package, ridgelines are
constructed manually using Gaussian kernel density estimates from
SciPy.  Families with fewer than `min_n` languages are excluded from
the plot.

"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # ensure non‑interactive backend
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

RAW_DIR = os.path.join('data', 'raw')
OUT_DIR = 'out'
FIG_DIR = 'figs'

PHOIBLE_FILE = os.path.join(RAW_DIR, 'phoible.csv')
LANGUOID_FILE = os.path.join(RAW_DIR, 'languoid.csv')

MIN_N = 10  # minimum number of languages per family to include in plot

def normalize_phoneme(x: pd.Series) -> pd.Series:
    """Normalise phoneme strings using NFC and strip whitespace.

    We use unicode normalization from Python's unicodedata.  No
    diacritic stripping is performed here because counts are based on
    exact segments.
    """
    import unicodedata
    return x.apply(lambda s: unicodedata.normalize('NFC', str(s)).strip())


def compute_inventory_sizes(phoible: pd.DataFrame) -> pd.DataFrame:
    """Compute total and vowel inventory sizes per (Glottocode, InventoryID).

    Returns a DataFrame with columns: Glottocode, InventoryID,
    total_inventory_size, vowel_inventory_size.
    """
    df = phoible.copy()
    df['Phoneme_norm'] = normalize_phoneme(df['Phoneme'])
    def agg_fn(group):
        total = group['Phoneme_norm'].nunique()
        vowels = group.loc[group['SegmentClass'] == 'vowel', 'Phoneme_norm'].nunique()
        return pd.Series({'total_inventory_size': total, 'vowel_inventory_size': vowels})
    sizes = df.groupby(['Glottocode', 'InventoryID']).apply(agg_fn).reset_index()
    return sizes


def select_largest_inventory(sizes: pd.DataFrame) -> pd.DataFrame:
    """Within each Glottocode, select the inventory with the largest total size.

    Ties are broken by the largest vowel inventory size, then the first.
    """
    sizes_sorted = sizes.sort_values(by=['Glottocode', 'total_inventory_size', 'vowel_inventory_size'], ascending=[True, False, False])
    largest = sizes_sorted.groupby('Glottocode').head(1).reset_index(drop=True)
    return largest


def compute_family_mapping(languoid: pd.DataFrame) -> pd.DataFrame:
    """Create a mapping from Glottocode to family name using the languoid table."""
    families = languoid[languoid['level'] == 'family'][['id', 'name']].rename(columns={'id': 'family_id', 'name': 'family_name'})
    langs = languoid[['id', 'family_id']]
    mapping = langs.merge(families, on='family_id', how='left')
    mapping['family_name'].fillna('(Unknown)', inplace=True)
    return mapping[['id', 'family_name']]


def compute_summary(language_data: pd.DataFrame) -> pd.DataFrame:
    """Compute per‑family counts, medians and IQRs of total inventory size."""
    summary = (language_data.groupby('family_name')['total_inventory_size']
               .agg(N='count', median_inventory='median',
                    q25=lambda x: np.percentile(x, 25),
                    q75=lambda x: np.percentile(x, 75))
               .reset_index())
    summary['iqr_inventory'] = summary['q75'] - summary['q25']
    summary.drop(columns=['q25', 'q75'], inplace=True)
    return summary


def make_ridgeline_plot(language_data: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Create and save the ridgeline density plot."""
    # Filter to families with at least MIN_N languages
    valid_fams = summary[summary['N'] >= MIN_N]['family_name']
    df = language_data[language_data['family_name'].isin(valid_fams)].copy()
    # Determine order by median inventory size (descending)
    fam_order = (df.groupby('family_name')['total_inventory_size']
                 .median().sort_values(ascending=False).index.tolist())
    # Global IQR across all languages
    q25 = np.percentile(language_data['total_inventory_size'], 25)
    q75 = np.percentile(language_data['total_inventory_size'], 75)
    print(f"[ridgelines] global IQR: {q25:.1f}–{q75:.1f}")
    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 6))
    # Colour map
    cmap = plt.get_cmap('viridis')
    # Determine x range
    x_min = language_data['total_inventory_size'].min() - 5
    x_max = language_data['total_inventory_size'].max() + 5
    xs = np.linspace(x_min, x_max, 512)
    # Plot each family
    for idx, fam in enumerate(fam_order):
        data = df[df['family_name'] == fam]['total_inventory_size']
        if len(data) < 2:
            continue
        kde = gaussian_kde(data)
        ys = kde(xs)
        ys = ys / ys.max()  # normalize density height
        offset = idx  # vertical offset for ridgeline
        ax.fill_between(xs, offset, ys + offset, color=cmap(idx / max(len(fam_order)-1, 1)), alpha=0.7)
        ax.text(x_min + 1, offset + 0.1, fam, va='bottom', ha='left', fontsize=9)
    # Draw global IQR band
    ax.axvspan(q25, q75, ymin=0, ymax=1, color='grey', alpha=0.1)
    # Formatting
    ax.set_xlabel('Total phoneme inventory size')
    ax.set_ylabel('Language family (ordered by median size)')
    ax.set_yticks([])
    ax.set_title('Phoneme inventory sizes by family')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1, len(fam_order) + 0.5)
    fig.tight_layout()
    # Save figures
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, 'inventory_ridgelines.pdf'))
    fig.savefig(os.path.join(FIG_DIR, 'inventory_ridgelines.png'), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, 'inventory_ridgelines.svg'))
    plt.close(fig)
    print('[ridgelines] plot saved to figs/inventory_ridgelines.{pdf,png,svg}')


def main() -> None:
    if not os.path.exists(PHOIBLE_FILE) or not os.path.exists(LANGUOID_FILE):
        raise FileNotFoundError('Required raw files missing; run 01_download_phoible.py first')
    # Read data
    print('[ridgelines] reading PHOIBLE …')
    phoible = pd.read_csv(PHOIBLE_FILE)
    print('[ridgelines] reading languoid …')
    languoid = pd.read_csv(LANGUOID_FILE)
    # Compute inventory sizes and select largest inventory per language
    print('[ridgelines] computing inventory sizes …')
    sizes = compute_inventory_sizes(phoible)
    largest = select_largest_inventory(sizes)
    # Merge language names
    language_meta = phoible[['Glottocode', 'LanguageName']].drop_duplicates()
    language_data = largest.merge(language_meta, on='Glottocode', how='left')
    # Compute family mapping
    mapping = compute_family_mapping(languoid)
    language_data = language_data.merge(mapping, left_on='Glottocode', right_on='id', how='left')
    language_data['family_name'] = language_data['family_name'].fillna('(Unknown)')
    # Compute summary statistics
    summary = compute_summary(language_data)
    # Write summary table
    os.makedirs(OUT_DIR, exist_ok=True)
    summary.to_csv(os.path.join(OUT_DIR, 'summary_tables.csv'), index=False)
    print(f"[ridgelines] summary written to {os.path.join(OUT_DIR, 'summary_tables.csv')}")
    # Make plot
    make_ridgeline_plot(language_data, summary)


if __name__ == '__main__':
    main()