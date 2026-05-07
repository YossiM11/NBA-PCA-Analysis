import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tabulate import tabulate

df = pd.read_csv('nba_cleaned.csv')

features = [
    'avg_minutes', 'points', 'assists', 'blocks', 'steals', 'turnovers',
    'reboundsTotal', 'foulsPersonal', 'fieldGoalsPercentage',
    'threePointersAttempted', 'threePointersPercentage',
    'freeThrowsAttempted', 'freeThrowsPercentage'
]

df_clean = df[features + ['firstName', 'lastName', 'year']].dropna().reset_index(drop=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features])

pca = PCA(n_components=3)
components = pca.fit_transform(X_scaled)

# Uncomment to print the variance explained by each principal component and the total.
# print(f"PC1 variance explained: {pca.explained_variance_ratio_[0]:.1%}")
# print(f"PC2 variance explained: {pca.explained_variance_ratio_[1]:.1%}")
# print(f"PC3 variance explained: {pca.explained_variance_ratio_[2]:.1%}")
# print(f"Total: {sum(pca.explained_variance_ratio_):.1%}")

loadings = pd.DataFrame(
    pca.components_.T,
    index=features,
    columns=['PC1', 'PC2', 'PC3']
).round(3)

# Uncomment to print the full loadings table sorted by PC1, showing how each
# statistical feature contributes to each principal component.
# print("\nLoadings:")
# print(loadings.sort_values('PC1'))

# Uncomment to print the PC3 loadings sorted from most negative to most positive,
# useful for interpreting the defensive disruption vs. offensive creation axis.
# print("\nPC3 loadings sorted:")
# print(loadings[['PC3']].sort_values('PC3'))

df_pca = df_clean[['firstName', 'lastName', 'year']].copy()
df_pca['PC1'] = components[:, 0]
df_pca['PC2'] = components[:, 1]
df_pca['PC3'] = components[:, 2]
df_pca['player'] = df_pca['firstName'] + ' ' + df_pca['lastName']

# Deduplicate — keep the best PC1 season per player per year
df_pca = df_pca.sort_values('PC1', ascending=False).drop_duplicates(
    subset=['player', 'year']
).sort_values(['player', 'year']).reset_index(drop=True)

# Uncomment to print sanity checks: year range, total player-seasons after
# deduplication, any remaining duplicate player-seasons, and LeBron James's
# PC1 scores by year to verify the pipeline is working correctly.
# print("--- SANITY CHECKS ---")
# print(f"Years in dataset: {df_pca['year'].min():.0f} — {df_pca['year'].max():.0f}")
# print(f"Total player-seasons after dedup: {len(df_pca)}")
# dupes = df_pca.groupby(['player', 'year']).size().reset_index(name='count')
# print(f"Duplicate player-seasons remaining: {dupes[dupes['count'] > 1].shape[0]}")
# print("\nLeBron James seasons:")
# print(df_pca[df_pca['player'] == 'LeBron James'][['year', 'PC1']].sort_values('year').to_string(index=False))

def get_top_unique(df, col, largest=True, n=10):
    sorted_df = df.sort_values(col, ascending=not largest)
    seen = set()
    rows = []
    for _, row in sorted_df.iterrows():
        name = row['player']
        if name not in seen:
            seen.add(name)
            rows.append(row)
        if len(rows) == n:
            break
    return pd.DataFrame(rows)

def print_table(label, subset, cols=None, headers=None):
    if cols is None:
        cols = ['player', 'year', 'PC1', 'PC2']
    if headers is None:
        headers = ['Player', 'Year', 'PC1', 'PC2']
    subset = subset[cols].copy()
    subset = subset.round({c: 2 for c in cols if c not in ['player', 'year']})
    if 'year' in cols:
        subset['year'] = subset['year'].astype(int)
    print(f"\n{'='*50}")
    print(f"  {label}")
    print('='*50)
    print(tabulate(subset, headers=headers, tablefmt='simple', showindex=False, floatfmt='.2f'))

# Uncomment to print the top and bottom 10 unique-player seasons by PC1 (usage/production axis)
# and PC2 (positional axis). High PC1 = high-volume stars, low PC1 = bench players.
# High PC2 = quintessential point guards, low PC2 = quintessential big-men.
# for label, col, largest in [
#     ('HIGH PC1 — Volume Stars',    'PC1', True),
#     ('LOW PC1  — Bench/Low Usage', 'PC1', False),
#     ('HIGH PC2 — Most Guard-like', 'PC2', True),
#     ('LOW PC2  — Most Big-like',   'PC2', False),
# ]:
#     print_table(label, get_top_unique(df_pca, col, largest))

# Uncomment to print the top 10 unique-player versatile seasons: high PC1 players
# with PC2 between -1.5 and 1.5, indicating productive players who don't skew
# strongly toward either a guard or big-man profile.
# versatile = df_pca[
#     (df_pca['PC1'] > 0) &
#     (df_pca['PC2'].between(-1.5, 1.5))
# ].sort_values('PC1', ascending=False)
# print_table('MOST VERSATILE PLAYERS (High PC1, |PC2| < 1.5)',
#             get_top_unique(versatile.reset_index(drop=True), 'PC1', n=10))

# Uncomment to print the top and bottom 5 unique-player seasons by PC3.
# High PC3 = defensive rim protectors (blocks, steals); low PC3 = offensive
# creators defined by free throw attempts, assists, and points.
# print_table(
#     'HIGH PC3 — Defensive Disruptors (blocks, steals, rim protection)',
#     get_top_unique(df_pca, 'PC3', largest=True, n=5),
#     cols=['player', 'year', 'PC1', 'PC2', 'PC3'],
#     headers=['Player', 'Year', 'PC1', 'PC2', 'PC3']
# )
# print_table(
#     'LOW PC3 — Scoring Playmakers (FT%, assists, points, FT attempts)',
#     get_top_unique(df_pca, 'PC3', largest=False, n=5),
#     cols=['player', 'year', 'PC1', 'PC2', 'PC3'],
#     headers=['Player', 'Year', 'PC1', 'PC2', 'PC3']
# )

# Career GOAT metric
career = (
    df_pca.groupby('player')
    .agg(
        career_PC1=('PC1', 'sum'),
        seasons=('PC1', 'count'),
        avg_PC1=('PC1', 'mean'),
        avg_PC2=('PC2', 'mean'),
        consistency=('PC1', 'std'),
    )
    .reset_index()
    .round(3)
)

rings = {
    'LeBron James':        4,
    'Karl Malone':         0,
    'Michael Jordan':      6,
    'Kevin Durant':        2,
    "Shaquille O'Neal":    4,
    'James Harden':        0,
    'Hakeem Olajuwon':     2,
    'Kobe Bryant':         5,
    'Kareem Abdul-Jabbar': 6,
    'Charles Barkley':     0,
    'Russell Westbrook':   0,
    'Wilt Chamberlain':    2,
    'Allen Iverson':       0,
    'Magic Johnson':       5,
    'David Robinson':      2,
    'Patrick Ewing':       0,
    'Tim Duncan':          5,
    'Moses Malone':        1,
    'Chris Paul':          0,
    'Dwyane Wade':         3,
}

career['rings'] = career['player'].map(rings).fillna(0).astype(int)
career['versatility_bonus'] = 1 + 0.05 * (1 - career['avg_PC2'].abs() / career['avg_PC2'].abs().max())
career['consistency_bonus'] = 1 + 0.1  * (1 - career['consistency'] / career['consistency'].max())
career['rings_bonus']       = 1 + 0.05 * career['rings']
career['pc1_sharpe']        = (career['avg_PC1'] / career['consistency']).round(3)
career['goat_score'] = (
    career['career_PC1'] *
    career['versatility_bonus'] *
    career['consistency_bonus'] *
    career['rings_bonus']
).round(2)
career = career.sort_values('goat_score', ascending=False).reset_index(drop=True)

# Uncomment to print the top 10 players by GOAT score, including career PC1,
# PC1 Sharpe ratio, average PC1, season-to-season consistency, and average PC2.
# print(f"\n{'='*90}")
# print("  CAREER GOAT SCORE (Career PC1 × Versatility × Consistency × Rings)")
# print('='*90)
# print(tabulate(
#     career[['player', 'seasons', 'rings', 'goat_score', 'career_PC1', 'pc1_sharpe', 'avg_PC1', 'consistency', 'avg_PC2']].head(10),
#     headers=['Player', 'Seasons', 'Rings', 'GOAT Score', 'Career PC1', 'PC1 Sharpe', 'Avg PC1', 'Consistency', 'Avg PC2'],
#     tablefmt='simple', showindex=False, floatfmt='.2f'
# ))

# Uncomment to generate and save a 2D density heatmap of all player-seasons
# projected onto PC1 (x-axis) and PC2 (y-axis), saved as 'pca_heatmap.png'.
# fig, ax = plt.subplots(figsize=(9, 7))
# h = ax.hist2d(
#     components[:, 0], components[:, 1],
#     bins=60,
#     cmap='YlOrRd',
#     density=True
# )
# plt.colorbar(h[3], ax=ax, label='Density')
# ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
# ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
# ax.set_title('NBA player-seasons projected onto PC1 and PC2')
# plt.tight_layout()
# plt.savefig('pca_heatmap.png', dpi=150)
# plt.show()