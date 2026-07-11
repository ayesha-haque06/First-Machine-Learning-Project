# ============================================================
#  IPL 2023 Auction – Management Audit
#  Programming for AI – Course Project
#  Research Question 1: Can we predict if a player will be
#                        Sold or Unsold? (Classification)
#  Research Question 2: Can we predict the Cost in dollars
#                        for a sold player? (Regression)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import StandardScaler
from sklearn.linear_model       import LogisticRegression, LinearRegression
from sklearn.ensemble           import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics            import (confusion_matrix, classification_report,
                                        precision_score, recall_score,
                                        f1_score, accuracy_score,
                                        mean_absolute_error,
                                        mean_squared_error, r2_score)

# ──────────────────────────────────────────────────────────────
# STEP 1 – LOAD & EXPLORE
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 – LOAD & EXPLORE")
print("=" * 60)

df = pd.read_csv('IPL_Squad_2023_Auction_NEWDataset.csv')

print("\nFirst 5 rows:")
print(df.head())
print("\nDataset shape (rows, columns):", df.shape)
print("\nColumn data types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe(include='all'))

# ──────────────────────────────────────────────────────────────
# STEP 2 – CLEAN THE DATA
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 – CLEAN THE DATA")
print("=" * 60)

print("\nMissing values BEFORE cleaning:")
print(df.isnull().sum())

# Drop useless index column
df.drop(columns=['Unnamed: 0'], inplace=True)

# Rename columns to clean names
df.rename(columns={
    "Player's List"  : 'Player',
    'COST IN ₹ (CR.)': 'Cost_CR',
    'Cost IN $ (000)': 'Cost_USD',
    '2022 Squad'     : 'PrevTeam'
}, inplace=True)

# Flag retained players (Base Price = "Retained" text)
df['Is_Retained'] = (df['Base Price'] == 'Retained').astype(int)

# Convert Base Price to number (Retained → NaN → 0)
df['Base Price'] = pd.to_numeric(df['Base Price'], errors='coerce')
df['Base Price'] = df['Base Price'].fillna(0).astype(int)

# Fill missing costs with 0 (unsold/retained = no cost)
df['Cost_CR']  = df['Cost_CR'].fillna(0)
df['Cost_USD'] = df['Cost_USD'].fillna(0)

# Fill missing previous team with "New"
df['PrevTeam'] = df['PrevTeam'].fillna('New')

print("\nMissing values AFTER cleaning:")
print(df.isnull().sum())
print("\nFirst 5 rows after cleaning:")
print(df.head())

# ──────────────────────────────────────────────────────────────
# STEP 3 – ENCODE CATEGORICAL FEATURES (One-Hot Encoding)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 – ENCODE CATEGORICAL FEATURES")
print("=" * 60)

df = pd.get_dummies(df, columns=['TYPE'], drop_first=False)
type_cols = ['TYPE_ALL-ROUNDER','TYPE_BATSMAN',
             'TYPE_BOWLER','TYPE_WICKETKEEPER']

print("\nColumns after one-hot encoding TYPE:")
print(df.columns.tolist())

# Rebuild player type label for charts
df['PlayerType'] = df[type_cols].idxmax(axis=1).str.replace('TYPE_','')

# ──────────────────────────────────────────────────────────────
# STEP 4 – SEPARATE FEATURES & TARGETS
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 – SEPARATE FEATURES & TARGETS")
print("=" * 60)

# --- DATASET A: For Classification (Sold vs Unsold) ---
# Only auctioned players (not retained)
clf_df = df[df['Is_Retained'] == 0].copy()
clf_df['Sold'] = (clf_df['Team'] != 'Unsold').astype(int)

X_clf = clf_df[['Base Price'] + type_cols]
y_clf = clf_df['Sold']

print("\nClassification dataset (all auctioned players):")
print(f"  Total : {len(clf_df)}")
print(f"  Sold  : {y_clf.sum()}")
print(f"  Unsold: {(y_clf==0).sum()}")

# --- DATASET B: For Regression (predict Cost in USD) ---
# Only SOLD auctioned players (unsold have $0 - not useful)
reg_df = df[(df['Team'] != 'Unsold') & (df['Is_Retained'] == 0)].copy()

X_reg = reg_df[['Base Price'] + type_cols]
y_reg = reg_df['Cost_USD']

print("\nRegression dataset (sold auctioned players only):")
print(f"  Total players: {len(reg_df)}")
print(f"  Cost_USD range: ${y_reg.min():.0f}K – ${y_reg.max():.0f}K")
print(f"  Average cost : ${y_reg.mean():.0f}K")

# ──────────────────────────────────────────────────────────────
# STEP 5 – TRAIN / TEST SPLIT
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 – TRAIN / TEST SPLIT")
print("=" * 60)

# Classification split
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42)

# Regression split
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

print(f"\nClassification split:")
print(f"  Training : {len(X_clf_train)} players")
print(f"  Testing  : {len(X_clf_test)} players")
print(f"\nRegression split:")
print(f"  Training : {len(X_reg_train)} players")
print(f"  Testing  : {len(X_reg_test)} players")

# ──────────────────────────────────────────────────────────────
# STEP 6 – SCALE NUMERIC FEATURES
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 – SCALE NUMERIC FEATURES")
print("=" * 60)
print("\nIMPORTANT: fit on training data only, then transform both!")

# Classification scaler
scaler_clf = StandardScaler()
X_clf_train_s = scaler_clf.fit_transform(X_clf_train)
X_clf_test_s  = scaler_clf.transform(X_clf_test)

# Regression scaler
scaler_reg = StandardScaler()
X_reg_train_s = scaler_reg.fit_transform(X_reg_train)
X_reg_test_s  = scaler_reg.transform(X_reg_test)

print("\nScaling complete for both datasets.")

# ──────────────────────────────────────────────────────────────
# STEP 7 – TRAIN MODELS
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 – TRAIN MODELS")
print("=" * 60)

# --- Classification Models ---
print("\n[ Classification – Sold vs Unsold ]")
clf_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest (Clf)': RandomForestClassifier(
                               n_estimators=100, max_depth=5, random_state=42)
}
for name, model in clf_models.items():
    model.fit(X_clf_train_s, y_clf_train)
    acc = model.score(X_clf_test_s, y_clf_test)
    print(f"  {name}: Accuracy = {acc*100:.1f}%")

# --- Regression Models ---
print("\n[ Regression – Predict Cost in USD ]")
reg_models = {
    'Linear Regression'  : LinearRegression(),
    'Random Forest (Reg)': RandomForestRegressor(
                               n_estimators=100, max_depth=5, random_state=42)
}
for name, model in reg_models.items():
    model.fit(X_reg_train_s, y_reg_train)
    y_pred = model.predict(X_reg_test_s)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
    r2   = r2_score(y_reg_test, y_pred)
    print(f"  {name}: RMSE = ${rmse:.0f}K,  R² = {r2:.3f}")

# ──────────────────────────────────────────────────────────────
# STEP 8 – EVALUATE
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 – EVALUATE")
print("=" * 60)

# --- Classification Evaluation (Logistic Regression) ---
best_clf = clf_models['Logistic Regression']
y_clf_pred = best_clf.predict(X_clf_test_s)

cm   = confusion_matrix(y_clf_test, y_clf_pred)
acc  = accuracy_score(y_clf_test,  y_clf_pred)
prec = precision_score(y_clf_test, y_clf_pred, zero_division=0)
rec  = recall_score(y_clf_test,    y_clf_pred, zero_division=0)
f1   = f1_score(y_clf_test,        y_clf_pred, zero_division=0)

print("\n--- Classification Results (Logistic Regression) ---")
print(f"Confusion Matrix:\n{cm}")
print(f"\nAccuracy  : {acc*100:.1f}%")
print(f"Precision : {prec*100:.1f}%")
print(f"Recall    : {rec*100:.1f}%")
print(f"F1-Score  : {f1*100:.1f}%")
print("\nNote: Low recall is due to class imbalance (325 Unsold vs 80 Sold).")

# --- Regression Evaluation (Linear Regression) ---
best_reg = reg_models['Linear Regression']
y_reg_pred = best_reg.predict(X_reg_test_s)

mae  = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
r2   = r2_score(y_reg_test, y_reg_pred)

print("\n--- Regression Results (Linear Regression) ---")
print(f"MAE  : ${mae:.0f}K   (on average, predictions off by ${mae:.0f}K)")
print(f"RMSE : ${rmse:.0f}K   (penalises large errors more)")
print(f"R²   : {r2:.3f}   (model explains {r2*100:.1f}% of cost variation)")

# ──────────────────────────────────────────────────────────────
# STEP 9 – HYPERPARAMETER NOTE
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9 – HYPERPARAMETER TUNING")
print("=" * 60)
print("""
  Classification:
    Logistic Regression  → max_iter=1000
    Random Forest        → n_estimators=100, max_depth=5

  Regression:
    Linear Regression    → default (no hyperparameters needed)
    Random Forest        → n_estimators=100, max_depth=5
""")

# ──────────────────────────────────────────────────────────────
# STEP 10 – PREDICT ON NEW PLAYERS
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 10 – PREDICT ON NEW PLAYERS")
print("=" * 60)

# New player: All-Rounder, Base Price ₹2 Crore = $240K
new_player = np.array([[20000000, 1, 0, 0, 0]])

# Predict Sold/Unsold
new_clf_scaled = scaler_clf.transform(new_player)
clf_pred = best_clf.predict(new_clf_scaled)
print(f"\nNew Player: All-Rounder, Base Price ₹2 Crore")
print(f"  → Sold/Unsold prediction : {'SOLD ✓' if clf_pred[0]==1 else 'UNSOLD ✗'}")

# Predict Cost in USD (only meaningful if player is sold)
new_reg_scaled = scaler_reg.transform(new_player)
reg_pred = best_reg.predict(new_reg_scaled)
print(f"  → Predicted Cost in USD  : ${reg_pred[0]:.0f}K")

# ══════════════════════════════════════════════════════════════
#  VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

# ── CHART 1 – Team Spending ───────────────────────────────────
team_spend = (df[df['Team'] != 'Unsold']
              .groupby('Team')['Cost_CR']
              .sum()
              .sort_values(ascending=False))

fig1, ax1 = plt.subplots(figsize=(12, 6))
bars = ax1.bar(team_spend.index, team_spend.values,
               color='royalblue', edgecolor='black')
ax1.set_title('Total Auction Spend by Team (₹ Crore)',
              fontsize=15, fontweight='bold')
ax1.set_xlabel('Team'); ax1.set_ylabel('Total Spend (₹ Crore)')
ax1.tick_params(axis='x', rotation=45)
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             f'{bar.get_height():.1f}',
             ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('chart1_team_spend.png', dpi=150)
plt.show()
print("\nChart 1 saved.")
print("  → This suggests SRH spent 7× more than KKR. Therefore,")
print("    we should check if high spending built better squads.")

# ── CHART 2 – Sold vs Unsold by Type ─────────────────────────
auction_only = df[df['Is_Retained'] == 0].copy()
auction_only['Status'] = np.where(
    auction_only['Team'] == 'Unsold', 'Unsold', 'Sold')
status_counts = (auction_only
                 .groupby(['PlayerType','Status'])
                 .size().unstack(fill_value=0))

fig2, ax2 = plt.subplots(figsize=(10, 6))
x = np.arange(len(status_counts.index))
width = 0.35
ax2.bar(x - width/2, status_counts.get('Sold',   0), width,
        label='Sold',   color='#27ae60', edgecolor='black')
ax2.bar(x + width/2, status_counts.get('Unsold', 0), width,
        label='Unsold', color='#e74c3c', edgecolor='black')
ax2.set_title('Sold vs Unsold Players by Type',
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Player Type'); ax2.set_ylabel('Number of Players')
ax2.set_xticks(x); ax2.set_xticklabels(status_counts.index)
ax2.legend()
plt.tight_layout()
plt.savefig('chart2_sold_unsold.png', dpi=150)
plt.show()
print("\nChart 2 saved.")
print("  → All-Rounders most unsold. Therefore, teams should")
print("    target Wicketkeepers for better auction success.")

# ── CHART 3 – Predicted vs Actual Cost in USD ────────────────
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.scatter(y_reg_test, y_reg_pred,
            color='royalblue', edgecolor='black',
            alpha=0.7, s=80, label='Players')
# Perfect prediction line
max_val = max(y_reg_test.max(), max(y_reg_pred))
ax3.plot([0, max_val], [0, max_val],
         color='red', linestyle='--',
         linewidth=2, label='Perfect Prediction')
ax3.set_title('Regression: Predicted vs Actual Cost (USD $000)',
              fontsize=14, fontweight='bold')
ax3.set_xlabel('Actual Cost ($000)')
ax3.set_ylabel('Predicted Cost ($000)')
ax3.legend()
ax3.text(0.05, 0.92,
         f'R² = {r2:.3f}\nRMSE = ${rmse:.0f}K\nMAE  = ${mae:.0f}K',
         transform=ax3.transAxes, fontsize=11,
         bbox=dict(boxstyle='round', facecolor='lightyellow',
                   edgecolor='orange'))
plt.tight_layout()
plt.savefig('chart3_regression.png', dpi=150)
plt.show()
print("\nChart 3 saved.")
print(f"  → Model explains {r2*100:.1f}% of cost variation (R²={r2:.3f}).")
print("    Therefore, Base Price + Type give a partial signal.")

# ── CHART 4 – Confusion Matrix ───────────────────────────────
fig4, ax4 = plt.subplots(figsize=(6, 5))
im = ax4.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax4)
ax4.set_title('Confusion Matrix – Logistic Regression',
              fontsize=13, fontweight='bold')
ax4.set_xlabel('Predicted Label')
ax4.set_ylabel('True Label')
ax4.set_xticks([0,1]); ax4.set_xticklabels(['Unsold','Sold'])
ax4.set_yticks([0,1]); ax4.set_yticklabels(['Unsold','Sold'])
thresh = cm.max() / 2
for i in range(2):
    for j in range(2):
        ax4.text(j, i, str(cm[i,j]),
                 ha='center', va='center', fontsize=16,
                 color='white' if cm[i,j] > thresh else 'black')
plt.tight_layout()
plt.savefig('chart4_confusion.png', dpi=150)
plt.show()
print("\nChart 4 saved.")

print("\n" + "=" * 60)
print("ALL STEPS COMPLETE!")
print("=" * 60)
