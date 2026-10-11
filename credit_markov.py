from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Credit rating states in the original S&P table
from_states = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C"]
to_states = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C", "D", "NR"]

# Original S&P one-year transition data (1981–2025), in percentages
raw_pct = np.array([
    [87.28, 8.91, 0.51, 0.03, 0.10, 0.03, 0.05, 0.00, 3.09],
    [0.44, 87.81, 7.45, 0.43, 0.05, 0.06, 0.02, 0.02, 3.73],
    [0.02, 1.52, 89.57, 4.52, 0.22, 0.09, 0.01, 0.04, 3.99],
    [0.00, 0.07, 3.07, 87.38, 3.12, 0.38, 0.09, 0.13, 5.75],
    [0.01, 0.02, 0.10, 4.37, 78.89, 6.13, 0.49, 0.55, 9.44],
    [0.00, 0.02, 0.05, 0.14, 4.43, 75.15, 4.76, 2.87, 12.59],
    [0.00, 0.00, 0.07, 0.12, 0.38, 13.08, 45.00, 26.12, 15.23]
])

# Remove NR and renormalise each row conditional on not entering NR
raw_P = raw_pct / 100
P_no_nr = raw_P[:, :-1]
row_sums = P_no_nr.sum(axis=1)
if np.any(row_sums <= 0):
    raise ValueError("Cannot renormalise a row with zero rated/default probability.")
P_rated = P_no_nr / row_sums[:, np.newaxis]

# Add default (D) as an absorbing state
states = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C", "D"]
default_row = np.array([[0, 0, 0, 0, 0, 0, 0, 1]])
P = np.vstack([P_rated, default_row])

if P.shape != (8, 8) or not np.all(np.isfinite(P)) or np.any(P < 0):
    raise ValueError("Transition matrix has invalid dimensions or probabilities.")
if not np.allclose(P.sum(axis=1), 1) or not np.allclose(P[7], default_row[0]):
    raise ValueError("Transition matrix must be stochastic with absorbing default.")

# Save figures next to this script, regardless of the working directory
figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# Default probabilities over years 0–10
starting_ratings = {"AAA": 0, "A": 2, "BBB": 3, "BB": 4}
years = range(0, 11)
default_probs = {}

for rating, i in starting_ratings.items():
    probs = []
    for n in years:
        Pn = np.linalg.matrix_power(P, n)
        default_prob = Pn[i, 7]
        probs.append(default_prob)
    default_probs[rating] = probs

fig, ax = plt.subplots(figsize=(8, 5))
for rating, probs in default_probs.items():
    ax.plot(years, probs, marker="o", label=rating)
ax.set_xlabel("Years")
ax.set_ylabel("Cumulative default probability")
ax.set_title("Model-Implied Default Probability by Initial Rating")
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_xticks(list(years))
ax.legend(title="Initial rating")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(figures_dir / "default_probability_by_rating.png", dpi=300, bbox_inches="tight")

# Absorbing Markov-chain analysis for all seven non-default ratings
Q = P[:7, :7]
R = P[:7, 7:]
I = np.eye(7)
N = np.linalg.solve(I - Q, I)
t = N @ np.ones(7)
transient_states = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C"]

absorption_probs = N @ R
m2 = (2 * N - I) @ t
variance_t = m2 - t**2
tol = 1e-10 * max(1.0, np.max(t**2))
if np.any(variance_t < -tol):
    raise ValueError("Negative variance larger than numerical roundoff.")
variance_t = np.maximum(variance_t, 0)
std_t = np.sqrt(variance_t)

# Model-implied median time to default: the first year cumulative default reaches 50%
search_horizon = 200
median_years = np.full(7, -1, dtype=int)
for k in range(1, search_horizon + 1):
    survival = np.linalg.matrix_power(Q, k) @ np.ones(7)
    cumulative_default = 1 - survival
    new_medians = (median_years == -1) & (cumulative_default >= 0.5)
    median_years[new_medians] = k
    if np.all(median_years != -1):
        break

print("\nModel-implied time to default (years):")
print(f"{'Rating':<7} {'Mean':>10} {'Std dev':>10} {'Median':>10} {'Eventual D':>13}")
for i, state in enumerate(transient_states):
    median_year = median_years[i]
    median_label = str(median_year) if median_year != -1 else f">{search_horizon}"
    print(f"{state:<7} {t[i]:>10.2f} {std_t[i]:>10.2f} {median_label:>10} {absorption_probs[i, 0]:>12.4%}")

# Five- and ten-year survival, first-default probabilities and conditional hazards
# Hazard in year k = P(first default in year k | survival through year k-1)
print("\nDefault-time analysis (all initial ratings):")
print(f"{'Rating':<7} {'Year':>4} {'Survival':>11} {'Default by':>12} {'First default':>14} {'Hazard':>11}")
for k in [5, 10]:
    Qk = np.linalg.matrix_power(Q, k)
    Qk_minus_1 = np.linalg.matrix_power(Q, k - 1)
    survival = Qk @ np.ones(7)
    cumulative_default = 1 - survival
    first_default = (Qk_minus_1 @ R).ravel()
    survival_previous = Qk_minus_1 @ np.ones(7)
    hazard = np.divide(first_default, survival_previous, out=np.full(7, np.nan), where=survival_previous > 0)
    for i, state in enumerate(transient_states):
        print(f"{state:<7} {k:>4} {survival[i]:>10.4%} {cumulative_default[i]:>11.4%} {first_default[i]:>13.4%} {hazard[i]:>10.4%}")

# Conditional rating distributions after 5 and 10 years
fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

for ax, k in zip(axes, [5, 10]):
    transient_row = np.linalg.matrix_power(Q, k)
    survival = transient_row.sum(axis=1)

    if np.any(survival <= 0):
        raise ValueError("Cannot condition on zero survival probability.")

    conditional_ratings = transient_row / survival[:, np.newaxis]

    im = ax.imshow(conditional_ratings * 100, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(transient_states)), labels=transient_states)
    ax.set_yticks(range(len(transient_states)), labels=transient_states)
    ax.set_xlabel(f"Rating after {k} years")
    ax.set_ylabel("Initial rating")
    ax.set_title(f"{k}-Year Rating Distribution Conditional on Survival")

    for i in range(len(transient_states)):
        for j in range(len(transient_states)):
            value = conditional_ratings[i, j] * 100
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color="white" if value > 50 else "black", fontsize=8)

fig.colorbar(im, ax=axes, label="Conditional probability (%)", shrink=0.85)
fig.savefig(figures_dir / "conditional_rating_distribution_5_10_years.png", dpi=300, bbox_inches="tight")

# Hypothetical stress: increase downgrade/default weights by 20%, then renormalise
stress_factor = 1.20
P_stressed = P.copy()
for i in range(7):
    row = P_stressed[i].copy()
    row[i + 1:] *= stress_factor
    row = row / row.sum()
    P_stressed[i] = row
P_stressed[7] = P[7]

if not np.all(np.isfinite(P_stressed)) or np.any(P_stressed < 0):
    raise ValueError("Stressed transition matrix contains invalid probabilities.")
if not np.allclose(P_stressed.sum(axis=1), 1) or not np.allclose(P_stressed[7], default_row[0]):
    raise ValueError("Stressed transition matrix must be stochastic with absorbing default.")

# Default risk: baseline vs stressed (BBB and BB)
fig, ax = plt.subplots(figsize=(8, 5))
for rating in ["BBB", "BB"]:
    i = starting_ratings[rating]
    baseline = [np.linalg.matrix_power(P, n)[i, 7] for n in years]
    stressed = [np.linalg.matrix_power(P_stressed, n)[i, 7] for n in years]
    line, = ax.plot(years, baseline, label=f"{rating} Baseline")
    ax.plot(years, stressed, "--", color=line.get_color(), label=f"{rating} Stressed (×{stress_factor:.2f})")

ax.set_xlabel("Years")
ax.set_ylabel("Cumulative default probability")
ax.set_title(f"Baseline vs Stressed (×{stress_factor:.2f}) Default Risk")
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_xticks(list(years))
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(figures_dir / "baseline_vs_stressed_default.png", dpi=300, bbox_inches="tight")

# Transition heatmaps for years 1, 3, 5 and 10 (all non-default starting ratings)
ratings = states[:-1]
P5 = np.linalg.matrix_power(P, 5)
fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
for ax, n in zip(axes.flat, [1, 3, 5, 10]):
    Pn = P5 if n == 5 else np.linalg.matrix_power(P, n)
    heatmap = Pn[:7, :] * 100
    im = ax.imshow(heatmap, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(states)), labels=states)
    ax.set_yticks(range(len(ratings)), labels=ratings)
    ax.set_xlabel("Rating at the horizon")
    ax.set_ylabel("Initial rating")
    ax.set_title(f"{n}-Year Credit Rating Transition Probabilities")

    for i in range(len(ratings)):
        for j in range(len(states)):
            value = heatmap[i, j]
            label = f"{value:.1f}%" if value >= 0.05 else ("<0.1%" if value > 0 else "0%")
            ax.text(j, i, label, ha="center", va="center", color="white" if value > 50 else "black", fontsize=7)

fig.colorbar(im, ax=axes.ravel().tolist(), label="Probability (%)", shrink=0.85, pad=0.02)
fig.suptitle("Model-Implied Credit Rating Transitions", fontsize=15)
fig.savefig(figures_dir / "transition_heatmaps_1_3_5_10_years.png", dpi=300, bbox_inches="tight")

# Temporary stress: 2 stressed years, then normal transition probabilities
stress_years = 2
print(f"\nScenario comparison: temporary stress ({stress_years} years, ×{stress_factor:.2f}) vs persistent stress (×{stress_factor:.2f})")
for n in [5, 10]:
    Pn = np.linalg.matrix_power(P, n)
    P_temporary = np.linalg.matrix_power(P_stressed, stress_years) @ np.linalg.matrix_power(P, n - stress_years)
    P_stressed_n = np.linalg.matrix_power(P_stressed, n)
    if not np.allclose(P_temporary.sum(axis=1), 1) or np.any(P_temporary < -1e-12):
        raise ValueError("Temporary-stress transition matrix is invalid.")

    print(f"\nYear {n} cumulative default probabilities:")
    print(f"{'Rating':<7} {'Baseline':>11} {'Temporary':>11} {'Persistent':>11} {'Temp Δ pp':>11} {'Pers Δ pp':>11}")
    for i, rating in enumerate(transient_states):
        baseline = Pn[i, 7]
        temporary = P_temporary[i, 7]
        stressed = P_stressed_n[i, 7]
        print(f"{rating:<7} {baseline:>11.4%} {temporary:>11.4%} {stressed:>11.4%} {(temporary - baseline)*100:>+11.4f} {(stressed - baseline)*100:>+11.4f}")

# GBP valuation inputs: fixed published snapshot, no internet required to rerun
market_date = "2026-09-30"
spreads = {"Investment grade": 85.72 / 10000, "High yield": 464.95 / 10000}
gilt_rates = {5: 4.922 / 100, 10: 5.371 / 100}
face_value = 100.0
coupon_rate = 0.05
maturity = 15
recovery_rate = 0.40

# Model-derived rating-specific spread estimates; not seven observed market spreads.
# S&P 10-year model-implied default intensities provide the within-bucket risk ordering.
# The square root smooths the extreme risk differences; each bucket's unweighted
# average estimated spread equals its observed MSCI sterling benchmark spread.
spread_power = 0.5
F10 = np.linalg.matrix_power(P, 10)[:7, 7]
risk_scores = (-np.log1p(-F10) / 10)**spread_power
rating_spreads = np.empty(7)
for indices, bucket in [(slice(0, 4), "Investment grade"), (slice(4, 7), "High yield")]:
    rating_spreads[indices] = spreads[bucket] * risk_scores[indices] / risk_scores[indices].mean()

if np.any(np.diff(rating_spreads) < 0) or not np.all(np.isfinite(rating_spreads)):
    raise ValueError("Rating-specific spread estimates are not nondecreasing and finite.")

print(f"\nGBP bond-value scenario analysis ({market_date} market snapshot):")
print(f"Published MSCI GBP index OAS: IG = {spreads['Investment grade']:.3%}, HY = {spreads['High yield']:.3%}")
print("Model-estimated rating-specific GBP spread proxies (not observed rating-level spreads):")
print(", ".join(f"{rating} = {rating_spreads[i]:.3%}" for i, rating in enumerate(transient_states)))

for n in [5, 10]:
    remaining = maturity - n
    yields = gilt_rates[remaining] + rating_spreads
    coupon = face_value * coupon_rate
    payment_years = np.arange(1, remaining + 1)
    V = np.array([np.sum(coupon / (1 + y)**payment_years) + face_value / (1 + y)**remaining for y in yields] + [face_value * recovery_rate])

    Pn = np.linalg.matrix_power(P, n)
    P_temporary = np.linalg.matrix_power(P_stressed, stress_years) @ np.linalg.matrix_power(P, n - stress_years)
    P_stressed_n = np.linalg.matrix_power(P_stressed, n)
    bond_baseline = Pn @ V
    bond_temporary = P_temporary @ V
    bond_stressed = P_stressed_n @ V

    print(f"\nYear {n} expected terminal bond values (£ per £100; {remaining} years remaining):")
    print(f"{'Rating':<7} {'Base £':>12} {'Temp £':>12} {'Stress £':>12}")
    for i, rating in enumerate(transient_states):
        print(f"{rating:<7} {bond_baseline[i]:>12.2f} {bond_temporary[i]:>12.2f} {bond_stressed[i]:>12.2f}")

print("\nSee README.md for data sources, spread calibration, bond assumptions and limitations.")
print(f"\nSaved figures in: {figures_dir}")
plt.show()
