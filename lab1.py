import numpy as np
import matplotlib.pyplot

data = np.loadtxt("Filippenko & Riess 1998.dat", skiprows=7)

distance = data[:, 0]
distance_err = data[:, 1]
velocity = data[:, 2]


# PLOT SETTINGS ------------------------------------------------------------------------------

save_path = r"C:\Users\jessi\Downloads"

# Color palette
TEAL = "#216869"
LIGHT_TEAL = "#A9D6D2"
DARK_TEAL = "#164A4B"
ORANGE = "#D97745"
WHITE = "#FFFFFF"
INK = "#263A38"
MUTED = "#607773"
YELLOW = "#E5B83D"
AX_BG = "#EAF7F5"

matplotlib.pyplot.rcParams.update({
    # Background
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,

    # Fonts
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Candara",
        "Corbel",
        "Trebuchet MS",
        "DejaVu Sans"
    ],
    "font.size": 13,
    "font.weight": "bold",
    "text.color": INK,

    "axes.labelcolor": INK,
    "axes.titlesize": 23,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "axes.labelweight": "bold",

    # Titles and labels
    "axes.labelcolor": INK,
    "axes.titlesize": 23,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "axes.labelweight": "bold",

    # Axes
    "axes.edgecolor": INK,
    "axes.linewidth": 1.3,
    "axes.spines.top": False,
    "axes.spines.right": False,

    # Ticks
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    # Grid
    "axes.grid": False,

    # Legend
    "legend.frameon": False,
    "legend.fontsize": 12,

    # Export
    "figure.dpi": 120,
    "savefig.dpi": 300,
})


# USING POLYFIT ------------------------------------------------------------------------------

coeff, cov = np.polyfit(velocity, distance, 1, cov=True)

slope, intercept = coeff

slope_err = np.sqrt(cov[0, 0])
intercept_err = np.sqrt(cov[1, 1])

print("NumPy polyfit:")
print("Slope:", slope, "+/-", slope_err)
print("Intercept:", intercept, "+/-", intercept_err)

x_fit = np.linspace(min(velocity), max(velocity), 100)
y_fit = slope * x_fit + intercept


# PLOT ONE: DISTANCE VS VELOCITY --------------------------------------------------------------

# Weighted linear regression
weighted_coeff, cov = np.polyfit(
    velocity,
    distance,
    1,
    w=1 / distance_err,
    cov="unscaled"
)

weighted_slope = weighted_coeff[0]
weighted_intercept = weighted_coeff[1]

weighted_y_fit = weighted_slope * x_fit + weighted_intercept

fig, ax = matplotlib.pyplot.subplots(figsize=(10, 6.5))
ax.set_facecolor(AX_BG)

ax.errorbar(
    velocity, distance,
    yerr=distance_err,
    fmt="o",
    color=TEAL,
    ecolor=TEAL,
    alpha=0.8,
    markersize=7.5,
    markerfacecolor=LIGHT_TEAL,
    markeredgecolor=TEAL,
    markeredgewidth=1.3,
    elinewidth=1,
    capsize=2,
    label="observed measurements",
    zorder=3
)

# ax.plot(
#     x_fit, y_fit,
#     color=ORANGE,
#     linewidth=2,
#     linestyle=(0, (3, 3)),
#     alpha=0.65,
#     label="linear regression",
#     zorder=2
# )

# Unweighted regression
ax.plot(
    x_fit, y_fit,
    color=ORANGE,
    linewidth=2,
    linestyle="--",
    alpha=0.7,
    label="unweighted regression",
    zorder=2
)

# Weighted regression
ax.plot(
    x_fit, weighted_y_fit,
    color=YELLOW,
    linewidth=2.2,
    linestyle="--",
    alpha=0.8,
    label="weighted regression",
    zorder=2
)

ax.set_xlabel("velocity (km/s)", labelpad=15)
ax.set_ylabel("distance (Mpc)", labelpad=15)

ax.set_title(
    "distance vs. velocity",
    loc="left",
    pad=25,
    color=DARK_TEAL
)

ax.grid(axis="y", color=MUTED, alpha=0.18, linewidth=0.8)
ax.set_axisbelow(True)

ax.legend(
    loc="upper left",
    prop={"size": 12, "weight": "bold"}
)

fig.tight_layout(pad=2)
fig.savefig(save_path + r"\distance_vs_velocity.png", bbox_inches="tight")


# PLOT TWO: RESIDUALS ------------------------------------------------------------------------------

residuals = distance - (slope * velocity + intercept)

fig, ax = matplotlib.pyplot.subplots(figsize=(10, 5))
ax.set_facecolor(AX_BG)

ax.errorbar(
    velocity, residuals,
    yerr=distance_err,
    fmt="o",
    color=TEAL,
    ecolor=TEAL,
    markersize=7.5,
    markerfacecolor=LIGHT_TEAL,
    markeredgecolor=TEAL,
    markeredgewidth=1.3,
    elinewidth=1,
    capsize=2,
    alpha=0.85,
    zorder=3
)

ax.axhline(
    0,
    color=ORANGE,
    linewidth=2.5,
    linestyle=(0, (6, 4)),
    label="zero residual"
)

ax.set_xlabel("velocity (km/s)", labelpad=15)
ax.set_ylabel("residuals (Mpc)", labelpad=15)

ax.set_title(
    "regression residuals",
    loc="left",
    pad=25,
    color=DARK_TEAL
)

ax.grid(axis="y", color=MUTED, alpha=0.18, linewidth=0.8)
ax.set_axisbelow(True)

ax.legend(
    loc="upper right",
    prop={"size": 12, "weight": "bold"}
)

fig.tight_layout(pad=2)
fig.savefig(save_path + r"\residuals.png", bbox_inches="tight")


# COMPARE OUR VERSION WITH POLYFIT -------------------------------------------------------------

x = velocity
y = distance

X = np.column_stack((x, np.ones(len(x))))

beta = np.linalg.inv(X.T @ X) @ X.T @ y

manual_slope = beta[0]
manual_intercept = beta[1]

print("\nManual normal equation:")
print("Slope:", manual_slope)
print("Intercept:", manual_intercept)

print("\nResults match:")
print(np.allclose(
    [slope, intercept],
    [manual_slope, manual_intercept]
))


# WEIGHTED LINEAR FIT --------------------------------------------------------------------------

weighted_coeff, cov = np.polyfit(
    velocity,
    distance,
    1,
    w=1 / distance_err,
    cov="unscaled"
)

weighted_slope = weighted_coeff[0]
weighted_intercept = weighted_coeff[1]


# CALCULATE UNCERTAINTIES ----------------------------------------------------------------------

slope_err = np.sqrt(cov[0, 0])
intercept_err = np.sqrt(cov[1, 1])

print("\nWeighted fit with uncertainties:")
print("Slope:", weighted_slope, "+/-", slope_err)
print("Intercept:", weighted_intercept, "+/-", intercept_err)

print("\nCovariance matrix:")
print(cov)


# CALCULATE APPROXIMATE AGE OF THE UNIVERSE ----------------------------------------------------

mpc_to_km = 3.08568e19
seconds_per_year = 31557600

age = weighted_slope * mpc_to_km / seconds_per_year
age_err = slope_err * mpc_to_km / seconds_per_year

age /= 1e9
age_err /= 1e9

print("\nEstimated Hubble age:")
print("Age:", age, "+/-", age_err, "billion years")


# MONTE CARLO SIMULATION (1000 TRIALS) ---------------------------------------------------------

slopes = []

for i in range(1000):
    new_distance = distance + np.random.normal(0, distance_err)

    new_slope, _ = np.polyfit(
        velocity,
        new_distance,
        1,
        w=1 / distance_err
    )

    slopes.append(new_slope)

print("\nMonte Carlo slope:", np.mean(slopes))
print("Monte Carlo uncertainty:", np.std(slopes, ddof=1))
print("Analytical uncertainty:", slope_err)


# PLOT THREE: MONTE CARLO ----------------------------------------------------------------------

fig, ax = matplotlib.pyplot.subplots(figsize=(10, 6.5))
ax.set_facecolor(AX_BG)

ax.hist(
    slopes,
    bins=30,
    color=TEAL,
    edgecolor=WHITE,
    linewidth=1.2,
    alpha=0.9
)

ax.axvline(
    weighted_slope,
    color=ORANGE,
    linewidth=3,
    linestyle=(0, (5, 3)),
    label="original weighted slope",
    zorder=4
)

ax.set_xlabel("best-fitting slope (Mpc / km/s)", labelpad=15)
ax.set_ylabel("frequency", labelpad=15)

ax.set_title(
    "distribution of best-fitting gradients",
    loc="left",
    pad=35,
    color=DARK_TEAL
)

ax.text(
    0, 1.025,
    "1,000 trials with randomly perturbed data points",
    transform=ax.transAxes,
    fontsize=12,
    fontweight="bold",
    color=MUTED
)

ax.grid(axis="y", color=MUTED, alpha=0.18, linewidth=0.8)
ax.set_axisbelow(True)

ax.legend(
    loc="upper right",
    prop={"size": 12, "weight": "bold"}
)

fig.tight_layout(pad=2.5)
fig.savefig(save_path + r"\monte_carlo.png", bbox_inches="tight")


# CENSOR DATA ----------------------------------------------------------------------------------

fractions = np.linspace(0, 0.9, 10)
error_factors = []

for fraction in fractions:
    errors = []

    for i in range(100):
        n = int(len(velocity) * (1 - fraction))
        idx = np.random.choice(len(velocity), n, replace=False)

        _, new_cov = np.polyfit(
            velocity[idx],
            distance[idx],
            1,
            w=1 / distance_err[idx],
            cov="unscaled"
        )

        errors.append(np.sqrt(new_cov[0, 0]))

    error_factors.append(np.mean(errors) / slope_err)


# PLOT FOUR: DATA CENSORING --------------------------------------------------------------------

fig, ax = matplotlib.pyplot.subplots(figsize=(10, 6.5))
ax.set_facecolor(AX_BG)

x_percent = fractions * 100

ax.fill_between(
    x_percent,
    1,
    error_factors,
    color=LIGHT_TEAL,
    alpha=0.5
)

ax.plot(
    x_percent,
    error_factors,
    color=TEAL,
    linewidth=3,
    marker="o",
    markersize=9,
    markerfacecolor=LIGHT_TEAL,
    markeredgecolor=TEAL,
    markeredgewidth=1.5,
    label="mean uncertainty increase",
    zorder=3
)

ax.axhline(
    1,
    color=ORANGE,
    linewidth=2.5,
    linestyle=(0, (5, 3)),
    label="original uncertainty"
)

ax.set_xlabel("data censored (%)", labelpad=15)
ax.set_ylabel("slope uncertainty multiplier", labelpad=15)

ax.set_title(
    "the cost of missing data",
    loc="left",
    pad=35,
    color=DARK_TEAL
)

ax.text(
    0, 1.025,
    "mean uncertainty after randomly removing observations",
    transform=ax.transAxes,
    fontsize=12,
    fontweight="bold",
    color=MUTED
)

ax.grid(axis="y", color=MUTED, alpha=0.18, linewidth=0.8)
ax.set_axisbelow(True)

ax.legend(
    loc="upper left",
    prop={"size": 12, "weight": "bold"}
)

fig.tight_layout(pad=2.5)
fig.savefig(save_path + r"\data_censoring.png", bbox_inches="tight")


# MAKE TICK LABELS BOLD ON ALL FOUR FIGURES ----------------------------------------------------

for figure in matplotlib.pyplot.get_fignums():
    current_fig = matplotlib.pyplot.figure(figure)

    for current_ax in current_fig.axes:
        for label in current_ax.get_xticklabels() + current_ax.get_yticklabels():
            label.set_fontweight("bold")

    current_fig.tight_layout(pad=2)

    filename = {
        1: "distance_vs_velocity.png",
        2: "residuals.png",
        3: "monte_carlo.png",
        4: "data_censoring.png"
    }.get(figure)

    if filename:
        current_fig.savefig(save_path + "\\" + filename, bbox_inches="tight")


# DISPLAY ALL FOUR PLOTS -----------------------------------------------------------------------

matplotlib.pyplot.show()
