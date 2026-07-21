import numpy as np

N = np.array([1.77, 4.72, 10.62, 25.17, 56.62])
L = np.array([1.8256, 1.67577, 1.55546, 1.48705, 1.415])

candidate_Es = np.arange(0.5, 1.41, 0.02)

best_E = None
best_r2 = -1.0
best_slope = None
best_intercept = None

log_N = np.log(N)

for E in candidate_Es:
    log_L_minus_E = np.log(L - E)
    slope, intercept = np.polyfit(log_N, log_L_minus_E, 1)
    r2 = np.corrcoef(log_N, log_L_minus_E)[0, 1] ** 2
    if r2 > best_r2:
        best_r2 = r2
        best_E = E
        best_slope = slope
        best_intercept = intercept

alpha = -best_slope
A = np.exp(best_intercept)
print(f"best floor E = {best_E:.3f}")
print(f"alpha (exponent) = {alpha:.3f}")
print(f"A (coefficient)  = {A:.3f}")
print(f"straightness R^2 = {best_r2:.4f}")