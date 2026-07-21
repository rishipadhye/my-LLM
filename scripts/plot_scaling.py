import matplotlib.pyplot as plt
import numpy as np

N = [1.77, 4.72, 10.62, 25.17, 56.62]
L = [1.8256, 1.67577, 1.55546, 1.48705, 1.415]
names = ["xs", "s", "m", "base", "l"]
E = 1.160
A = 0.780
alpha = 0.276

N_smooth = np.logspace(np.log10(min(N)), np.log10(max(N)), 200)
L_fit = E + A * N_smooth ** (-alpha)

fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(N, L, label="runs (40k budget)", zorder=3)
ax.plot(N_smooth, L_fit, color="orange", label="fit: L = 1.16 + 0.78·N^-0.276")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Non-embedding params (millions)")
ax.set_ylabel("Best val_loss")
ax.set_title("TinyStories scaling: loss vs params (log-log, 40k ladder + 80k diagnostic)")

for x, y, name in zip(N, L, names):
    ax.annotate(name, (x, y))

L_80k = 1.325
ax.scatter([56.62], [L_80k], color="red", marker="*", s=250, zorder=4, label="scale_l @ 80k (2x budget)")
ax.annotate("", xy=(56.62, L_80k), xytext=(56.62, 1.415), arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
ax.legend()
fig.savefig("assets/scaling_loglog.png", dpi=150, bbox_inches="tight")
print("saved to assets/scaling_loglog.png")
