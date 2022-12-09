import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "legend.fontsize": 14,  # this is the font size in legends
        "xtick.labelsize": 14,  # this and next are the font of ticks
        "ytick.labelsize": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 18,  # this is the foflags.N of axes labels
        "savefig.format": "pdf",  # how figures should be saved
        "legend.edgecolor": "0.0",
        "legend.framealpha": 1.0,
    }
)

data41 = np.load(
    "string_2d_Ising_V4_1_critical_nucleus_size_normalized_gamma.npy", allow_pickle=True
)
data42 = np.load(
    "string_2d_Ising_V4_2_critical_nucleus_size_normalized_gamma.npy", allow_pickle=True
)

Jc = 1 / 4
J41 = Jc * 2.27 / 1.5
J42 = Jc * 2.27 / 1.9

fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)

ax.plot(
    data41[0],
    data41[1],
    "o-",
    color="tab:blue",
    label=r"$\frac{J}{J_c}=\frac{2.27}{1.5}$ String Method",
)
ax.plot(
    data42[0],
    data42[1],
    "o-",
    color="tab:orange",
    label=r"$\frac{J}{J_c}=\frac{2.27}{1.9}$ String Method",
)

ax.plot(
    data41[0],
    data41[-1],
    "o-",
    color="tab:green",
    label=r"$\frac{J}{J_c}=\frac{2.27}{1.5}$ CNT",
)
ax.plot(
    data42[0],
    data42[-1],
    "o-",
    color="tab:purple",
    label=r"$\frac{J}{J_c}=\frac{2.27}{1.9}$ CNT",
)
h_mesh = np.linspace(0.001, 0.2, 100)
ax.plot(h_mesh, np.pi / 4 / h_mesh ** 2, "k--", label="$\pi/(4h^2)$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("$A/\gamma^2$")
ax.set_xlabel("$h$")
ax.legend()

fig.savefig("string_2d_Ising_V4_[1-2]_critical_nucleus_size_normalized_gamma")
