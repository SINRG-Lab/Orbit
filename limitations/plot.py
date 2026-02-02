#!/usr/bin/env python3
"""
plot_ladder_comparison.py

Load ladder comparison time-series from CSV and plot:
- Available bandwidth
- Oracle per-object ladders (near/far)
- Static global ladder (near/far)

Requires:
  pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.figsize'] = 20, 8
matplotlib.rcParams.update({'font.size': 35})

fig, ax = plt.subplots()
fig.subplots_adjust(left=0.1, bottom=0.15, right=0.96, top=0.96)


# Path to CSV file
csv_path = "ladder_comparison_timeseries.csv"

# Load data (skip header row)
data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

time   = data[:, 0]
bw     = data[:, 1]
o_near = data[:, 2]
o_far  = data[:, 3]
s_near = data[:, 4]
s_far  = data[:, 5]

plt.plot(time, bw, linestyle='-', linewidth=5, color='black')
plt.plot(time, o_near, linestyle='dotted', linewidth=5, color='green', alpha=0.7, label="Oracle - Near Object")
plt.plot(time, o_far,  linestyle='-.', linewidth=3, color='blue', alpha=0.7, label="Oracle - Far Object")
plt.plot(time, s_near, linestyle='dashed', linewidth=5, color='red', alpha=0.7, label="Global ladder - Near Object")
plt.plot(time, s_far,  linestyle='--', linewidth=3, color='magenta', alpha=0.7, label="Global ladder - Far Object")

ax = plt.gca()

# Turn off top/right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Make left/bottom spines thicker
ax.spines["left"].set_linewidth(3)
ax.spines["bottom"].set_linewidth(3)

# (optional but nice) match tick thickness
ax.tick_params(axis="both", width=3, length=12)


plt.xlabel("Time (seconds)")
plt.ylabel("Bitrate (Mbps)")
plt.ylim(0, 135)
plt.xlim(0, 250)
plt.legend()
plt.grid(True, alpha=0.25)
plt.show()

