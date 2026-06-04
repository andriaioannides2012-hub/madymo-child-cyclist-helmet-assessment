# -*- coding: utf-8 -*-
"""
Generate publication-quality figures for the MADYMO child cyclist
reconstruction sensitivity study.

Distribution figures use a transparent box plot (median, IQR, whiskers)
with every individual run value overlaid as a semi-transparent jittered
point, plus a diamond marker for the mean.

All distribution figures use the 46 VALID runs (HIC15 <= 10,000).
The HIC15 overview figure shows all 55 completed runs, with the 9
numerically invalid runs (HIC15 > 10,000) flagged in red.

Each figure block is self-contained. Comment out any block you do not
want to regenerate.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

#-----------------------------------------------------------------------------#
"""Settings"""
#-----------------------------------------------------------------------------#

filePath_results = 'C:/Users/andri/OneDrive/Imperial/Year_4/Masters Project/Results'
fileName_results = '26_05_FINAL_SEMI_LONG_SIMULATION_sens_impactData.xlsx'
filePath_figures = 'C:/Users/andri/OneDrive/Imperial/Year_4/Masters Project/Results/Figures'

HIC_VALIDITY_THRESHOLD = 10000
G = 9.81

ACCEL_EN1080_G = 250
DAI_THRESHOLD  = 10000
NECK_EXT_LIMIT = 135

POINT_ALPHA = 0.65
JITTER_W    = 0.10
SEED        = 1

#-----------------------------------------------------------------------------#
"""Styling"""
#-----------------------------------------------------------------------------#

plt.rcParams.update({
    'font.family':          'DejaVu Sans',
    'font.size':            11,
    'axes.titlesize':       12,
    'axes.titleweight':     'bold',
    'axes.titlepad':        12,
    'axes.labelsize':       11,
    'axes.labelpad':        8,
    'axes.linewidth':       0.8,
    'axes.edgecolor':       '#CCCCCC',
    'axes.facecolor':       'white',
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.spines.left':     True,
    'axes.spines.bottom':   True,
    'axes.grid':            True,
    'grid.color':           '#E5E5E5',
    'grid.linewidth':       0.7,
    'grid.alpha':           1.0,
    'grid.linestyle':       '-',
    'figure.facecolor':     'white',
    'figure.dpi':           150,
    'legend.framealpha':    0.95,
    'legend.edgecolor':     '#CCCCCC',
    'legend.fontsize':      8.5,
    'xtick.labelsize':      10,
    'ytick.labelsize':      10,
    'xtick.color':          '#555555',
    'ytick.color':          '#555555',
    'axes.labelcolor':      '#333333',
    'axes.titlecolor':      '#222222',
})

C_PRIMARY   = '#3A7CB8'
C_SECONDARY = '#E07B39'
C_GREEN     = '#3E9E6F'
C_PURPLE    = '#7B67B0'
C_RED       = '#C0392B'
C_THRESH    = '#C0392B'
C_BOX       = '#444444'

os.makedirs(filePath_figures, exist_ok=True)
rng = np.random.default_rng(SEED)

#-----------------------------------------------------------------------------#
"""Load data"""
#-----------------------------------------------------------------------------#

df_all   = pd.read_excel(os.path.join(filePath_results, fileName_results), sheet_name='sheet1')
df_valid = df_all[df_all['HIC15'] <= HIC_VALIDITY_THRESHOLD].copy().reset_index(drop=True)
df_invalid = df_all[df_all['HIC15'] > HIC_VALIDITY_THRESHOLD].copy().reset_index(drop=True)

print(f'Total completed runs : {len(df_all)}')
print(f'Valid runs           : {len(df_valid)}')
print(f'Invalid runs         : {len(df_invalid)} -> {df_invalid["RUN"].astype(int).tolist()}')


def save(fig, name):
    path = os.path.join(filePath_figures, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved:', name)


def box_strip(ax, data_list, labels, point_color, box_color=None):
    bc = box_color or C_BOX
    positions = np.arange(1, len(data_list) + 1)

    bp = ax.boxplot(data_list, positions=positions, widths=0.42,
                    patch_artist=True, showfliers=False, zorder=2,
                    medianprops=dict(color=bc, linewidth=2.4),
                    whiskerprops=dict(color=bc, linewidth=1.1, linestyle='-'),
                    capprops=dict(color=bc, linewidth=1.1),
                    boxprops=dict(facecolor='none', edgecolor=bc, linewidth=1.3))

    for i, d in enumerate(data_list):
        d = np.asarray(d)
        x = positions[i] + rng.uniform(-JITTER_W, JITTER_W, size=len(d))
        ax.scatter(x, d, s=30, color=point_color, alpha=POINT_ALPHA,
                   edgecolors='white', linewidth=0.35, zorder=3)
        ax.scatter([positions[i]], [np.nanmean(d)], marker='D', s=58,
                   color=C_SECONDARY, edgecolors='white', linewidth=1.0, zorder=6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    return bp


def threshold_line(ax, y, label, color=C_THRESH, ls='--', lw=1.4):
    ax.axhline(y, color=color, ls=ls, lw=lw, zorder=0, alpha=0.85)
    xlim = ax.get_xlim()
    xpos = xlim[1] - (xlim[1]-xlim[0])*0.01
    ax.text(xpos, y, f'  {label}', va='bottom', ha='right',
            fontsize=8, color=color,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5, alpha=0.8))


def legend_box_strip(ax, point_color=None, loc='upper left'):
    pc = point_color or C_PRIMARY
    handles = [
        Line2D([0],[0], color=C_BOX, lw=2.4, label='Median'),
        mpatches.Patch(facecolor='none', edgecolor=C_BOX, lw=1.3, label='IQR (25–75%)'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor=C_SECONDARY,
               markeredgecolor='white', markersize=9, label='Mean'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=pc,
               markeredgecolor='white', markersize=8, alpha=POINT_ALPHA,
               label='Individual run'),
    ]
    leg = ax.legend(handles=handles, loc=loc, framealpha=0.95,
                    edgecolor='#CCCCCC', fontsize=8.5)
    leg.get_frame().set_linewidth(0.7)


#=============================================================================#
# FIGURE 1 — HIC15 overview across ALL runs, invalid flagged red
#=============================================================================#

fig, ax = plt.subplots(figsize=(15, 5.5))
runs = df_all['RUN'].astype(int).astype(str).tolist()
hics = df_all['HIC15'].tolist()

bar_colors, display_vals = [], []
for h in hics:
    if   h > HIC_VALIDITY_THRESHOLD: bar_colors.append(C_RED);       display_vals.append(HIC_VALIDITY_THRESHOLD)
    elif h > 2400:                    bar_colors.append('#E07B39');    display_vals.append(h)
    elif h > 1000:                    bar_colors.append('#E8C547');    display_vals.append(h)
    else:                             bar_colors.append(C_PRIMARY);    display_vals.append(h)

ax.bar(runs, display_vals, color=bar_colors, edgecolor='white', linewidth=0.3, width=0.75)
for i, h in enumerate(hics):
    if h > HIC_VALIDITY_THRESHOLD:
        ax.text(i, HIC_VALIDITY_THRESHOLD * 1.015, '▲', ha='center',
                va='bottom', color=C_RED, fontsize=9, fontweight='bold')

ax.axhline(2400, color='#E07B39', ls='--', lw=1.3, zorder=0, alpha=0.9)
ax.axhline(1000, color='#C9A227', ls='--', lw=1.3, zorder=0, alpha=0.9)
ax.axhline(HIC_VALIDITY_THRESHOLD, color=C_RED, ls='-', lw=1.1, zorder=0, alpha=0.7)
ax.text(len(runs)-0.5, 2400, '  EN 1078 equiv. (2400)', va='bottom', ha='right', fontsize=8, color='#C07020',
        bbox=dict(facecolor='white', edgecolor='none', pad=1.0, alpha=0.85))
ax.text(len(runs)-0.5, 1000, '  AIS3+ 18% risk (1000)', va='bottom', ha='right', fontsize=8, color='#A07800',
        bbox=dict(facecolor='white', edgecolor='none', pad=1.0, alpha=0.85))

ax.set_ylim(0, HIC_VALIDITY_THRESHOLD * 1.09)
ax.set_xlabel('Run number', labelpad=8); ax.set_ylabel('HIC15')
ax.set_title('HIC15 across all completed simulations\n(numerically invalid runs capped and flagged in red)')
ax.tick_params(axis='x', rotation=90, labelsize=6.5)
ax.spines['bottom'].set_color('#CCCCCC'); ax.spines['left'].set_color('#CCCCCC')
ax.legend(handles=[
    mpatches.Patch(color=C_PRIMARY,  label='HIC15 ≤ 1000'),
    mpatches.Patch(color='#E8C547',  label='1000 < HIC15 ≤ 2400'),
    mpatches.Patch(color='#E07B39',  label='2400 < HIC15 ≤ 10 000'),
    mpatches.Patch(color=C_RED,      label='HIC15 > 10 000 (invalid, excluded)'),
], fontsize=8.5, loc='upper right', framealpha=0.95, edgecolor='#CCCCCC')
save(fig, 'Fig1_HIC15_overview_all_runs.png')


#=============================================================================#
# FIGURE 2 — Velocities
#=============================================================================#
fig, ax = plt.subplots(figsize=(7, 5.5))
data_v = [np.abs(df_valid['Head impact speed (m/s)'].dropna().values),
          df_valid['Motorcycle impact vres (m/s)'].dropna().values]
box_strip(ax, data_v, ['Head impact\nspeed', 'Bicycle impact speed\nrelative to van'], C_PRIMARY)
ax.set_ylim(bottom=0)
threshold_line(ax, 5.42, 'EN 1080: 5.42 m/s')
threshold_line(ax, 4.57, 'EN 1080: 4.57 m/s', ls=':')
ax.set_ylabel('Speed (m/s)')
ax.set_title('Impact speed distributions (n = 46)')
legend_box_strip(ax, C_PRIMARY, loc='upper left')
save(fig, 'Fig2_velocity_distributions.png')


#=============================================================================#
# FIGURE 3 — Impact angle
#=============================================================================#
fig, ax = plt.subplots(figsize=(5.5, 5.5))
data_ang = [df_valid['headVres-surface angle (deg)'].dropna().values]
box_strip(ax, data_ang, ['Head–surface\nimpact angle'], C_GREEN)
ax.set_ylim(0, 100)
threshold_line(ax, 90, 'EN 1080: 90° (normal impact)')
ax.set_ylabel('Angle (degrees)')
ax.set_title('Impact angle distribution (n = 46)')
legend_box_strip(ax, C_GREEN, loc='lower left')
save(fig, 'Fig3_impact_angle_distribution.png')


#=============================================================================#
# FIGURE 4 — Euler angles  (legend moved to lower right)
#=============================================================================#
fig, ax = plt.subplots(figsize=(8.5, 5.5))
data_e = [np.degrees(df_valid['Head Euler Y rot (rad)'].dropna().values),
          np.degrees(df_valid['Head Euler X rot (rad)'].dropna().values),
          np.degrees(df_valid['Head Euler Z rot (rad)'].dropna().values)]
box_strip(ax, data_e,
          ['Euler Y\n(forward tilt / pitch)', 'Euler X\n(roll)', 'Euler Z\n(yaw)'],
          C_PURPLE)
ax.set_ylabel('Angle (degrees)')
ax.set_title('Head Euler angle distributions at impact (n = 46)')
legend_box_strip(ax, C_PURPLE, loc='lower right')   # moved from center left
save(fig, 'Fig4_euler_angle_distributions.png')


#=============================================================================#
# FIGURE 5 — Linear acceleration (g) — X, Y, Z as absolute values
#=============================================================================#
fig, ax = plt.subplots(figsize=(9, 5.5))
data_a = [df_valid['Head Ares (m/s^2)'].dropna().values / G,
          np.abs(df_valid['Head Ax (m/s^2)'].dropna().values / G),
          np.abs(df_valid['Head Ay (m/s^2)'].dropna().values / G),
          np.abs(df_valid['Head Az (m/s^2)'].dropna().values / G)]
box_strip(ax, data_a, ['Resultant', '|X|', '|Y|', '|Z|'], C_PRIMARY)
threshold_line(ax, ACCEL_EN1080_G, 'EN 1080 limit (250 g)')
ax.set_ylim(bottom=0)
ax.set_ylabel('Peak linear acceleration (g)')
ax.set_title('Head linear acceleration distributions (n = 46)')
legend_box_strip(ax, C_PRIMARY, loc='upper right')
save(fig, 'Fig5_linear_acceleration_distributions.png')


#=============================================================================#
# FIGURE 6 — Angular acceleration — X, Y, Z signed with ± DAI threshold lines
#=============================================================================#
fig, ax = plt.subplots(figsize=(9, 5.5))
data_ar = [df_valid['Head ares (rad/s^2)'].dropna().values,
           df_valid['Head ax (rad/s^2)'].dropna().values,
           df_valid['Head ay (rad/s^2)'].dropna().values,
           df_valid['Head az (rad/s^2)'].dropna().values]
box_strip(ax, data_ar, ['Resultant', 'X', 'Y', 'Z'], C_SECONDARY)

# Positive threshold
ax.axhline( DAI_THRESHOLD, color=C_THRESH, ls='--', lw=1.4, zorder=0, alpha=0.85)
# Negative threshold
ax.axhline(-DAI_THRESHOLD, color=C_THRESH, ls='--', lw=1.4, zorder=0, alpha=0.85)

xlim = ax.get_xlim()
xpos = xlim[1] - (xlim[1]-xlim[0])*0.01
ax.text(xpos,  DAI_THRESHOLD, '  +10 000 rad/s²', va='bottom', ha='right', fontsize=8, color=C_THRESH,
        bbox=dict(facecolor='white', edgecolor='none', pad=1.5, alpha=0.8))
ax.text(xpos, -DAI_THRESHOLD, '  −10 000 rad/s²', va='top',    ha='right', fontsize=8, color=C_THRESH,
        bbox=dict(facecolor='white', edgecolor='none', pad=1.5, alpha=0.8))

ax.set_ylabel('Peak angular acceleration (rad/s²)')
ax.set_title('Head angular acceleration distributions (n = 46)')
legend_box_strip(ax, C_SECONDARY, loc='upper right')
save(fig, 'Fig6_angular_acceleration_distributions.png')


#=============================================================================#
# FIGURE 7 — Helmet contact force — X, Y, Z as absolute values
#=============================================================================#
fig, ax = plt.subplots(figsize=(9, 5.5))
data_f = [df_valid['Helm Fres (N)'].dropna().values,
          np.abs(df_valid['Helm Fx (N)'].dropna().values),
          np.abs(df_valid['Helm Fy (N)'].dropna().values),
          np.abs(df_valid['Helm Fz (N)'].dropna().values)]
box_strip(ax, data_f, ['Resultant', '|X|', '|Y|', '|Z|'], C_PRIMARY)
ax.set_ylim(bottom=0)
ax.set_ylabel('Peak helmet contact force (N)')
ax.set_title('Helmet contact force distributions at peak head force (n = 46)')
legend_box_strip(ax, C_PRIMARY, loc='upper right')
save(fig, 'Fig7_helmet_force_distributions.png')


#=============================================================================#
# FIGURE 8 — Neck force — X, Y, Z as absolute values
#=============================================================================#
fig, ax = plt.subplots(figsize=(9, 5.5))
data_n = [df_valid['Neck Fres (N)'].dropna().values,
          np.abs(df_valid['Neck Fx (N)'].dropna().values),
          np.abs(df_valid['Neck Fy (N)'].dropna().values),
          np.abs(df_valid['Neck Fz (N)'].dropna().values)]
box_strip(ax, data_n, ['Resultant', '|X|', '|Y|', '|Z|'], C_GREEN)
ax.set_ylim(bottom=0)
ax.set_ylabel('Peak neck force (N)')
ax.set_title('Neck force distributions (n = 46)')
legend_box_strip(ax, C_GREEN, loc='upper right')
save(fig, 'Fig8_neck_force_distributions.png')


#=============================================================================#
# FIGURE 9 — Neck extension moment (kept signed — extension vs flexion matters)
#=============================================================================#
fig, ax = plt.subplots(figsize=(5.5, 5.5))
data_m = [df_valid['Peak neck extension (Nm)'].dropna().values]
box_strip(ax, data_m, ['Peak neck\nextension moment'], C_PURPLE)
threshold_line(ax, -NECK_EXT_LIMIT, 'Hybrid III adult limit (−135 Nm)')
ax.set_ylabel('Moment (Nm)')
ax.set_title('Neck extension moment distribution (n = 46)')
legend_box_strip(ax, C_PURPLE, loc='upper left')
save(fig, 'Fig9_neck_moment_distribution.png')


print('\nAll figures generated in:', filePath_figures)
