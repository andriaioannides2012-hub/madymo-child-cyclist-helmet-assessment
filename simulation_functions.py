# -*- coding: utf-8 -*-
"""
simulation_functions.py

Core processing functions for extracting and computing head impact metrics
from MADYMO multibody simulation output files.

Adapted from original code by Lucas Lacroix (ll6115), Imperial College London.
Modified by Andria Elia, Imperial College London, 2026.

Modifications include:
  - CSV parser rewritten for MADYMO 2021.1 single-row output format
  - HIC15 time-scaling bug fix (removed erroneous /1000 factor)
  - Impact onset threshold added to find_peaks (height=100) to filter
    spurious near-zero contact force peaks
  - Bicycle-van contact force extraction added (ptw_car_Fres)
  - Variable naming updated throughout
"""

import math
import numpy as np
import os
import pandas as pd
import scipy.signal
import shutil


# =============================================================================
# Injury metric functions
# =============================================================================

def GetHIC(time, acc, t_window=0.015):
    """
    Calculate HIC15 (Head Injury Criterion, 15 ms window).

    HIC = max[ (t2-t1) * ( (1/(t2-t1)) * integral(a(t), t1, t2) )^2.5 ]

    Parameters
    ----------
    time : array-like
        Time values [s]
    acc : array-like
        Resultant head acceleration [g]

    Returns
    -------
    hic : int
    hic_t1 : float  Start time of worst-case window [s]
    hic_t2 : float  End time of worst-case window [s]
    """
    time = np.asarray(time)
    acc  = np.asarray(acc)
    n    = len(time)

    # Cumulative integral over entire pulse (trapezoidal)
    I = np.zeros_like(acc)
    I[1:] = np.cumsum(((acc[:-1] + acc[1:]) / 2) * np.diff(time))

    hic    = 0
    hic_t1 = 0.0
    hic_t2 = 0.0
    j      = 0

    for i in range(n):
        while j < n and (time[j] - time[i]) <= t_window:
            j += 1
        for k in range(i + 1, j):
            dt       = time[k] - time[i]
            integral = I[k] - I[i]
            avg_acc  = integral / dt
            hic_temp = dt * (avg_acc ** 2.5)
            if hic_temp > hic:
                hic    = int(np.round(hic_temp))
                hic_t1 = time[i]
                hic_t2 = time[j - 1]

    return hic, hic_t1, hic_t2


def GetContiguous3ms(time, acc, t_window=0.03):
    """
    Compute the contiguous 3 ms acceleration clip criterion.

    Returns the maximum value for which the acceleration equals or exceeds
    that value continuously for at least 3 ms.

    Parameters
    ----------
    time : array-like
        Time values [s]
    acc : array-like
        Resultant head acceleration [g]

    Returns
    -------
    contiguous3ms : int   [g]
    c3ms_t1 : float
    c3ms_t2 : float
    """
    time = np.asarray(time)
    acc  = np.asarray(acc)
    n    = len(time)

    contiguous3ms = 0
    c3ms_t1 = 0.0
    c3ms_t2 = 0.0
    j = 0

    for i in range(n):
        while j < n and (time[j] - time[i]) < t_window:
            j += 1
        if j >= n:
            break
        contiguous3ms_temp = np.min(acc[i + 1:j])
        if contiguous3ms_temp > contiguous3ms:
            contiguous3ms = int(np.round(contiguous3ms_temp))
            c3ms_t1 = time[i]
            c3ms_t2 = time[j - 1]

    return contiguous3ms, c3ms_t1, c3ms_t2


# =============================================================================
# Geometry and linear algebra utilities
# =============================================================================

def RotationMatrix(rx, ry, rz):
    """Return ZYX rotation matrix from Euler angles [rad]."""
    return np.array([
        [math.cos(ry)*math.cos(rz),
         math.sin(rx)*math.sin(ry)*math.cos(rz) - math.cos(rx)*math.sin(rz),
         math.cos(rx)*math.sin(ry)*math.cos(rz) + math.sin(rx)*math.sin(rz)],
        [math.cos(ry)*math.sin(rz),
         math.sin(rx)*math.sin(ry)*math.sin(rz) + math.cos(rx)*math.cos(rz),
         math.cos(rx)*math.sin(ry)*math.sin(rz) - math.sin(rx)*math.cos(rz)],
        [-math.sin(ry),
         math.sin(rx)*math.cos(ry),
         math.cos(rx)*math.cos(ry)]
    ])


def TransformationMatrix(rotMat, P):
    """Assemble 4x4 homogeneous transformation matrix."""
    P = np.array(P, dtype=float)
    transMat = np.column_stack((rotMat, P))
    transMat = np.vstack((transMat, [0, 0, 0, 1]))
    return transMat


def TransMat2EulerZYX(T):
    """Extract ZYX Euler angles [rad] from a 4x4 transformation matrix."""
    R = T[:3, :3]
    if abs(R[2, 0]) < 1:
        pitch = -np.arcsin(R[2, 0])
        yaw   =  np.arctan2(R[1, 0], R[0, 0])
        roll  =  np.arctan2(R[2, 1], R[2, 2])
    else:
        pitch = np.pi / 2 if R[2, 0] <= -1 else -np.pi / 2
        yaw   = np.arctan2(-R[0, 1], R[1, 1])
        roll  = 0
    return yaw, pitch, roll


def NormaliseVector(V):
    """Return unit vector of V."""
    V = np.array(V, dtype=float)
    return V / np.linalg.norm(V)


def AngleBetweenVectors(V1, V2):
    """Return angle between two vectors in degrees."""
    dot = np.clip(np.dot(np.array(V1), np.array(V2)), -1.0, 1.0)
    return 57.3 * np.arccos(dot)


# =============================================================================
# Peak extraction utility
# =============================================================================

def GetPeakAfterImpact(time, data, i_peak_impact, timeWindow=10, direction="pos"):
    """
    Find the peak of a signal within a time window following the impact onset.

    Parameters
    ----------
    time : list
        Time array [ms or s, must be consistent with timeWindow units]
    data : list
        Signal to search
    i_peak_impact : int
        Index of impact onset — search begins here
    timeWindow : int
        Duration after impact onset to search [same units as time], default 10
    direction : str
        'pos'  — find largest positive peak
        'neg'  — find largest negative peak (signal inverted internally)
        'abs'  — find largest absolute-value peak

    Returns
    -------
    i_peak_data : int
    """
    if direction == "neg":
        data = [-x for x in data]
    elif direction == "abs":
        data = [abs(x) for x in data]

    i_peaks, _ = scipy.signal.find_peaks(data, distance=5, prominence=50)
    i_peaks = i_peaks.tolist()

    if i_peaks:
        i_peak_data = i_peak_impact
        for idx in i_peaks:
            if (time[idx] - time[i_peak_impact] <= timeWindow and
                    time[idx] - time[i_peak_impact] > 0 and
                    data[idx] > data[i_peak_data]):
                i_peak_data = idx
    else:
        i_peak_data = i_peak_impact

    return i_peak_data


# =============================================================================
# MADYMO CSV extraction
# =============================================================================

def ExtractResults(fileName_results):
    """
    Parse MADYMO CSV output files into Python lists.

    Parameters
    ----------
    fileName_results : list of str
        Ordered list of MADYMO output filenames (with extensions).
        Expected order:
          0  MadymoOutput_dvl.csv   — relative distance/speed, head to impact points
          1  MadymoOutput_cntfrc.csv — contact forces
          2  MadymoOutput_rtf.csv   — neck joint forces
          3  MadymoOutput_aps.csv   — angular positions (neck, torso in head frame)
          4  MadymoOutput_lps.csv   — linear positions (unused directly)
          5  MadymoOutput_aac.csv   — angular accelerations (helmet, head)
          6  MadymoOutput_rds.csv   — relative distance/coords, head to impact points
          7  MadymoOutput_lvl.csv   — linear velocities (head, van, bicycle)
          8  MadymoOutput_rtt.csv   — neck joint torques
          9  MadymoOutput_avl.csv   — head angular velocity
         10  MadymoOutput_lac.csv   — head linear acceleration

    Returns
    -------
    Tuple of parsed data arrays (see variable names below).
    """
    nImpactPoints = 9

    for i in range(len(fileName_results)):
        with open(fileName_results[i], "r") as f:
            fileText = f.readlines()

        # All MADYMO 2021.1 CSV files have 5 header rows
        del fileText[0:5]

        # Parse rows to float lists
        fileText2 = []
        for line in fileText:
            parts = [x.strip() for x in line.strip().split(',') if x.strip()]
            try:
                fileText2.append([float(v) for v in parts])
            except ValueError:
                continue

        j = 0

        if i == 0:  # Relative distance and speed, head to impact points
            van_head_dvl = [[]]
            for _ in range(nImpactPoints):
                van_head_dvl.append([])
            while j <= len(fileText2) - 1:
                row = fileText2[j]
                van_head_dvl[0].append(row[0])
                for k in range(nImpactPoints):
                    van_head_dvl[k + 1].append(row[2 + k * 2])
                j += 1

        elif i == 1:  # Contact forces
            # Column indices (0-based):
            # Helmet_HumanHead : cols 1-8   (Fres, Fx, Fy, Fz, Tres, Tx, Ty, Tz)
            # Helmet_Van       : cols 21-28
            # Head_Van         : cols 41-48
            # Bicycle_Van      : col  61    (Fres only)
            headCOM_cntfrc    = [[] for _ in range(9)]  # t, Fres, Fx, Fy, Fz, Tres, Tx, Ty, Tz
            helmetCOM_cntfrc  = [[] for _ in range(9)]
            headCOM_van_cntfrc = [[] for _ in range(9)]
            bicycle_cntfrc    = [[], []]                # t, Fres

            while j <= len(fileText2) - 1:
                row = fileText2[j]
                for arr in [headCOM_cntfrc, helmetCOM_cntfrc, headCOM_van_cntfrc, bicycle_cntfrc]:
                    arr[0].append(row[0])

                # Helmet_HumanHead
                for k, col in enumerate(range(1, 9)):
                    headCOM_cntfrc[k + 1].append(row[col])

                # Helmet_Van
                for k, col in enumerate(range(21, 29)):
                    helmetCOM_cntfrc[k + 1].append(row[col])

                # Head_Van
                for k, col in enumerate(range(41, 49)):
                    headCOM_van_cntfrc[k + 1].append(row[col])

                # Bicycle_Van
                bicycle_cntfrc[1].append(row[61])

                j += 1

        elif i == 2:  # Neck joint forces
            neck_rtf = [[] for _ in range(11)]  # t, Fres, Fx(neck), Fy, Fz, Bx, By, Bz, Fx(head), Fy(head), Fz(head)
            while j <= len(fileText2) - 1:
                row = fileText2[j]
                neck_rtf[0].append(row[0])
                for k, col in enumerate([1, 2, 3, 4, 5, 6, 7]):
                    neck_rtf[k + 1].append(row[col])
                neck_rtf[8].append(row[12])
                neck_rtf[9].append(row[13])
                neck_rtf[10].append(row[14])
                j += 1

        elif i == 3:  # Angular positions (neck, torso in head frame)
            neck_head_aps  = [[] for _ in range(4)]
            torso_head_aps = [[] for _ in range(4)]
            while j <= len(fileText2) - 1:
                row = fileText2[j]
                neck_head_aps[0].append(row[0])
                torso_head_aps[0].append(row[0])
                neck_head_aps[1].append(row[4])
                neck_head_aps[2].append(row[5])
                neck_head_aps[3].append(row[6])
                torso_head_aps[1].append(row[7])
                torso_head_aps[2].append(row[8])
                torso_head_aps[3].append(row[9])
                j += 1

        elif i == 5:  # Angular accelerations (helmet, head)
            headCOM_aac   = [[] for _ in range(5)]
            helmetCOM_aac = [[] for _ in range(5)]
            while j <= len(fileText2) - 1:
                row = fileText2[j]
                headCOM_aac[0].append(row[0])
                helmetCOM_aac[0].append(row[0])
                for k in range(1, 5):
                    helmetCOM_aac[k].append(row[k])
                    headCOM_aac[k].append(row[k + 4])
                j += 1

        elif i == 6:  # Relative distance and coords, head to impact points
            van_head_rds = [[]]
            for _ in range(nImpactPoints):
                van_head_rds.append([[], [], [], []])
            while j <= len(fileText2) - 1:
                row = fileText2[j]
                van_head_rds[0].append(row[0])
                for k in range(nImpactPoints):
                    van_head_rds[k + 1][0].append(row[1 + k * 4])
                    van_head_rds[k + 1][1].append(row[2 + k * 4])
                    van_head_rds[k + 1][2].append(row[3 + k * 4])
                    van_head_rds[k + 1][3].append(row[4 + k * 4])
                j += 1

        elif i == 7:  # Linear velocities (head, van, bicycle)
            head_lvl      = [[] for _ in range(4)]
            van_head_lvl  = [[] for _ in range(4)]
            bicycle_van_lvl = [[], []]
            while j <= len(fileText2) - 1:
                row = fileText2[j]
                for arr in [head_lvl, van_head_lvl, bicycle_van_lvl]:
                    arr[0].append(row[0])
                head_lvl[1].append(row[2])
                head_lvl[2].append(row[3])
                head_lvl[3].append(row[4])
                van_head_lvl[1].append(row[6])
                van_head_lvl[2].append(row[7])
                van_head_lvl[3].append(row[8])
                bicycle_van_lvl[1].append(row[9])
                j += 1

        elif i == 8:  # Neck joint torques
            neck_rtt = [[] for _ in range(11)]
            while j <= len(fileText2) - 1:
                row = fileText2[j]
                neck_rtt[0].append(row[0])
                for k, col in enumerate([1, 2, 3, 4, 5, 6, 7]):
                    neck_rtt[k + 1].append(row[col])
                neck_rtt[8].append(row[12])
                neck_rtt[9].append(row[13])
                neck_rtt[10].append(row[14])
                j += 1

        elif i == 9:  # Head angular velocity
            head_avl = [[] for _ in range(5)]
            while j <= len(fileText2) - 1:
                row = fileText2[j]
                for k in range(5):
                    head_avl[k].append(row[k])
                j += 1

        elif i == 10:  # Head linear acceleration
            head_lac = [[] for _ in range(5)]
            while j <= len(fileText2) - 1:
                row = fileText2[j]
                for k in range(5):
                    head_lac[k].append(row[k])
                j += 1

    return (van_head_dvl, headCOM_cntfrc, helmetCOM_cntfrc, headCOM_van_cntfrc,
            neck_rtf, neck_rtt, headCOM_aac, helmetCOM_aac, van_head_rds,
            head_lvl, head_lac, head_avl, van_head_lvl, bicycle_cntfrc,
            bicycle_van_lvl, neck_head_aps, torso_head_aps)


# =============================================================================
# Result processing
# =============================================================================

def ProcessResults(impactData, timeHistory, helmetMOI, closestPoint,
                   van_head_dvl, headCOM_cntfrc, helmetCOM_cntfrc,
                   headCOM_van_cntfrc, neck_rtf, neck_rtt, headCOM_aac,
                   helmetCOM_aac, van_head_rds, head_lvl, head_lac,
                   van_head_lvl, bicycle_cntfrc, bicycle_van_lvl,
                   neck_head_aps, torso_head_aps):
    """
    Extract peak impact metrics from parsed MADYMO time-history data
    and append a row to impactData.
    """

    # -------------------------------------------------------------------------
    # Bicycle-van impact onset
    # -------------------------------------------------------------------------
    i_peaks, _ = scipy.signal.find_peaks(bicycle_cntfrc[1], distance=5, prominence=50)
    i_peaks = i_peaks.tolist()

    if i_peaks:
        i_peak_bicycle = i_peaks[0]
        i_impact_bicycle = i_peaks[0]
        while bicycle_cntfrc[1][i_impact_bicycle] >= bicycle_cntfrc[1][i_peak_bicycle] * 0.1:
            i_impact_bicycle -= 1
        bicycle_van_Fres = bicycle_cntfrc[1][i_peak_bicycle]
    else:
        i_peak_bicycle   = 0
        i_impact_bicycle = 0
        bicycle_van_Fres = 0

    # -------------------------------------------------------------------------
    # Helmet-van primary impact time and onset
    # height=100 filters spurious near-zero peaks before real contact
    # -------------------------------------------------------------------------
    i_peaks, _ = scipy.signal.find_peaks(
        helmetCOM_cntfrc[1], distance=5, prominence=50, height=100)
    i_peaks = i_peaks.tolist()

    if i_peaks:
        i_peak = i_peaks[0]
        for idx in range(1, len(i_peaks)):
            if (helmetCOM_cntfrc[0][i_peaks[idx]] - helmetCOM_cntfrc[0][i_peaks[0]] <= 30 and
                    helmetCOM_cntfrc[1][i_peaks[idx]] > helmetCOM_cntfrc[1][i_peak]):
                i_peak = i_peaks[idx]
        i_impact = i_peaks[0]
        while helmetCOM_cntfrc[1][i_impact] >= helmetCOM_cntfrc[1][i_peak] * 0.1:
            i_impact -= 1
    else:
        i_peak   = 0
        i_impact = 0

    t_peak   = helmetCOM_cntfrc[0][i_peak]
    t_impact = helmetCOM_cntfrc[0][i_impact]

    # Map impact time to equivalent indices in other arrays
    t_peak_impact = helmetCOM_cntfrc[0][i_peak]
    i_peak_neck = min(range(len(neck_rtf[0])),   key=lambda x: abs(neck_rtf[0][x]   - t_peak_impact))
    i_peak_lac  = min(range(len(head_lac[0])),   key=lambda x: abs(head_lac[0][x]   - t_peak_impact))
    i_peak_aac  = min(range(len(headCOM_aac[0])),key=lambda x: abs(headCOM_aac[0][x]- t_peak_impact))

    # -------------------------------------------------------------------------
    # Peak neck forces and moments within 50 ms post-impact
    # -------------------------------------------------------------------------
    i_peak_neck = GetPeakAfterImpact(neck_rtf[0], neck_rtf[1], i_peak_neck, timeWindow=50)

    i_peak_neckFx_pos = GetPeakAfterImpact(neck_rtf[0], neck_rtf[2], i_peak_neck, timeWindow=50)
    i_peak_neckFx_neg = GetPeakAfterImpact(neck_rtf[0], neck_rtf[2], i_peak_neck, timeWindow=50, direction="neg")
    i_peak_neckFy_pos = GetPeakAfterImpact(neck_rtf[0], neck_rtf[3], i_peak_neck, timeWindow=50)
    i_peak_neckFy_neg = GetPeakAfterImpact(neck_rtf[0], neck_rtf[3], i_peak_neck, timeWindow=50, direction="neg")
    i_peak_neckFz_pos = GetPeakAfterImpact(neck_rtf[0], neck_rtf[4], i_peak_neck, timeWindow=50)
    i_peak_neckFz_neg = GetPeakAfterImpact(neck_rtf[0], neck_rtf[4], i_peak_neck, timeWindow=50, direction="neg")
    i_peak_neckTx_pos = GetPeakAfterImpact(neck_rtf[0], neck_rtt[2], i_peak_neck, timeWindow=50)
    i_peak_neckTx_neg = GetPeakAfterImpact(neck_rtf[0], neck_rtt[2], i_peak_neck, timeWindow=50, direction="neg")
    i_peak_neckTy_pos = GetPeakAfterImpact(neck_rtf[0], neck_rtt[3], i_peak_neck, timeWindow=50)
    i_peak_neckTy_neg = GetPeakAfterImpact(neck_rtf[0], neck_rtt[3], i_peak_neck, timeWindow=50, direction="neg")
    i_peak_neckTz_pos = GetPeakAfterImpact(neck_rtf[0], neck_rtt[4], i_peak_neck, timeWindow=50)
    i_peak_neckTz_neg = GetPeakAfterImpact(neck_rtf[0], neck_rtt[4], i_peak_neck, timeWindow=50, direction="neg")

    # -------------------------------------------------------------------------
    # Peak head linear and angular acceleration within 50 ms post-impact
    # -------------------------------------------------------------------------
    i_peak_headAres  = GetPeakAfterImpact(head_lac[0],    head_lac[1],    i_peak_lac,  timeWindow=50)
    i_peak_headares  = GetPeakAfterImpact(headCOM_aac[0], headCOM_aac[1], i_peak_aac,  timeWindow=50)
    i_peak_headAx_pos = GetPeakAfterImpact(head_lac[0], head_lac[2], i_peak_lac, timeWindow=50)
    i_peak_headAx_neg = GetPeakAfterImpact(head_lac[0], head_lac[2], i_peak_lac, timeWindow=50, direction="neg")
    i_peak_headAy_pos = GetPeakAfterImpact(head_lac[0], head_lac[3], i_peak_lac, timeWindow=50)
    i_peak_headAy_neg = GetPeakAfterImpact(head_lac[0], head_lac[3], i_peak_lac, timeWindow=50, direction="neg")
    i_peak_headAz_pos = GetPeakAfterImpact(head_lac[0], head_lac[4], i_peak_lac, timeWindow=50)
    i_peak_headAz_neg = GetPeakAfterImpact(head_lac[0], head_lac[4], i_peak_lac, timeWindow=50, direction="neg")
    i_peak_headax_pos = GetPeakAfterImpact(headCOM_aac[0], headCOM_aac[2], i_peak_aac, timeWindow=50)
    i_peak_headax_neg = GetPeakAfterImpact(headCOM_aac[0], headCOM_aac[2], i_peak_aac, timeWindow=50, direction="neg")
    i_peak_headay_pos = GetPeakAfterImpact(headCOM_aac[0], headCOM_aac[3], i_peak_aac, timeWindow=50)
    i_peak_headay_neg = GetPeakAfterImpact(headCOM_aac[0], headCOM_aac[3], i_peak_aac, timeWindow=50, direction="neg")
    i_peak_headaz_pos = GetPeakAfterImpact(headCOM_aac[0], headCOM_aac[4], i_peak_aac, timeWindow=50)
    i_peak_headaz_neg = GetPeakAfterImpact(headCOM_aac[0], headCOM_aac[4], i_peak_aac, timeWindow=50, direction="neg")

    # -------------------------------------------------------------------------
    # Peak head contact force within 10 ms post-impact
    # -------------------------------------------------------------------------
    i_peak_headVan = GetPeakAfterImpact(headCOM_van_cntfrc[0], headCOM_van_cntfrc[1], i_peak, timeWindow=10)

    # Combined head force = helmet contact + neck reaction on head
    Fres_headCOM_list = []
    Tres_headCOM_list = []
    for k in range(len(headCOM_cntfrc[1])):
        Fres_headCOM_list.append(math.sqrt(
            (headCOM_cntfrc[2][k] + neck_rtf[8][k])**2 +
            (headCOM_cntfrc[3][k] + neck_rtf[9][k])**2 +
            (headCOM_cntfrc[4][k] + neck_rtf[10][k])**2))
        Tres_headCOM_list.append(math.sqrt(
            (headCOM_cntfrc[6][k] + neck_rtt[8][k])**2 +
            (headCOM_cntfrc[7][k] + neck_rtt[9][k])**2 +
            (headCOM_cntfrc[8][k] + neck_rtt[10][k])**2))

    i_peak_head = GetPeakAfterImpact(headCOM_cntfrc[0], Fres_headCOM_list, i_peak, timeWindow=10)

    # -------------------------------------------------------------------------
    # Extract values at peak indices
    # -------------------------------------------------------------------------

    # Neck forces
    Fres_neck = neck_rtf[1][i_peak_neck]
    Fx_neck   = neck_rtf[2][i_peak_neck]
    Fy_neck   = neck_rtf[3][i_peak_neck]
    Fz_neck   = neck_rtf[4][i_peak_neck]
    Tres_neck = neck_rtt[1][i_peak_neck]
    Tx_neck   = neck_rtt[2][i_peak_neck]
    Ty_neck   = neck_rtt[3][i_peak_neck]
    Tz_neck   = neck_rtt[4][i_peak_neck]

    Fx_neck_pos = neck_rtf[2][i_peak_neckFx_pos]
    Fy_neck_pos = neck_rtf[3][i_peak_neckFy_pos]
    Fz_neck_pos = neck_rtf[4][i_peak_neckFz_pos]
    Fx_neck_neg = -1 * neck_rtf[2][i_peak_neckFx_neg]
    Fy_neck_neg = -1 * neck_rtf[3][i_peak_neckFy_neg]
    Fz_neck_neg = -1 * neck_rtf[4][i_peak_neckFz_neg]

    Tx_neck_pos = neck_rtt[2][i_peak_neckTx_pos]
    Ty_neck_pos = neck_rtt[3][i_peak_neckTy_pos]
    Tz_neck_pos = neck_rtt[4][i_peak_neckTz_pos]
    Tx_neck_neg = -1 * neck_rtt[2][i_peak_neckTx_neg]
    Ty_neck_neg = -1 * neck_rtt[3][i_peak_neckTy_neg]
    Tz_neck_neg = -1 * neck_rtt[4][i_peak_neckTz_neg]

    Fx_neckExt = neck_rtf[2][i_peak_neckTy_neg]
    Fz_neckExt = neck_rtf[4][i_peak_neckTy_neg]
    Ty_neckExt = neck_rtt[3][i_peak_neckTy_neg]

    # Head accelerations
    Ares_headCOM = head_lac[1][i_peak_headAres]
    Ax_headCOM   = head_lac[2][i_peak_headAres]
    Ay_headCOM   = head_lac[3][i_peak_headAres]
    Az_headCOM   = head_lac[4][i_peak_headAres]
    ares_headCOM = headCOM_aac[1][i_peak_headares]
    ax_headCOM   = headCOM_aac[2][i_peak_headares]
    ay_headCOM   = headCOM_aac[3][i_peak_headares]
    az_headCOM   = headCOM_aac[4][i_peak_headares]

    Ax_headCOM_pos = head_lac[2][i_peak_headAx_pos]
    Ay_headCOM_pos = head_lac[3][i_peak_headAy_pos]
    Az_headCOM_pos = head_lac[4][i_peak_headAz_pos]
    Ax_headCOM_neg = -1 * head_lac[2][i_peak_headAx_neg]
    Ay_headCOM_neg = -1 * head_lac[3][i_peak_headAy_neg]
    Az_headCOM_neg = -1 * head_lac[4][i_peak_headAz_neg]
    ax_headCOM_pos = headCOM_aac[2][i_peak_headax_pos]
    ay_headCOM_pos = headCOM_aac[3][i_peak_headay_pos]
    az_headCOM_pos = headCOM_aac[4][i_peak_headaz_pos]
    ax_headCOM_neg = -1 * headCOM_aac[2][i_peak_headax_neg]
    ay_headCOM_neg = -1 * headCOM_aac[3][i_peak_headay_neg]
    az_headCOM_neg = -1 * headCOM_aac[4][i_peak_headaz_neg]

    # Helmet contact forces at peak
    Fres_helmetCOM = helmetCOM_cntfrc[1][i_peak]
    Fx_helmetCOM   = helmetCOM_cntfrc[2][i_peak]
    Fy_helmetCOM   = helmetCOM_cntfrc[3][i_peak]
    Fz_helmetCOM   = helmetCOM_cntfrc[4][i_peak]
    Tx_helmetCOM   = helmetMOI[0] * helmetCOM_aac[1][i_peak] + headCOM_cntfrc[6][i_peak]
    Ty_helmetCOM   = helmetMOI[1] * helmetCOM_aac[2][i_peak] + headCOM_cntfrc[7][i_peak]
    Tz_helmetCOM   = helmetMOI[2] * helmetCOM_aac[3][i_peak] + headCOM_cntfrc[8][i_peak]

    # Head-van contact forces
    Fres_headCOM_van = headCOM_van_cntfrc[1][i_peak_headVan]
    Fx_headCOM_van   = headCOM_van_cntfrc[2][i_peak_headVan]
    Fy_headCOM_van   = headCOM_van_cntfrc[3][i_peak_headVan]
    Fz_headCOM_van   = headCOM_van_cntfrc[4][i_peak_headVan]
    Tx_headCOM_van   = headCOM_van_cntfrc[6][i_peak_headVan]
    Ty_headCOM_van   = headCOM_van_cntfrc[7][i_peak_headVan]
    Tz_headCOM_van   = headCOM_van_cntfrc[8][i_peak_headVan]

    # Helmet-head contact forces (no neck)
    Fres_headCOM_noNeck = headCOM_cntfrc[1][i_peak_head]
    Fx_headCOM_noNeck   = headCOM_cntfrc[2][i_peak_head]
    Fy_headCOM_noNeck   = headCOM_cntfrc[3][i_peak_head]
    Fz_headCOM_noNeck   = headCOM_cntfrc[4][i_peak_head]
    Tres_headCOM_noNeck = headCOM_cntfrc[5][i_peak_head]
    Tx_headCOM_noNeck   = headCOM_cntfrc[6][i_peak_head]
    Ty_headCOM_noNeck   = headCOM_cntfrc[7][i_peak_head]
    Tz_headCOM_noNeck   = headCOM_cntfrc[8][i_peak_head]

    # Combined head forces (helmet contact + neck reaction)
    Fres_headCOM = Fres_headCOM_list[i_peak_head]
    Fx_headCOM   = headCOM_cntfrc[2][i_peak_head] + neck_rtf[8][i_peak_head]
    Fy_headCOM   = headCOM_cntfrc[3][i_peak_head] + neck_rtf[9][i_peak_head]
    Fz_headCOM   = headCOM_cntfrc[4][i_peak_head] + neck_rtf[10][i_peak_head]
    Tres_headCOM = Tres_headCOM_list[i_peak_head]
    Tx_headCOM   = headCOM_cntfrc[6][i_peak_head] + neck_rtt[8][i_peak_head]
    Ty_headCOM   = headCOM_cntfrc[7][i_peak_head] + neck_rtt[9][i_peak_head]
    Tz_headCOM   = headCOM_cntfrc[8][i_peak_head] + neck_rtt[10][i_peak_head]

    # -------------------------------------------------------------------------
    # Impact surface normal vector (from 3 closest reference points on van)
    # -------------------------------------------------------------------------
    dres_impact = [1000]
    i_Ps = []

    for k in range(1, len(van_head_rds)):
        dres_impact.append(van_head_rds[k][0][i_impact])

    if closestPoint >= 1:
        validImpact = 1 if van_head_rds[closestPoint][0][i_peak] <= 1.5 * 2 * 0.15 else 0
    elif closestPoint == -1:
        validImpact = 1

    for _ in range(3):
        idx = dres_impact.index(min(dres_impact))
        i_Ps.append(idx)
        dres_impact[idx] = max(dres_impact) + 1

    V12 = [van_head_rds[i_Ps[1]][m][i_impact] - van_head_rds[i_Ps[0]][m][i_impact] for m in range(1, 4)]
    V13 = [van_head_rds[i_Ps[2]][m][i_impact] - van_head_rds[i_Ps[0]][m][i_impact] for m in range(1, 4)]
    V12 = NormaliseVector(V12)
    V13 = NormaliseVector(V13)
    N_surf = np.cross(V12, V13)

    V1Head = [0 - van_head_rds[i_Ps[0]][m][i_impact] for m in range(1, 4)]
    N_surf = NormaliseVector(N_surf)
    V1Head = NormaliseVector(V1Head)

    if AngleBetweenVectors(V1Head, N_surf) > 90:
        N_surf = [-x for x in N_surf]

    # -------------------------------------------------------------------------
    # Bicycle-van relative velocity at impact onset
    # -------------------------------------------------------------------------
    bicycle_van_velRes = bicycle_van_lvl[1][i_impact_bicycle]

    # -------------------------------------------------------------------------
    # Neck and torso orientation in head frame at impact
    # -------------------------------------------------------------------------
    neck_head_Impaps  = [neck_head_aps[k][i_peak] * 57.3  for k in range(1, 4)]
    torso_head_Impaps = [torso_head_aps[k][i_peak] * 57.3 for k in range(1, 4)]

    # -------------------------------------------------------------------------
    # Head-van relative velocity vector and impact angle
    # -------------------------------------------------------------------------
    V_vel = [0 - van_head_lvl[k][i_impact] for k in range(1, 4)]
    velRes    = math.sqrt(sum(v**2 for v in V_vel))
    velAngle_N = AngleBetweenVectors(N_surf, NormaliseVector(V_vel))
    velAngle_T = velAngle_N - 90

    # -------------------------------------------------------------------------
    # Head orientation relative to impact surface (Euler angles for anvil test)
    # -------------------------------------------------------------------------
    V_surfy = NormaliseVector(np.cross(N_surf, NormaliseVector(V_vel)))
    V_surfx = NormaliseVector(np.cross(V_surfy, N_surf))
    P_surf  = [van_head_rds[i_Ps[0]][m][i_impact] for m in range(1, 4)]

    rotMat_head_N_surf  = np.column_stack((V_surfx, V_surfy, N_surf))
    transMat_head_N_surf = TransformationMatrix(rotMat_head_N_surf, P_surf)
    transMat_N_surf_head = np.linalg.inv(transMat_head_N_surf)
    rotMat_lab_N_surf    = RotationMatrix(0, (90 - velAngle_T) / 57.3, 0)
    transMat_lab_N_surf  = TransformationMatrix(rotMat_lab_N_surf, [0, 0, 0])
    transMat_lab_head    = np.dot(transMat_lab_N_surf, transMat_N_surf_head)
    alpha, beta, gamma   = TransMat2EulerZYX(transMat_lab_head)

    # Head-van closest point relative speed at impact
    van_head_dvl_impact = van_head_dvl[i_Ps[0]][i_impact]

    # -------------------------------------------------------------------------
    # HIC15 and contiguous 3 ms clip
    # -------------------------------------------------------------------------
    hic, hic_t1, hic_t2 = GetHIC(head_lac[0], [x / 9.81 for x in head_lac[1]], t_window=0.015)
    contiguous3ms, c3ms_t1, c3ms_t2 = GetContiguous3ms(
        head_lac[0], [x / 9.81 for x in head_lac[1]], t_window=0.003)

    # -------------------------------------------------------------------------
    # Collect results
    # -------------------------------------------------------------------------
    impactData.append([
        t_peak, t_impact,
        bicycle_van_velRes, velRes, van_head_dvl_impact, velAngle_T,
        alpha, beta, gamma,
        neck_head_Impaps[0], neck_head_Impaps[1], neck_head_Impaps[2],
        torso_head_Impaps[0], torso_head_Impaps[1], torso_head_Impaps[2],
        Ares_headCOM, Ax_headCOM, Ay_headCOM, Az_headCOM,
        ares_headCOM, ax_headCOM, ay_headCOM, az_headCOM,
        Ax_headCOM_pos, Ax_headCOM_neg, Ay_headCOM_pos, Ay_headCOM_neg,
        Az_headCOM_pos, Az_headCOM_neg,
        ax_headCOM_pos, ax_headCOM_neg, ay_headCOM_pos, ay_headCOM_neg,
        az_headCOM_pos, az_headCOM_neg,
        Fres_helmetCOM, Fx_helmetCOM, Fy_helmetCOM, Fz_helmetCOM,
        Fres_headCOM_van, Fx_headCOM_van, Fy_headCOM_van, Fz_headCOM_van,
        Tx_headCOM_van, Ty_headCOM_van, Tz_headCOM_van,
        Fres_headCOM_noNeck, Fx_headCOM_noNeck, Fy_headCOM_noNeck, Fz_headCOM_noNeck,
        Tx_headCOM_noNeck, Ty_headCOM_noNeck, Tz_headCOM_noNeck,
        Fres_headCOM, Fx_headCOM, Fy_headCOM, Fz_headCOM,
        Tx_headCOM, Ty_headCOM, Tz_headCOM,
        Fres_neck, Fx_neck, Fy_neck, Fz_neck,
        Tres_neck, Tx_neck, Ty_neck, Tz_neck,
        Fx_neck_pos, Fx_neck_neg, Fy_neck_pos, Fy_neck_neg,
        Fz_neck_pos, Fz_neck_neg,
        Tx_neck_pos, Tx_neck_neg, Ty_neck_pos, Ty_neck_neg,
        Tz_neck_pos, Tz_neck_neg,
        Fx_neckExt, Fz_neckExt, Ty_neckExt,
        hic, hic_t1, hic_t2, contiguous3ms, c3ms_t1, c3ms_t2,
        validImpact
    ])
    timeHistory.append(
        helmetCOM_cntfrc[0:5] + headCOM_van_cntfrc[1:5] +
        headCOM_cntfrc[1:5] + neck_rtf[1:5] + neck_rtt[1:5]
    )

    return impactData, timeHistory


# =============================================================================
# Save and move utilities
# =============================================================================

def SaveResults(impactData_header, impactData, impactData_all,
                timeHistory, fileName_saveResults, sheetNames):
    """Save impact data to Excel."""
    pd.DataFrame(impactData, columns=impactData_header).to_excel(
        fileName_saveResults + '_impactData.xlsx', sheet_name='sheet1', index=False)

    df_all = pd.DataFrame(impactData_all, columns=impactData_header)
    if not os.path.isfile('allSens_impactData.xlsx'):
        df_all.to_excel('allSens_impactData.xlsx', sheet_name='sheetNew', index=False)
    else:
        with pd.ExcelWriter('allSens_impactData.xlsx', mode='a', if_sheet_exists='replace') as writer:
            df_all.to_excel(writer, sheet_name='sheetNew', index=False)


def MoveResults(fileName_results, filePath_src, filePath_dst):
    """Move MADYMO output files into a numbered results subfolder."""
    os.makedirs(filePath_dst, exist_ok=True)
    for fname in fileName_results:
        shutil.move(os.path.join(filePath_src, fname),
                    os.path.join(filePath_dst, fname))
