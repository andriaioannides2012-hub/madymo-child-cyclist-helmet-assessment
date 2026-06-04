# -*- coding: utf-8 -*-
"""
extract_results.py

Extract impact metrics from completed MADYMO simulation runs without
re-running the solver. Reads CSV output files from each simResults subfolder,
processes them through simulation_functions.py, and saves peak metrics to Excel.

Usage
-----
1. Set simName, sheetNames and file paths below to match your simulation.
2. Run from within the Extracting_Results folder so simulation_functions.py
   is importable.
3. Omit any run numbers that crashed or produced invalid results from sheetNames.

Author: Andria Elia, Imperial College London, 2026.
Adapted from original code by Lucas Lacroix (ll6115), Imperial College London.
"""

import os
from simulation_functions import (
    ExtractResults, ProcessResults, SaveResults
)

# =============================================================================
# Settings — update to match your simulation
# =============================================================================

simName    = '26_05_FINAL_SEMI_LONG_SIMULATION'

# List only the run numbers you want to extract (omit crashed/invalid runs)
sheetNames = [
    '1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9',  '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
    '21', '22', '23', '25', '26', '27', '28', '30',
    '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
    '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
    '51', '52', '53', '54', '55', '56', '57', '58',
]

filePath_simResults  = ('C:/Users/andri/OneDrive/Imperial/Year_4/Masters Project/'
                        + simName + '/simResults')
filePath_saveResults = 'C:/Users/andri/OneDrive/Imperial/Year_4/Masters Project/Results'
fileName_saveResults = simName + '_sens'

# MADYMO output CSV filenames (must match TIME_HISTORY FILENAME in XML)
fileName_results = [
    'MadymoOutput_dvl.csv',
    'MadymoOutput_cntfrc.csv',
    'MadymoOutput_rtf.csv',
    'MadymoOutput_aps.csv',
    'MadymoOutput_lps.csv',
    'MadymoOutput_aac.csv',
    'MadymoOutput_rds.csv',
    'MadymoOutput_lvl.csv',
    'MadymoOutput_rtt.csv',
    'MadymoOutput_avl.csv',
    'MadymoOutput_lac.csv',
]

# Helmet moments of inertia [kg·m²] — Ixx, Iyy, Izz (solid ellipsoid, 0.26 kg)
helmetMOI = [0.001787, 0.002647, 0.002935]

# Index (1–9) of the predefined van impact reference point closest to the
# real crash contact location. Set to -1 for a ground impact (always valid).
closestPoint = 1

# Column headers for output Excel file
impactData_header = [
    't impact peak (ms)', 't impact onset (ms)',
    'Bicycle impact vres (m/s)', 'Head impact vres (m/s)',
    'Head impact speed (m/s)', 'headVres-surface angle (deg)',
    'Head Euler Z rot (rad)', 'Head Euler Y rot (rad)', 'Head Euler X rot (rad)',
    'Neck-head rx (deg)', 'Neck-head ry (deg)', 'Neck-head rz (deg)',
    'Torso-head rx (deg)', 'Torso-head ry (deg)', 'Torso-head rz (deg)',
    'Head Ares (m/s^2)', 'Head Ax (m/s^2)', 'Head Ay (m/s^2)', 'Head Az (m/s^2)',
    'Head ares (rad/s^2)', 'Head ax (rad/s^2)', 'Head ay (rad/s^2)', 'Head az (rad/s^2)',
    'Head +Ax (m/s^2)', 'Head -Ax (m/s^2)', 'Head +Ay (m/s^2)', 'Head -Ay (m/s^2)',
    'Head +Az (m/s^2)', 'Head -Az (m/s^2)',
    'Head +ax (rad/s^2)', 'Head -ax (rad/s^2)', 'Head +ay (rad/s^2)', 'Head -ay (rad/s^2)',
    'Head +az (rad/s^2)', 'Head -az (rad/s^2)',
    'Helm Fres (N)', 'Helm Fx (N)', 'Helm Fy (N)', 'Helm Fz (N)',
    'Head-Van Fres (N)', 'Head-Van Fx (N)', 'Head-Van Fy (N)', 'Head-Van Fz (N)',
    'Head-Van Tx (Nm)', 'Head-Van Ty (Nm)', 'Head-Van Tz (Nm)',
    'HeadHelm Fres (N)', 'HeadHelm Fx (N)', 'HeadHelm Fy (N)', 'HeadHelm Fz (N)',
    'HeadHelm Tx (Nm)', 'HeadHelm Ty (Nm)', 'HeadHelm Tz (Nm)',
    'Head Fres (N)', 'Head Fx (N)', 'Head Fy (N)', 'Head Fz (N)',
    'Head Tx (Nm)', 'Head Ty (Nm)', 'Head Tz (Nm)',
    'Neck Fres (N)', 'Neck Fx (N)', 'Neck Fy (N)', 'Neck Fz (N)',
    'Neck Tres (Nm)', 'Neck Tx (Nm)', 'Neck Ty (Nm)', 'Neck Tz (Nm)',
    'Neck +Fx (N)', 'Neck -Fx (N)', 'Neck +Fy (N)', 'Neck -Fy (N)',
    'Neck +Fz (N)', 'Neck -Fz (N)',
    'Neck +Tx (Nm)', 'Neck -Tx (Nm)', 'Neck +Ty (Nm)', 'Neck -Ty (Nm)',
    'Neck +Tz (Nm)', 'Neck -Tz (Nm)',
    'Neck Fx at peak neck extension (N)', 'Neck Fz at peak neck extension (N)',
    'Peak neck extension (Nm)',
    'HIC15', 'HIC15 t1 (s)', 'HIC15 t2 (s)',
    'Contiguous 3ms (g)', 'C3ms t1 (s)', 'C3ms t2 (s)',
    'Valid impact',
]

# =============================================================================
# Extract and save
# =============================================================================

impactData  = []
timeHistory = []

for sheet in sheetNames:
    filePath_run = os.path.join(filePath_simResults, sheet)
    os.chdir(filePath_run)
    print(f'Extracting run {sheet} ...')

    (van_head_dvl, headCOM_cntfrc, helmetCOM_cntfrc, headCOM_van_cntfrc,
     neck_rtf, neck_rtt, headCOM_aac, helmetCOM_aac, van_head_rds,
     head_lvl, head_lac, head_avl, van_head_lvl, bicycle_cntfrc,
     bicycle_van_lvl, neck_head_aps, torso_head_aps) = ExtractResults(fileName_results)

    impactData, timeHistory = ProcessResults(
        impactData, timeHistory, helmetMOI, closestPoint,
        van_head_dvl, headCOM_cntfrc, helmetCOM_cntfrc, headCOM_van_cntfrc,
        neck_rtf, neck_rtt, headCOM_aac, helmetCOM_aac, van_head_rds,
        head_lvl, head_lac, van_head_lvl, bicycle_cntfrc, bicycle_van_lvl,
        neck_head_aps, torso_head_aps)

    print(f'Done run {sheet}')

os.chdir(filePath_saveResults)
SaveResults(impactData_header, impactData, impactData,
            timeHistory, fileName_saveResults, sheetNames)
print('Saved to', filePath_saveResults)
