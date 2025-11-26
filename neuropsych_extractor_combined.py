#!/usr/bin/env python3
"""
Neuropsych Data Extractor - Combined PDF/CSV Shiny App
Extracts neuropsychological data from PDF summary sheets and CSV assessment files.
"""

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from shiny import App, Inputs, Outputs, Session, reactive, render, ui

try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Utility functions
def classify_range(pct):
    if pd.isna(pct): return None
    p = float(pct)
    if p >= 98: return "Exceptionally High"
    elif p >= 91: return "Above Average"
    elif p >= 75: return "High Average"
    elif p >= 25: return "Average"
    elif p >= 9: return "Low Average"
    elif p >= 2: return "Below Average"
    else: return "Exceptionally Low"

def compute_percentile(score, score_type):
    params = {"scaled_score": (10, 3), "standard_score": (100, 15), "t_score": (50, 10)}
    if score_type not in params or pd.isna(score): return np.nan
    mu, sd = params[score_type]
    return round(stats.norm.cdf((score - mu) / sd) * 100, 1)

def parse_score_string(score_str):
    if not score_str or score_str in ['-', '—', '']: return None, None
    score_str = str(score_str).strip()
    if m := re.match(r'ss=(\d+)', score_str, re.I): return int(m.group(1)), 'scaled_score'
    if m := re.match(r'T=(\d+)', score_str, re.I): return int(m.group(1)), 't_score'
    if m := re.match(r'^(\d+\.?\d*)$', score_str):
        val = float(m.group(1))
        return (val, 'standard_score') if val >= 70 else (val, 't_score') if val >= 30 else (val, 'scaled_score')
    return None, None

def generate_result_text(row):
    scale, range_val, pct = row.get('scale', ''), row.get('range', ''), row.get('percentile')
    if pd.isna(pct) or not range_val: return ''
    p = int(round(pct))
    suffix = 'st' if p % 10 == 1 and p != 11 else 'nd' if p % 10 == 2 and p != 12 else 'rd' if p % 10 == 3 and p != 13 else 'th'
    return f"{scale} fell within the {range_val} range and ranked at the {p}{suffix} percentile."

# PDF extraction
SECTION_HEADERS = {'VALIDITY': 'Validity', 'ESTIMATED PREMORBID': 'Premorbid Estimate', 
    'INTELLECTUAL FUNCTIONING': 'General Cognitive Ability', 'ATTENTION/WORKING MEMORY': 'Attention/Executive',
    'LANGUAGE': 'Verbal/Language', 'EXECUTIVE FUNCTIONING': 'Attention/Executive',
    'VISUOSPATIAL': 'Visual Perception/Construction', 'MEMORY': 'Memory'}

TEST_HEADERS = {'WAIS-IV': ('wais4', 'WAIS-IV'), 'WMS-IV': ('wms4', 'WMS-IV'), 'WRAT-5': ('wrat5', 'WRAT-5'),
    'DKEFS CWIT': ('dkefs', 'D-KEFS CWI'), 'DKEFS VERBAL FLUENCY': ('dkefs', 'D-KEFS VF'),
    'NAB': ('nab', 'NAB'), 'CVLT 3 Brief': ('cvlt3', 'CVLT-3'), 'TRAIL MAKING TEST': ('tmt', 'TMT'), 'WCST': ('wcst', 'WCST')}

PAI_SCALES = {'ICN', 'INF', 'NIM', 'PIM', 'SOM', 'ANX', 'ARD', 'DEP', 'MAN', 'PAR', 'SCZ', 'BOR', 'ANT', 'ALC', 'DRG', 'AGG', 'SUI', 'STR', 'NON', 'RXR', 'DOM', 'WRM'}
BEHAVIORAL_TESTS = {'pai', 'bdi2', 'bai', 'basc', 'brief'}

def extract_pdf(file_content):
    if not PDF_SUPPORT: return pd.DataFrame()
    records, current_domain, current_test, seen = [], None, None, set()
    
    with pdfplumber.open(io.BytesIO(file_content)) as pdf:
        text = "".join(p.extract_text() or "" for p in pdf.pages)
    
    for line in text.split('\n'):
        line = line.strip()
        if not line: continue
        
        for k, d in SECTION_HEADERS.items():
            if k in line.upper(): current_domain = d; break
        for k, (tid, tname) in TEST_HEADERS.items():
            if k in line and '—' in line: current_test = (tid, tname); break
        
        if 'Raw Score' in line: continue
        parts = line.split()
        if len(parts) < 3: continue
        
        # PAI scales
        if parts[0] in PAI_SCALES:
            raw = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            score = float(parts[2]) if len(parts) > 2 and re.match(r'^\d+$', parts[2]) else None
            pct = compute_percentile(score, 't_score') if score else None
            if (parts[0], 'pai') not in seen:
                seen.add((parts[0], 'pai'))
                records.append({'scale': parts[0], 'raw_score': raw, 'score': score, 'percentile': pct,
                    'range': classify_range(pct), 'score_type': 't_score', 'test': 'pai', 'test_name': 'PAI',
                    'test_type': 'rating_scale', 'domain': 'Emotional/Behavioral/Personality'})
            continue
        
        # Find numeric start
        num_idx = next((i for i, p in enumerate(parts) if re.match(r'^\d|^(ss|T)=', p, re.I)), None)
        if not num_idx: continue
        
        scale = ' '.join(parts[:num_idx])
        if any(h in scale for h in TEST_HEADERS): continue
        
        data = parts[num_idx:]
        raw = float(data[0]) if data and re.match(r'^\d+\.?\d*$', data[0]) else None
        score, stype = parse_score_string(data[1]) if len(data) > 1 else (None, None)
        pct = float(data[2]) if len(data) > 2 and re.match(r'^\d+\.?\d*$', data[2]) else None
        
        if score is None and pct is None: continue
        if pct is None and score and stype: pct = compute_percentile(score, stype)
        
        tid, tname = current_test if current_test else (None, None)
        if (scale, tid) not in seen:
            seen.add((scale, tid))
            records.append({'scale': scale, 'raw_score': raw, 'score': score, 'percentile': pct,
                'range': classify_range(pct), 'score_type': stype, 'test': tid, 'test_name': tname,
                'test_type': 'npsych_test', 'domain': current_domain})
    
    return pd.DataFrame(records)

def merge_with_lookup(df, lookup_df):
    if lookup_df is None or len(df) == 0: return df
    df = df.copy()
    df['scale_clean'] = df['scale'].str.lower().str.strip()
    lookup = lookup_df.copy()
    lookup['scale_clean'] = lookup['scale'].str.lower().str.strip()
    
    merged = df.merge(lookup[['scale_clean', 'test', 'domain', 'subdomain', 'narrow', 'pass', 'verbal', 'timed', 'absort', 'description']],
        on=['scale_clean', 'test'], how='left', suffixes=('', '_lk'))
    
    for c in ['domain', 'subdomain', 'narrow', 'pass', 'verbal', 'timed', 'absort', 'description']:
        if f'{c}_lk' in merged.columns:
            merged[c] = merged[f'{c}_lk'].fillna(merged.get(c, np.nan))
            merged.drop(columns=[f'{c}_lk'], inplace=True, errors='ignore')
    merged.drop(columns=['scale_clean'], inplace=True, errors='ignore')
    return merged

def split_data(df):
    behav = df['test'].isin(BEHAVIORAL_TESTS)
    return df[~behav].copy(), df[behav].copy()

# Shiny App
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("📊 Neuropsych Extractor"),
        ui.input_file("data_file", "Upload PDF or CSV", accept=[".pdf", ".csv"]),
        ui.input_file("lookup_file", "Lookup Table (optional)", accept=[".csv"]),
        ui.hr(),
        ui.input_checkbox("compute_missing", "Compute missing percentiles", True),
        ui.download_button("download_all", "📥 Download CSVs"),
        ui.p(f"PDF: {'✅' if PDF_SUPPORT else '❌'}"), width=280,
    ),
    ui.navset_tab(
        ui.nav_panel("Neurocog", ui.output_data_frame("neurocog_tbl")),
        ui.nav_panel("Neurobehav", ui.output_data_frame("neurobehav_tbl")),
        ui.nav_panel("Summary", ui.output_ui("summary")),
    ),
    title="Neuropsych Data Extractor",
)

def server(input, output, session):
    @reactive.calc
    def lookup():
        if f := input.lookup_file(): return pd.read_csv(f[0]["datapath"])
        p = Path("/mnt/project/neuropsych_lookup_table.csv")
        return pd.read_csv(p) if p.exists() else None
    
    @reactive.calc
    def data():
        if not (f := input.data_file()): return pd.DataFrame()
        path, name = Path(f[0]["datapath"]), f[0]["name"]
        
        if name.lower().endswith('.pdf'):
            df = extract_pdf(path.read_bytes())
        else:
            df = pd.DataFrame()  # Add CSV parsing if needed
        
        if len(df) == 0: return df
        df = merge_with_lookup(df, lookup())
        
        if input.compute_missing():
            for i, r in df.iterrows():
                if pd.isna(r.get('percentile')) and not pd.isna(r.get('score')):
                    pct = compute_percentile(r['score'], r.get('score_type', 'scaled_score'))
                    df.at[i, 'percentile'] = pct
                    if pd.isna(r.get('range')): df.at[i, 'range'] = classify_range(pct)
        
        df['result'] = df.apply(generate_result_text, axis=1)
        return df
    
    @reactive.calc
    def split(): return split_data(data()) if len(data()) else (pd.DataFrame(), pd.DataFrame())
    
    @render.data_frame
    def neurocog_tbl():
        nc, _ = split()
        return render.DataGrid(nc[['scale', 'test', 'score', 'percentile', 'range', 'domain']], filters=True, height="500px") if len(nc) else pd.DataFrame()
    
    @render.data_frame
    def neurobehav_tbl():
        _, nb = split()
        return render.DataGrid(nb[['scale', 'test', 'score', 'percentile', 'range', 'domain']], filters=True, height="500px") if len(nb) else pd.DataFrame()
    
    @render.ui
    def summary():
        nc, nb = split()
        return ui.layout_column_wrap(
            ui.card(ui.card_header("Neurocog"), f"{len(nc)} scales"),
            ui.card(ui.card_header("Neurobehav"), f"{len(nb)} scales"),
            width=1/2
        )
    
    @render.download(filename="neuropsych_data.zip")
    def download_all():
        nc, nb = split()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as zf:
            if len(nc): zf.writestr("neurocog.csv", nc.to_csv(index=False))
            if len(nb): zf.writestr("neurobehav.csv", nb.to_csv(index=False))
        buf.seek(0)
        return buf.read()

app = App(app_ui, server)
