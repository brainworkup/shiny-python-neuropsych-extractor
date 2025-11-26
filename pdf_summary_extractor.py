#!/usr/bin/env python3
"""
PDF Summary Sheet Extractor for Neuropsychological Reports

Extracts test scores from PDF summary sheets (Word tables converted to PDF)
and outputs neurocog.csv and neurobehav.csv files compatible with neuro2.
"""

import pdfplumber
import pandas as pd
import numpy as np
from scipy import stats
import re
from pathlib import Path


def classify_range(pct):
    """Classify percentile into performance range category."""
    if pd.isna(pct):
        return None
    try:
        p = float(pct)
    except:
        return None
    if p >= 98: return "Exceptionally High"
    elif p >= 91: return "Above Average"
    elif p >= 75: return "High Average"
    elif p >= 25: return "Average"
    elif p >= 9: return "Low Average"
    elif p >= 2: return "Below Average"
    else: return "Exceptionally Low"


def compute_percentile(score, score_type):
    """Compute percentile from score and score type."""
    params = {
        "scaled_score": {"mu": 10, "sd": 3},
        "standard_score": {"mu": 100, "sd": 15},
        "t_score": {"mu": 50, "sd": 10},
    }
    if score_type not in params or pd.isna(score):
        return np.nan
    mu, sd = params[score_type]["mu"], params[score_type]["sd"]
    z = (score - mu) / sd
    return round(stats.norm.cdf(z) * 100, 1)


def parse_score_string(score_str):
    """Parse score string like 'ss=10', 'T=49', or plain number."""
    if not score_str or score_str in ['-', '—', '', 'None']:
        return None, None
    
    score_str = str(score_str).strip()
    
    # Scaled score: ss=10
    ss_match = re.match(r'ss=(\d+)', score_str, re.IGNORECASE)
    if ss_match:
        return int(ss_match.group(1)), 'scaled_score'
    
    # T-score: T=49
    t_match = re.match(r'T=(\d+)', score_str, re.IGNORECASE)
    if t_match:
        return int(t_match.group(1)), 't_score'
    
    # Plain number
    num_match = re.match(r'^(\d+\.?\d*)$', score_str)
    if num_match:
        val = float(num_match.group(1))
        # Determine type by value range
        if val >= 70:
            return val, 'standard_score'
        elif val >= 30:
            return val, 't_score'
        else:
            return val, 'scaled_score'
    
    return None, None


def parse_percentile_string(pct_str):
    """Parse percentile string handling ranges and special formats."""
    if not pct_str or pct_str in ['-', '—', '', 'None']:
        return None
    
    pct_str = str(pct_str).strip()
    
    # Plain number
    if re.match(r'^\d+\.?\d*$', pct_str):
        return float(pct_str)
    
    # Range like "51-75" - take midpoint
    range_match = re.match(r'^(\d+)-(\d+)$', pct_str)
    if range_match:
        return (float(range_match.group(1)) + float(range_match.group(2))) / 2
    
    # Cumulative percentile like "≤25" or ">16"
    cum_match = re.match(r'^[≤<>]+(\d+)', pct_str)
    if cum_match:
        return float(cum_match.group(1))
    
    # Base rate like "100 BR"
    br_match = re.match(r'^(\d+)\s*BR', pct_str)
    if br_match:
        return float(br_match.group(1))
    
    return None


class PDFSummaryExtractor:
    """Extract neuropsychological data from PDF summary sheets."""
    
    # Section headers that indicate domain changes
    SECTION_HEADERS = {
        'VALIDITY': 'Validity',
        'ESTIMATED PREMORBID': 'Premorbid Estimate', 
        'INTELLECTUAL FUNCTIONING': 'General Cognitive Ability',
        'ATTENTION/WORKING MEMORY': 'Attention/Executive',
        'PROCESSING SPEED': 'Attention/Executive',
        'LANGUAGE': 'Verbal/Language',
        'EXECUTIVE FUNCTIONING': 'Attention/Executive',
        'VISUOSPATIAL': 'Visual Perception/Construction',
        'MEMORY': 'Memory',
        'PSYCHIATRIC': 'Emotional/Behavioral/Personality',
        'BEHAVIORAL FUNCTIONING': 'Emotional/Behavioral/Personality',
    }
    
    # Test battery headers
    TEST_HEADERS = {
        'WAIS-IV': ('wais4', 'WAIS-IV', 'npsych_test'),
        'WAIS-V': ('wais5', 'WAIS-5', 'npsych_test'),
        'WMS-IV': ('wms4', 'WMS-IV', 'npsych_test'),
        'Wechsler Memory Scale IV': ('wms4', 'WMS-IV', 'npsych_test'),
        'WRAT-5': ('wrat5', 'WRAT-5', 'npsych_test'),
        'DKEFS CWIT': ('dkefs', 'D-KEFS Color-Word Interference', 'npsych_test'),
        'DKEFS VERBAL FLUENCY': ('dkefs', 'D-KEFS Verbal Fluency', 'npsych_test'),
        'D-KEFS': ('dkefs', 'D-KEFS', 'npsych_test'),
        'NAB': ('nab', 'NAB', 'npsych_test'),
        'CVLT 3 Brief': ('cvlt3', 'CVLT-3 Brief', 'npsych_test'),
        'CVLT-3': ('cvlt3', 'CVLT-3', 'npsych_test'),
        'TRAIL MAKING TEST': ('tmt', 'Trail Making Test', 'npsych_test'),
        'WCST': ('wcst', 'WCST', 'npsych_test'),
        'Clock Drawing': ('cdt', 'Clock Drawing Test', 'npsych_test'),
        'Rey 15': ('rey15', 'Rey 15-Item Test', 'npsych_test'),
        'PAI Scale': ('pai', 'PAI', 'rating_scale'),
        'PAI': ('pai', 'PAI', 'rating_scale'),
        'BDI-II': ('bdi2', 'BDI-II', 'rating_scale'),
        'BAI': ('bai', 'BAI', 'rating_scale'),
        'DSM-CCS': ('dsm_ccs', 'DSM-5 Cross-Cutting', 'rating_scale'),
    }
    
    # Behavioral/psychiatric measures (go to neurobehav.csv)
    BEHAVIORAL_TESTS = {'pai', 'bdi2', 'bai', 'dsm_ccs', 'pcl', 'psqi', 'basc', 'brief'}
    
    # PAI scale abbreviations for direct detection
    PAI_SCALES = {'ICN', 'INF', 'NIM', 'PIM', 'SOM', 'ANX', 'ARD', 'DEP', 
                  'MAN', 'PAR', 'SCZ', 'BOR', 'ANT', 'ALC', 'DRG', 'AGG',
                  'SUI', 'STR', 'NON', 'RXR', 'DOM', 'WRM'}
    
    def __init__(self, pdf_path, lookup_path=None):
        self.pdf_path = Path(pdf_path)
        self.lookup_path = Path(lookup_path) if lookup_path else None
        self.lookup_df = None
        self.records = []
        
    def load_lookup_table(self):
        """Load the neuropsych lookup table."""
        if self.lookup_path and self.lookup_path.exists():
            self.lookup_df = pd.read_csv(self.lookup_path)
            print(f"Loaded lookup table: {len(self.lookup_df)} entries")
        
    def extract_text(self):
        """Extract full text from PDF."""
        full_text = ""
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        return full_text
    
    def parse_data_line(self, line, current_test, current_domain):
        """Parse a single data line and return record dict or None."""
        parts = line.split()
        if len(parts) < 3:
            return None
        
        # Check if first part is a PAI scale abbreviation
        if parts[0] in self.PAI_SCALES:
            # PAI format: SCALE RawScore TScore % DC Descriptor
            scale_name = parts[0]
            data_parts = parts[1:]
            
            raw_score = None
            score = None
            score_type = 't_score'
            percentile = None
            
            if len(data_parts) >= 1:
                try:
                    raw_score = int(data_parts[0])
                except:
                    pass
            if len(data_parts) >= 2:
                try:
                    score = float(data_parts[1])
                except:
                    pass
            
            # Compute percentile from T-score
            if score is not None:
                percentile = compute_percentile(score, 't_score')
            
            return {
                'scale': scale_name,
                'raw_score': raw_score,
                'score': score,
                'percentile': percentile,
                'range': classify_range(percentile) if percentile else None,
                'score_type': 't_score',
                'test': 'pai',
                'test_name': 'PAI',
                'test_type': 'rating_scale',
                'domain': 'Emotional/Behavioral/Personality',
            }
        
        # Find where numeric data starts
        numeric_start = None
        for i, part in enumerate(parts):
            # Check for numeric patterns
            if (re.match(r'^\d+\.?\d*$', part) or 
                re.match(r'^(ss|T)=\d+', part, re.IGNORECASE) or
                re.match(r'^\d+/\d+$', part)):  # Like 42/43 or 9/9
                numeric_start = i
                break
        
        if numeric_start is None or numeric_start == 0:
            return None
        
        scale_name = ' '.join(parts[:numeric_start])
        data_parts = parts[numeric_start:]
        
        # Skip if scale name is a test header
        if any(h in scale_name for h in ['WAIS-IV', 'WMS-IV', 'NAB', 'WCST', 'DKEFS']):
            return None
        
        # Parse data parts
        raw_score = None
        score = None
        score_type = None
        percentile = None
        descriptor = None
        
        # First part is usually raw score
        if len(data_parts) >= 1:
            raw_str = data_parts[0]
            if re.match(r'^\d+\.?\d*$', raw_str):
                try:
                    raw_score = float(raw_str)
                    if raw_score == int(raw_score):
                        raw_score = int(raw_score)
                except:
                    pass
            elif re.match(r'^\d+/\d+$', raw_str):  # Fraction like 42/43
                raw_score = raw_str
        
        # Second part is score (ss=X, T=X, or standard score)
        if len(data_parts) >= 2:
            score, score_type = parse_score_string(data_parts[1])
        
        # Third part is percentile
        if len(data_parts) >= 3:
            percentile = parse_percentile_string(data_parts[2])
        
        # Fourth+ parts may include DC code and descriptor
        if len(data_parts) >= 5:
            # Usually: raw score score percentile DC descriptor...
            descriptor = ' '.join(data_parts[4:])
        elif len(data_parts) >= 4:
            # Could be: raw score score percentile descriptor (no DC)
            if data_parts[3] not in ['A', 'AE', 'AERS', 'E', 'R', 'S']:
                descriptor = ' '.join(data_parts[3:])
        
        # Skip if no meaningful data
        if score is None and percentile is None:
            return None
        
        # Get range
        range_val = descriptor if descriptor else classify_range(percentile)
        
        # If percentile missing but score available, compute it
        if percentile is None and score is not None and score_type:
            percentile = compute_percentile(score, score_type)
            if percentile and not range_val:
                range_val = classify_range(percentile)
        
        test_id, test_name, test_type = current_test if current_test else (None, None, None)
        
        return {
            'scale': scale_name,
            'raw_score': raw_score,
            'score': score,
            'percentile': percentile,
            'range': range_val,
            'score_type': score_type,
            'test': test_id,
            'test_name': test_name,
            'test_type': test_type,
            'domain': current_domain,
        }
    
    def extract(self):
        """Main extraction method."""
        self.load_lookup_table()
        full_text = self.extract_text()
        lines = full_text.split('\n')
        
        current_domain = None
        current_test = None
        
        # Track seen scales to avoid duplicates
        seen_scales = set()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers
            line_upper = line.upper()
            for key, domain in self.SECTION_HEADERS.items():
                if key in line_upper:
                    current_domain = domain
                    # Reset test when entering new section
                    if 'PSYCHIATRIC' in key or 'BEHAVIORAL' in key:
                        current_test = None
                    break
            
            # Check for test headers
            for key, (test_id, test_name, test_type) in self.TEST_HEADERS.items():
                if key in line:
                    # Check if it's a header row (contains —)
                    if '—' in line or 'Raw Score' in line:
                        current_test = (test_id, test_name, test_type)
                        # Set domain for behavioral tests
                        if test_id in self.BEHAVIORAL_TESTS:
                            current_domain = 'Emotional/Behavioral/Personality'
                        break
                    # Or if it's the PAI Scale header specifically
                    if key == 'PAI Scale' and 'T Score' in line:
                        current_test = (test_id, test_name, test_type)
                        current_domain = 'Emotional/Behavioral/Personality'
                        break
                    # PAI scales appearing in data rows
                    if test_id == 'pai':
                        current_test = ('pai', 'PAI', 'rating_scale')
                        current_domain = 'Emotional/Behavioral/Personality'
            
            # Skip header rows
            if 'Raw Score' in line or 'Standard Score' in line:
                continue
            if line.startswith('—') or line == '—':
                continue
            if 'T Score' in line and '%' in line:  # PAI header
                continue
            
            # Parse data line
            record = self.parse_data_line(line, current_test, current_domain)
            if record:
                # Check for duplicates using scale+test as key
                key = (record['scale'], record['test'])
                if key not in seen_scales:
                    seen_scales.add(key)
                    self.records.append(record)
        
        return pd.DataFrame(self.records)
    
    def merge_with_lookup(self, df):
        """Merge extracted data with lookup table."""
        if self.lookup_df is None:
            return df
        
        # Clean scale names for matching
        df['scale_clean'] = df['scale'].str.lower().str.strip()
        lookup = self.lookup_df.copy()
        lookup['scale_clean'] = lookup['scale'].str.lower().str.strip()
        
        # Merge on scale and test
        merged = df.merge(
            lookup[['scale_clean', 'test', 'domain', 'subdomain', 'narrow', 
                    'pass', 'verbal', 'timed', 'test_type', 'score_type', 
                    'absort', 'description']],
            on=['scale_clean', 'test'],
            how='left',
            suffixes=('', '_lookup')
        )
        
        # Use lookup values where available
        for col in ['domain', 'subdomain', 'narrow', 'pass', 'verbal', 'timed', 
                    'test_type', 'score_type', 'absort', 'description']:
            lookup_col = f'{col}_lookup'
            if lookup_col in merged.columns:
                merged[col] = merged[lookup_col].fillna(merged.get(col, np.nan))
                merged.drop(columns=[lookup_col], inplace=True)
        
        merged.drop(columns=['scale_clean'], inplace=True)
        
        return merged
    
    def split_neurocog_neurobehav(self, df):
        """Split data into neurocog and neurobehav dataframes."""
        # Behavioral tests go to neurobehav
        is_behavioral = df['test'].isin(self.BEHAVIORAL_TESTS)
        
        neurobehav = df[is_behavioral].copy()
        neurocog = df[~is_behavioral].copy()
        
        return neurocog, neurobehav
    
    def generate_result_text(self, row):
        """Generate result text for a row."""
        scale = row['scale']
        range_val = row.get('range', '')
        percentile = row.get('percentile')
        
        if pd.isna(percentile) or not range_val:
            return ''
        
        pct = int(round(percentile))
        suffix = 'th'
        if pct % 10 == 1 and pct != 11:
            suffix = 'st'
        elif pct % 10 == 2 and pct != 12:
            suffix = 'nd'
        elif pct % 10 == 3 and pct != 13:
            suffix = 'rd'
        
        return (f"{scale} fell within the {range_val} range and ranked at the "
                f"{pct}{suffix} percentile, indicating performance as good as or "
                f"better than {pct}% of same-age peers from the general population.")
    
    def process(self, output_dir=None):
        """Full processing pipeline."""
        print(f"Extracting from: {self.pdf_path}")
        
        # Extract
        df = self.extract()
        print(f"Extracted {len(df)} records")
        
        # Merge with lookup
        df = self.merge_with_lookup(df)
        
        # Generate result text
        df['result'] = df.apply(self.generate_result_text, axis=1)
        
        # Add standard columns
        df['ci_95'] = np.nan
        df['filename'] = self.pdf_path.name
        
        # Split into neurocog/neurobehav
        neurocog, neurobehav = self.split_neurocog_neurobehav(df)
        
        # Reorder columns to match expected format
        column_order = [
            'scale', 'raw_score', 'score', 'percentile', 'range', 'ci_95',
            'test', 'test_name', 'domain', 'subdomain', 'narrow',
            'pass', 'verbal', 'timed', 'test_type', 'score_type',
            'absort', 'result', 'filename', 'description'
        ]
        
        for col in column_order:
            if col not in neurocog.columns:
                neurocog[col] = np.nan
            if col not in neurobehav.columns:
                neurobehav[col] = np.nan
        
        neurocog = neurocog[[c for c in column_order if c in neurocog.columns]]
        neurobehav = neurobehav[[c for c in column_order if c in neurobehav.columns]]
        
        # Save if output_dir provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            if len(neurocog) > 0:
                neurocog_path = output_path / 'neurocog.csv'
                neurocog.to_csv(neurocog_path, index=False)
                print(f"Saved neurocog.csv: {len(neurocog)} rows")
            
            if len(neurobehav) > 0:
                neurobehav_path = output_path / 'neurobehav.csv'
                neurobehav.to_csv(neurobehav_path, index=False)
                print(f"Saved neurobehav.csv: {len(neurobehav)} rows")
        
        return neurocog, neurobehav


def main():
    """Main entry point."""
    import sys
    
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/Summary_Table.pdf'
    lookup_path = sys.argv[2] if len(sys.argv) > 2 else '/mnt/user-data/uploads/neuropsych_lookup_table.csv'
    output_dir = sys.argv[3] if len(sys.argv) > 3 else '/mnt/user-data/outputs'
    
    extractor = PDFSummaryExtractor(pdf_path, lookup_path)
    neurocog, neurobehav = extractor.process(output_dir)
    
    print("\n=== NEUROCOG ===")
    print(neurocog[['scale', 'test', 'score', 'percentile', 'range', 'domain']].to_string())
    
    print("\n=== NEUROBEHAV ===")
    print(neurobehav[['scale', 'test', 'score', 'percentile', 'range', 'domain']].to_string())


if __name__ == '__main__':
    main()

app = App(app_ui, server)
