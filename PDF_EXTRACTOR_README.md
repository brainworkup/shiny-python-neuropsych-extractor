# Neuropsych PDF Summary Sheet Extractor

Extracts neuropsychological test data from PDF summary sheets (Word tables converted to PDF) and outputs `neurocog.csv` and `neurobehav.csv` files compatible with the neuro2 workflow.

## Features

- **PDF Parsing**: Extracts data from structured neuropsych summary sheets
- **Lookup Table Merge**: Automatically enriches data with metadata (domain, subdomain, description)
- **Percentile Computation**: Calculates missing percentiles from standardized scores
- **Performance Range Classification**: Assigns qualitative descriptors (Average, High Average, etc.)
- **Result Text Generation**: Creates clinical narrative text for each scale

## Supported Tests

### Cognitive (→ neurocog.csv)
- WAIS-IV/V (composites and subtests)
- WMS-IV (Logical Memory, Visual Reproduction)
- WRAT-5 (Word Reading)
- D-KEFS (Color-Word Interference, Verbal Fluency)
- NAB (Naming, Judgment)
- CVLT-3 Brief
- Trail Making Test
- WCST

### Behavioral (→ neurobehav.csv)
- PAI (all scales: ICN, INF, NIM, SOM, ANX, DEP, etc.)
- BDI-II, BAI
- BASC, BRIEF

## Usage

### Command Line
```bash
pip install pdfplumber pandas scipy

python pdf_summary_extractor.py <pdf_file> <lookup_table> <output_dir>

# Example:
python pdf_summary_extractor.py Summary_Table.pdf ~/Dropbox/neuropsych_lookup_table.csv ./output/
```

### Shiny App
```bash
pip install shiny pdfplumber pandas scipy

shiny run neuropsych_extractor_combined.py
```

Then open http://localhost:8000, upload your PDF, and download the extracted CSVs.

## Output Format

Both output files follow the neuro2 standard format:

| Column | Description |
|--------|-------------|
| scale | Test/subtest name |
| raw_score | Raw score value |
| score | Standardized score (SS, ss, or T) |
| percentile | Percentile rank |
| range | Qualitative descriptor |
| test | Test identifier (e.g., wais4, pai) |
| test_name | Full test name |
| domain | Clinical domain |
| subdomain | Subdomain classification |
| narrow | Narrow ability |
| score_type | scaled_score, standard_score, or t_score |
| result | Clinical narrative text |

## Score Type Conversion

| Type | Mean | SD | Example |
|------|------|-----|---------|
| standard_score | 100 | 15 | WAIS-IV composites |
| scaled_score | 10 | 3 | WAIS-IV subtests |
| t_score | 50 | 10 | PAI, TMT |

## Performance Range Classification

| Percentile | Range |
|------------|-------|
| ≥98 | Exceptionally High |
| 91-97 | Above Average |
| 75-90 | High Average |
| 25-74 | Average |
| 9-24 | Low Average |
| 2-8 | Below Average |
| <2 | Exceptionally Low |
