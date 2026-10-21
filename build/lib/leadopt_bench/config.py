"""Paths and series-selection thresholds.

Everything the pipeline downloads or produces stays under this package, so a run
is reproducible from an empty cache and nothing leaks into ~/.cache.
"""

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"
RAW_DIR = DATA_DIR / "raw"          # verbatim ChEMBL API pages (gzipped JSON)
SERIES_DIR = DATA_DIR / "series"    # one JSON per accepted series
REPORT_DIR = DATA_DIR / "reports"   # survey tables

for _d in (DATA_DIR, RAW_DIR, SERIES_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- ChEMBL API ---------------------------------------------------------------
API_BASE = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_LIMIT = 1000          # server maximum
REQUEST_TIMEOUT = 60
RETRIES = 4
SLEEP_BETWEEN_PAGES = 0.2  # be polite to EBI

# Only direct, single-protein assignments. This is the spec's confidence 9.
CONFIDENCE_SCORE = 9
ASSAY_TYPE = "B"           # binding; excludes cell-based/functional/ADMET panels
ACTIVITY_TYPES = ("IC50", "Ki")

# --- series acceptance criteria ----------------------------------------------
MIN_COMPOUNDS = 20
MIN_MEAN_TANIMOTO = 0.40   # congeneric, not a heap of unrelated hits
MIN_LOG_RANGE = 2.0        # pchembl span, in log units

# --- fingerprint --------------------------------------------------------------
FP_RADIUS = 2
FP_BITS = 2048
