# Large cohorts

```python
from mipy import io as mio
joined, report = mio.match_ids(visits, clinical, on="patient_id", engine="duckdb")
long = mio.wide_to_long(wide, engine="polars")
wide = mio.long_to_wide(long, engine="duckdb")

from mipy.chunked import impute_chunked
X_done = impute_chunked(X, patient_codes, method="softImpute", chunk_rows=500_000)
```

`mipy.bigdata.run_engine_benchmark()` times these steps and their memory for
cohorts of 10,000 to 10 million patients and skips steps predicted to exceed a
time limit or the free memory. Chunked imputation fits the model once (on all
rows for means, on a subsample for softImpute and IterativeImputer) and applies
it patient block by patient block, so memory is bounded by the chunk size.
