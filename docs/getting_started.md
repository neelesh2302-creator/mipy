# Getting started

## The one-call route

```python
import mipy

df = mipy.example_cohort()                 # or pd.read_csv("my_data.csv"), or a file path
res = mipy.analyze(df, out="example_results", methods="quick", repeats=1)
res.show()                                 # inline in Jupyter; text and file paths elsewhere
res.decision                               # best method under each mechanism
res.completed                              # the completed table
```

`notebooks/quickstart.ipynb` shows the output, figures included. See
[your_data.md](your_data.md) for the table layout and every option.

## Step by step, with simulated data

```python
import mipy
from mipy.datasets import simulate_longitudinal
from mipy.masking import Amputer

sim = simulate_longitudinal(n_patients=100, n_proteins=80, seed=1)
X, patient, time = sim["X"], sim["patient"], sim["time"]

amp = Amputer(X, time=time, rate=0.2).ampute("MNAR", seed=7)
ctx = mipy.ImputeContext(patient=patient, time=time,
                         schema=mipy.build_variable_schema(amp["X"]))
for name in ["Mean", "softImpute", "MICE-Long", "MICE-LongMNAR"]:
    out, draws = mipy.run(name, amp["X"], ctx)
    print(name, mipy.evaluate(X, out, amp["mask"], patient=patient))
```

With a data frame (rows = patient-visits):

```python
done = mipy.impute(df, method="MICE-Long", id_col="Patient_ID",
                   time_col="sampling_point", log2=True)
```

The notebook `notebooks/getting_started.ipynb` runs these steps with output.
