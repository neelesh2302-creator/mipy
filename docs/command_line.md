# Command line

Installing MIPY adds the command `mipy` (also available as `python -m mipy`).

```bash
mipy example example.csv                 # write the example cohort
mipy check my_data.csv                   # how MIPY reads the table; nothing imputed
mipy analyze my_data.csv --out results   # benchmark, impute, figures, report
mipy impute my_data.csv --method MICE-Long --m 5   # one method, no benchmark
mipy methods                             # the methods and their evidence
mipy --version
```

## mipy analyze

```
mipy analyze DATA [--id COL] [--time COL] [--features A,B | first:last]
                  [--clinical A,B | none] [--survival TIME EVENT | --no-survival]
                  [--out DIR] [--methods quick|standard|complete|all|A,B]
                  [--mechanisms MCAR,MAR,MNAR,LOD] [--repeats 3] [--rate 0.2]
                  [--strength 2] [--m 5] [--export auto|METHOD|none]
                  [--workers 1|auto] [--seed 2026] [--log2 auto|yes|no]
                  [--max-missing 0.9] [--sheet NAME] [--layout wide|long
                  --feature-col COL --value-col COL] [--fresh] [--open] [--quiet]
```

Example with the columns of the GSE65622 analysis files:

```bash
mipy analyze RFS_data.csv --id Patient_ID --time sampling_point --survival RFS event --out RFS_results --open
```

On Windows, a path with spaces needs quotes: `mipy analyze "C:\My data\cohort.xlsx"`.

The command prints each step, and at the end the best method per mechanism and
the path of the report. Errors in the input (a missing file or column) are
printed as one line starting with `mipy:`, with exit code 2.
