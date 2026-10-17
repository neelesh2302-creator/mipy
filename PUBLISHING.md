# Publishing mipy-impute 2.1.0: A to Z

Eleven stages, in order. Each ends with a checkpoint: what you should see
before going on. If something else appears, stop there and send me the last
lines of the window.

Everything typed goes into the **Anaconda Prompt** (Start menu > Anaconda
Prompt). To paste into it, right-click or press Ctrl+V. Lines starting with
`REM` are comments; you do not need to type them.

Time: about 3 hours, half of it waiting for runs.

---

## Stage 0. Things you need once (15 min)

1. **Git for Windows.** In the Anaconda Prompt type `git --version`.
   * `git version 2.xx` means it is installed.
   * `'git' is not recognized` means it is not: download it from
     <https://git-scm.com/download/win>, install with every default option,
     then close and reopen the Anaconda Prompt and try again.
2. **An authenticator app on your phone**: Google Authenticator or Microsoft
   Authenticator. PyPI and TestPyPI will not let you upload without
   two-factor authentication.
3. **Internet without a blocking proxy.** If the campus network stops
   `pip` or `git` (SSL or proxy errors), use a phone hotspot for the upload
   stages.

**Checkpoint 0:** `git --version` prints a version; the app is on your phone.

---

## Stage 1. Unpack the files (10 min)

1. Download from the chat into `Downloads`: `mipy_package_v2.1.0.zip`,
   `PUBLISHING.md`, `example_report.html`, `MIPY_A_to_Z_one_cell.py`,
   `MIPY_A_to_Z.ipynb`. Use the zip from the latest message.
2. Open File Explorer at `C:\Users\DELL\Missing_Python`. If a folder
   `mipy_pkg` is already there, rename it `mipy_pkg_v2.0_old`. That keeps
   the old version as a backup and stops the two from mixing.
3. Right-click `mipy_package_v2.1.0.zip` > **Extract All...**. Windows
   suggests a path ending in `\mipy_package_v2.1.0`. Delete that part, so the
   box says exactly `C:\Users\DELL\Missing_Python`, then click Extract.
4. Check that this file exists:
   `C:\Users\DELL\Missing_Python\mipy_pkg\pyproject.toml`.
   If it ended up one folder deeper (`...\mipy_package_v2.1.0\mipy_pkg\...`),
   drag `mipy_pkg` up into `Missing_Python`.
5. Replace your old A to Z files with the two new ones
   (`MIPY_A_to_Z_one_cell.py`, `MIPY_A_to_Z.ipynb`), and delete the old
   copies. The old A to Z writes version 2.0 files into
   `Missing_Python\mipy_pkg\src\mipy`, which would overwrite the new package.
   The new A to Z writes the same files as the zip, so it is safe.
6. Make an empty folder for trial runs:
   `C:\Users\DELL\Missing_Python\mipy_trial`. Every test run happens there,
   so no result files end up on GitHub.

**Checkpoint 1:** `mipy_pkg\pyproject.toml` is at the right place, old A to Z
files are gone, `mipy_trial` exists.

---

## Stage 2. Install the package in `gse` and run its tests (10 min)

```bat
conda activate gse
cd /d C:\Users\DELL\Missing_Python\mipy_pkg
python -m pip install -e ".[test]"
python -c "import mipy; print(mipy.__version__, mipy.__file__)"
python -m pytest -q
```

What you should see:

* the pip line ends with `Successfully installed ... mipy-impute-2.1.0 ...`
  (it may add `openpyxl`);
* the import line prints `2.1.0` and a path inside
  `Missing_Python\mipy_pkg\src\mipy`;
* pytest runs for 1 to 3 minutes and its last line is about `71 passed`
  (or `70 passed, 1 skipped` if it cannot find R). Lines with "warning" are
  fine. Any `failed` or `error` is not.

If the import line shows `2.0.0`, the old package is still installed:
run `python -m pip uninstall -y mipy-impute`, then the `pip install -e`
line again.

**Checkpoint 2:** send me the pytest summary line if it is not all passed.

---

## Stage 3. Use it the way a stranger would (30 to 60 min)

```bat
cd /d C:\Users\DELL\Missing_Python\mipy_trial
mipy --version
mipy example ex.csv
mipy check ex.csv
mipy analyze ex.csv --methods quick --repeats 1 --open
```

The last command prints its progress for about a minute, then your browser
opens `mipy_results\report.html`. The folder `mipy_results` has `figures`,
`tables` and `completed_data`.

Now your own files (Part A of the A to Z copied them to
`Missing_Python\data`; the originals are in `N:\Research\MIPY\Data`):

```bat
mipy check "C:\Users\DELL\Missing_Python\data\RFS_data.csv"
```

Read what it prints. It should say: ID column `Patient_ID`, visit column
`sampling_point`, survival columns `RFS` / `event`, clinical columns
`diarrhea_grade, slide_print_batch, trg_score, ypn_stage, ypt_stage`,
about 10.5% missing, and "values analysed as log2(x + 1)". If it says
"values analysed as given" instead, add `--log2 yes` to the commands below,
because the paper used the log2 scale.

```bat
mipy analyze "C:\Users\DELL\Missing_Python\data\MFS_data.csv" --out MFS_results --workers auto --open
mipy analyze "C:\Users\DELL\Missing_Python\data\RFS_data.csv" --out RFS_results --workers auto --open
```

Each takes 10 to 20 minutes. For MFS the report has a survival section.
For RFS the report says the survival screen was skipped (5 events; it needs
10). That is correct behaviour, not an error.

Optional, in Jupyter (kernel `gse`), the same thing with figures in the
notebook:

```python
import mipy
res = mipy.analyze(r"C:\Users\DELL\Missing_Python\data\MFS_data.csv", out="MFS_nb")
res.decision
```

**Checkpoint 3:** the reports open and show figures. Send me the "Most
accurate method under each assumption" box from the MFS report if anything
looks odd.

---

## Stage 4. Make the accounts (30 min)

**GitHub.** <https://github.com/signup>, verify your email. Write down your
user name exactly as GitHub shows it (top right > your picture).

**PyPI.** <https://pypi.org/account/register/>, then:

1. verify your email (the link in PyPI's email);
2. Account settings > **Two factor authentication** > Add 2FA with
   authentication application > scan the QR code with the phone app > type
   the 6-digit code;
3. download the **recovery codes** and keep them somewhere safe; PyPI asks
   you to type one back to prove you saved them.

**TestPyPI.** <https://test.pypi.org/account/register/>. A separate site
with a separate account; repeat the same three steps there.

**Checkpoint 4:** you can log in to all three sites, and PyPI and TestPyPI
both show 2FA as on.

---

## Stage 5. Final checks on the files (10 min)

1. **GitHub user name in the files.** The files say `neelesh2302`. If your
   GitHub user name is exactly that, skip to 2. Otherwise open each of these
   three files in Notepad, press Ctrl+H, put `neelesh2302` in "Find what" and
   your user name in "Replace with", click **Replace All**, save:
   * `C:\Users\DELL\Missing_Python\mipy_pkg\pyproject.toml`
   * `C:\Users\DELL\Missing_Python\mipy_pkg\README.md`
   * `C:\Users\DELL\Missing_Python\mipy_pkg\docs\index.md`
2. **The name is still free.** Open <https://pypi.org/project/mipy-impute/>.
   It should say "404 Not Found". If a project is there, stop and tell me.
3. **No patient data inside the package.**
   `dir C:\Users\DELL\Missing_Python\mipy_pkg\src\mipy\data` should list only
   `README.txt`.
4. **Co-authors.** Dr. Bhattacharjee, Prof. Vishwakarma and Tanmoy Majumdar
   are listed as package authors. Get their go-ahead before Stage 9.

**Checkpoint 5:** user name set, PyPI shows 404 for `mipy-impute`, data
folder has only README.txt.

---

## Stage 6. Put the code on GitHub (20 min, plus 10 min of tests)

1. On github.com: **+** (top right) > **New repository**.
   * Repository name: `mipy`
   * Description: `Missing-data imputation and benchmarking for high-dimensional longitudinal data`
   * **Public**
   * Leave "Add a README file" **unticked**, .gitignore **None**, license
     **None** (the package has all three).
   * Create repository.
2. Tell Git who you are (once per computer). Use the email of your GitHub
   account:

```bat
git config --global user.name "Neelesh Kumar"
git config --global user.email "your-github-email@example.com"
```

3. Upload. Replace `YOUR-USERNAME` with your GitHub user name:

```bat
cd /d C:\Users\DELL\Missing_Python\mipy_pkg
git init
git add .
git status
```

   Read the `git status` list before going on. It shows about 55 lines
   starting with `new file:`, such as `.github/workflows/ci.yml`,
   `README.md`, `pyproject.toml`, `src/mipy/analysis.py`, `tests/...`,
   `docs/...` and `notebooks/...`. None of them may start with `dist/`, end
   in `.csv`, or sit in a `*_results` folder. If one does, stop and tell me.

```bat
git commit -m "MIPY 2.1.0"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/mipy.git
git push -u origin main
```

   The first push opens a window: choose **Sign in with your browser** and
   approve. Messages like "LF will be replaced by CRLF" are harmless. The
   push ends with `branch 'main' set up to track 'origin/main'`.
4. Refresh the repository page. The files are there and the README shows
   below them, with the accuracy figure.
5. Click the **Actions** tab. The workflow "tests" runs 8 jobs (Linux,
   Windows and macOS; Python 3.9, 3.11, 3.13). Wait about 10 minutes for
   green ticks. If one turns red, click it, open the red step and send me
   its last lines.

**Checkpoint 6:** README visible on GitHub; Actions all green.

---

## Stage 7. Build the files that go to PyPI (5 min)

You build them on your own computer so that they contain any change you made
in Stage 5.

```bat
conda activate gse
cd /d C:\Users\DELL\Missing_Python\mipy_pkg
python -m pip install --upgrade build twine
rmdir /s /q dist
python -m build
python -m twine check --strict dist/*
```

* `rmdir` may say "The system cannot find the file specified"; that is fine.
* `python -m build` ends with
  `Successfully built mipy_impute-2.1.0.tar.gz and mipy_impute-2.1.0-py3-none-any.whl`.
* `twine check` prints `PASSED` twice.

**Checkpoint 7:** two files in `mipy_pkg\dist`, both PASSED.

---

## Stage 8. Trial upload to TestPyPI (20 min)

1. **Token.** test.pypi.org > Account settings > **API tokens** > Add API
   token. Name `mipy upload`, scope **Entire account (all projects)**,
   Create. Copy the token (it starts with `pypi-`). It is shown only once;
   keep it in Notepad until the upload is done, then close Notepad without
   saving.
2. **Upload:**

```bat
python -m twine upload --repository testpypi dist/*
```

   If twine asks for a username, type `__token__` (two underscores on each
   side). When it asks for the API token or password, right-click to paste
   it; nothing appears on screen, which is normal. Press Enter. The upload
   ends with `View at: https://test.pypi.org/project/mipy-impute/2.1.0/`.
3. Open that link. The README shows, figure included.
4. **Install it like a stranger, in a fresh environment.** The dependencies
   come from the real PyPI, only mipy-impute from TestPyPI:

```bat
conda create -n mipytest python=3.11 -y
conda activate mipytest
pip install numpy pandas scipy scikit-learn psutil matplotlib openpyxl
pip install --index-url https://test.pypi.org/simple/ --no-deps mipy-impute
cd /d C:\Users\DELL\Missing_Python\mipy_trial
mipy --version
mipy example ex2.csv
mipy analyze ex2.csv --methods quick --repeats 1 --out test_results --open
conda activate gse
```

**Checkpoint 8:** `mipy 2.1.0` and the report opens from the TestPyPI copy.

---

## Stage 9. Publish on PyPI (15 min)

Only when Checkpoints 2 to 8 passed and the co-authors agreed.

1. **Token** on pypi.org, exactly as in Stage 8.1 (a TestPyPI token does
   not work on PyPI).
2. **Upload:**

```bat
conda activate gse
cd /d C:\Users\DELL\Missing_Python\mipy_pkg
python -m twine upload dist/*
```

   It ends with `View at: https://pypi.org/project/mipy-impute/2.1.0/`.
   From now on the version number 2.1.0 is used up: PyPI never accepts the
   same version twice, even after a deletion.
3. **Swap the token.** pypi.org > Account settings > API tokens: delete the
   "Entire account" token, and when you next need one, make it with scope
   "Project: mipy-impute".
4. **Install it from PyPI in a fresh environment:**

```bat
conda create -n mipycheck python=3.11 -y
conda activate mipycheck
pip install mipy-impute
cd /d C:\Users\DELL\Missing_Python\mipy_trial
mipy example ex3.csv
mipy analyze ex3.csv --methods quick --repeats 1 --out pypi_results --open
conda activate gse
```

   If pip says "No matching distribution found", wait five minutes and try
   again.

**Checkpoint 9:** <https://pypi.org/project/mipy-impute/> is live and
`pip install mipy-impute` works in a clean environment. The package is
published.

---

## Stage 10. DOI for the paper (20 min)

1. <https://zenodo.org> > Log in > **Log in with GitHub** > Authorize.
2. Your name (top right) > **GitHub**. Find `YOUR-USERNAME/mipy` and switch
   it **On**. If it is not listed, click **Sync now**. This has to happen
   before step 3.
3. On GitHub, in the repository: **Releases** (right side) > **Create a new
   release**.
   * Choose a tag: type `v2.1.0`, click "Create new tag: v2.1.0 on publish".
   * Target: `main`. Title: `MIPY 2.1.0`.
   * Description: paste the "2.1.0" section of `CHANGELOG.md`.
   * **Publish release**.
4. A few minutes later Zenodo shows the record under **My dashboard**
   (Uploads), with a DOI like `10.5281/zenodo.1234567`. The record page shows two DOIs: the
   one for this version, and a "concept" DOI for all versions. Cite the
   version DOI in the paper.
5. Add the DOI on GitHub: open `README.md` > pencil icon, and under the
   three badge lines add

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.1234567)
```

   then open `CITATION.cff` > pencil, add the line
   `doi: "10.5281/zenodo.1234567"` under `date-released`, and commit both
   (with your real number). Then bring your computer's copy up to date:

```bat
cd /d C:\Users\DELL\Missing_Python\mipy_pkg
git pull
```

   The PyPI page keeps the old README until the next version; that is fine.
6. In the manuscript, replace [MISSING: repository URL and archive DOI] with
   the GitHub address and the DOI.

**Checkpoint 10:** DOI badge on GitHub; DOI in the manuscript.

---

## Later: releasing a fix (version 2.1.1 and on)

1. Change the code; run `python -m pytest -q`.
2. Raise the version in three places: `pyproject.toml` (`version =`),
   `src\mipy\__init__.py` (`__version__ =`), `CITATION.cff` (`version:`).
   Add a section to `CHANGELOG.md`.
3. `git add .`, `git commit -m "MIPY 2.1.1"`, `git push`; wait for green
   Actions.
4. Stage 7 (build), then `python -m twine upload dist/*` with a
   project-scoped token.
5. A GitHub release `v2.1.1`; Zenodo makes the new DOI by itself.

---

## When something goes wrong

| Message | What it means and what to do |
|---|---|
| `'mipy' is not recognized` | the environment is not active, or the install did not finish: `conda activate gse` and repeat the pip line of Stage 2 |
| `'git' is not recognized` | install Git (Stage 0), then reopen the Anaconda Prompt |
| `ERROR: file:///... does not appear to be a Python project` | the prompt is not in the folder that has `pyproject.toml`; repeat the `cd /d` line |
| push: `rejected ... fetch first` | the repository was created with a README. Run `git pull origin main --allow-unrelated-histories`, save the merge message, then `git push -u origin main` |
| push: `Repository not found` | wrong user name in the address: `git remote set-url origin https://github.com/RIGHT-NAME/mipy.git`, then push again |
| twine: `403 Forbidden` or `Invalid or non-existent authentication` | wrong token, a TestPyPI token used on PyPI (or the other way round), or the username was not `__token__` |
| twine: `400 File already exists` | this version is already uploaded; raise the version (see "Later") and build again |
| twine: `400 ... too similar to an existing project` | PyPI refuses the name; tell me and we change it in `pyproject.toml` only (users would still `import mipy`) |
| build: `PermissionError` or `Access is denied` | OneDrive, antivirus or an open Jupyter is holding a file; close Jupyter and try again |
| `SSL` or `ProxyError` | the network blocks it; use a phone hotspot |
| a red job in GitHub Actions | send me the last 30 lines of the red step |
