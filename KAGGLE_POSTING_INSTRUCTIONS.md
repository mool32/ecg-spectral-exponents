# How to Post the Notebook on Kaggle

## Step 1: Create a Precomputed Dataset

The notebook uses two precomputed CSV files (IRASA takes ~45 min, beat variability ~20 min).
You need to upload them as a private Kaggle dataset so the notebook can load them.

### Files to upload:
```
results/beta_features.csv        (~17 MB, 21,797 rows × 44 columns)
results/block_D_beat_variability.csv  (~1 MB, 3,000 rows × 23 columns)
```

### How:
1. Go to https://www.kaggle.com/datasets
2. Click **"+ New Dataset"**
3. Name it: `heartbeat-chaos-precomputed`
4. Drag and drop both CSV files
5. Set visibility: **Public** (so the notebook is fully reproducible)
6. Click **"Create"**

After creation, the files will be accessible at:
```
/kaggle/input/heartbeat-chaos-precomputed/beta_features.csv
/kaggle/input/heartbeat-chaos-precomputed/block_D_beat_variability.csv
```

---

## Step 2: Update Paths in the Notebook

Before uploading, edit **cell 3** (the path configuration cell). Change:

```python
# Locally:
DATA_DIR = 'ptb-xl'
PRECOMPUTED_DIR = 'results'
```

To:

```python
# On Kaggle:
DATA_DIR = '/kaggle/input/ptb-xl-a-large-publicly-available-electrocardiography-dataset/'
PRECOMPUTED_DIR = '/kaggle/input/heartbeat-chaos-precomputed/'
```

(Comment out or delete the local paths.)

---

## Step 3: Upload the Notebook

1. Go to https://www.kaggle.com/code
2. Click **"+ New Notebook"**
3. In the new notebook page, click **File → Import Notebook**
4. Select `kaggle_notebook.ipynb` from your computer
5. The notebook will be imported with all cells

---

## Step 4: Attach the Datasets

In the right sidebar of the notebook editor:

1. Click **"+ Add Input"** (or the **Data** tab)
2. Search for **"PTB-XL"** and add:
   - `ptb-xl-a-large-publicly-available-electrocardiography-dataset` (by PhysioNet)
3. Search for **"heartbeat-chaos-precomputed"** and add:
   - Your dataset from Step 1

You should see both datasets listed under "Input" in the sidebar.

---

## Step 5: Configure the Kernel

In the right sidebar:

- **Language:** Python
- **Environment:** Latest available (Docker image with scikit-learn, scipy, etc.)
- **Accelerator:** None (CPU is fine — we use precomputed features)
- **Internet:** ON (only needed if `COMPUTE_FROM_SCRATCH = True` and you want to `pip install neurodsp`)
- **GPU/TPU:** Not needed

### If running from scratch (optional):
If someone sets `COMPUTE_FROM_SCRATCH = True`, they also need:
```python
!pip install neurodsp wfdb
```
Add this as the first code cell. But for the default precomputed path, no extra packages are needed.

---

## Step 6: Run & Verify

1. Click **"Run All"** (or Shift+Enter through cells)
2. Expected runtime: **~2-3 minutes** (precomputed path)
3. Verify all 4 figures render correctly
4. Check the Summary Statistics cell at the bottom for the key numbers:
   - β_NORM ≈ 1.76
   - CD Cohen's d ≈ 1.0
   - ρ_age ≈ −0.17
   - Breakpoint ≈ 42y
   - Beat variability ρ ≈ 0.165
   - Bio age r ≈ 0.60

---

## Step 7: Set Metadata & Publish

1. **Title:** `The Heartbeat at the Edge of Chaos: What 21,000 ECGs Reveal About Cardiac Complexity`

2. **Tags** (add via sidebar):
   - `ecg`
   - `cardiology`
   - `signal processing`
   - `aging`
   - `ptb-xl`
   - `spectral analysis`
   - `biological age`
   - `1/f noise`

3. **Slug** (URL): Will auto-generate from title, or set manually:
   `heartbeat-edge-of-chaos-21000-ecgs`

4. Click **"Save Version"**:
   - Version name: `v1.0`
   - Save & Run: **"Save & Run All (Commit)"**
   - This will execute the notebook on Kaggle servers and save outputs

5. Wait for the run to complete (check the "Versions" tab)

6. Once the run succeeds, click **"Share" → "Public"**

---

## Step 8: Write a Competition/Discussion Post (Optional)

If you want maximum visibility, create a discussion post in the PTB-XL dataset forum:

1. Go to https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset/discussion (or the official PTB-XL dataset page)
2. Click **"+ New Topic"**
3. Title: `Spectral Exponent β Reveals Aging-Pathology Bifurcation in 21,000 ECGs`
4. Body (short version):

```
I applied IRASA spectral decomposition to all 21,797 PTB-XL records and found:

1. Each diagnosis has a unique spectral fingerprint — CD is the strongest outlier (d ≈ 1.0)
2. Aging and pathology drive β in opposite directions (bifurcation at ~42 years)
3. Beat-to-beat morphological variability independently confirms complexity loss
4. A bio-age model from β-features achieves r = 0.60

Full analysis with code and figures: [link to your notebook]
```

---

## Checklist Before Publishing

- [ ] Both datasets attached (PTB-XL + precomputed)
- [ ] Paths point to `/kaggle/input/...` (not local)
- [ ] `COMPUTE_FROM_SCRATCH = False` and `COMPUTE_BEATS = False`
- [ ] All 4 figures render correctly
- [ ] Summary numbers match expected values
- [ ] Title and tags are set
- [ ] Notebook runs successfully on Kaggle (check Versions tab)
- [ ] Set to Public
