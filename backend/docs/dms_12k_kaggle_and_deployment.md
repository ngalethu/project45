# DMS 3-class 12k: Kaggle T4 and deployment

## Ready artifacts

- Dataset: `data/processed/dms_yolo_3class_v4_12k`
- Kaggle upload folder: `backend/outputs/kaggle_dms_3class_v4_12k_bundle`
- Notebook: `kaggle_train_dms_3class_12k.ipynb`
- Classes in fixed order: `phone`, `seatbelt`, `no-seatbelt`

The training split contains exactly 12,000 images from 12,000 capture groups. Validation (2,801) and test (2,681) are unchanged. The audit reports zero groups crossing splits.

## Kaggle T4 procedure

1. Create a Kaggle Dataset from the complete bundle folder. If using the Kaggle CLI, replace `YOUR_KAGGLE_USERNAME` in `dataset-metadata.json` first.
2. Create a GPU notebook, select a T4 accelerator, attach that dataset, then import and run `kaggle_train_dms_3class_12k.ipynb`.
3. The default run uses YOLO11m, 768 px, batch 8, 60 epochs, patience 12, AMP, one T4 and two workers. Expected duration is roughly 4–8 hours depending on the assigned T4 and Kaggle I/O.
4. Download everything under `/kaggle/working/dms_export/`:
   - `best.pt`
   - `metrics_summary.json`
   - `best.onnx` when ONNX export succeeds

The notebook evaluates the final checkpoint on the independent test split. It does not manufacture or copy training metrics into the test report.

## Safe model installation

Run from `backend` after downloading the artifacts:

```powershell
py -3 scripts/install_kaggle_dms_model.py `
  --best-pt "C:\path\to\best.pt" `
  --metrics "C:\path\to\metrics_summary.json" `
  --best-onnx "C:\path\to\best.onnx"
```

The installer verifies the checkpoint class order, verifies the per-class metrics, requires the configured mAP50/F1 target by default, backs up the deployed model, and then atomically replaces it. Do not rename the current four-class model manually.

## Evaluation rules

- Image metrics: per-class precision, recall, F1, AP50 and AP50–95 from the test split.
- Video metrics: one decision per video/event; repeated frames never multiply TP/FP/FN.
- Temporal confirmation: a 12-frame sliding window with 5 phone votes or 8 no-seatbelt votes.
- External domain: the 69 windshield-source test images are listed in `external_test_manifest.json` and are never used for training.
- The eight CC BY 4.0 real-car images remain weak external evaluation only because they belong to one driver and have no compatible detection boxes.

Training and deployment are not complete until Kaggle has produced genuine `best.pt` and `metrics_summary.json` artifacts.
