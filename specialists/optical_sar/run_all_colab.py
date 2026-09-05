"""Master End-to-End Execution Script for Google Colab.
Runs verification, stability diagnostics, 20-epoch training, and test evaluation in one single pass.
"""

import os
import sys
import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader

os.chdir("/content/SatQuery")
if "/content/SatQuery" not in sys.path:
    sys.path.insert(0, "/content/SatQuery")

print("=" * 70)
print("SATQUERY DIVISION 4: MASTER END-TO-END TRAINING & EVALUATION")
print(f"Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 70)

# -------------------------------------------------------------------------
# STAGE 1: Real Dataset Verification & Batch Dimensional Contract Checks
# -------------------------------------------------------------------------
print("\n[STAGE 1/4] Verifying Real Dataset & Batch Dimensional Contract...")
from specialists.optical_sar.dataset import OpticalSarPairedDataset
from specialists.optical_sar.query_intent import QueryIntentInterpreter, FiLMQueryModulator

dataset_path = "data/official_whu_opt_sar"
train_ds = OpticalSarPairedDataset(dataset_path, split="train", num_classes=8)
val_ds = OpticalSarPairedDataset(dataset_path, split="val", num_classes=8)
test_ds = OpticalSarPairedDataset(dataset_path, split="test", num_classes=8)

print(f"  - Train tiles: {len(train_ds)} (expected: 11880)")
print(f"  - Val tiles  : {len(val_ds)} (expected: 2310)")
print(f"  - Test tiles : {len(test_ds)} (expected: 2970)")

assert len(train_ds) == 11880, f"Train count mismatch: {len(train_ds)}"
assert len(val_ds) == 2310, f"Val count mismatch: {len(val_ds)}"
assert len(test_ds) == 2970, f"Test count mismatch: {len(test_ds)}"

loader = DataLoader(train_ds, batch_size=16, shuffle=False)
batch = next(iter(loader))

print(f"  - Batch optical shape: {list(batch['optical'].shape)}")
print(f"  - Batch SAR shape    : {list(batch['sar'].shape)}")
print(f"  - Batch label shape  : {list(batch['label'].shape)}")
print(f"  - Batch intent shape : {list(batch['intent_vector'].shape)}")

assert batch['optical'].shape == (16, 3, 256, 256)
assert batch['sar'].shape == (16, 2, 256, 256)
assert batch['label'].shape == (16, 256, 256)
assert batch['intent_vector'].shape == (16, 8)

film = FiLMQueryModulator(feature_channels=256, num_classes=8)
mod = film(torch.randn(16, 256, 32, 32), batch['intent_vector'])
assert mod.shape == (16, 256, 32, 32)
assert torch.isfinite(mod).all(), "FiLM output contains NaN/Inf!"
print(">>> STAGE 1 VERIFICATION PASSED SUCCESSFULLY! <<<\n")

# -------------------------------------------------------------------------
# STAGE 2: Numerical Stability Diagnostics
# -------------------------------------------------------------------------
print("[STAGE 2/4] Running Numerical Stability Diagnostics...")
from specialists.optical_sar.train_colab import ColabTrainer
trainer = ColabTrainer(dataset_dir=dataset_path)

# 3-step optimizer check
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
optimizer = torch.optim.AdamW(list(trainer.fusion_neck.parameters()) + list(trainer.task_head.parameters()), lr=1e-3)

for step, batch in enumerate(train_loader):
    if step >= 3:
        break
    opt = batch["optical"].to(trainer.device)
    sar = batch["sar"].to(trainer.device)
    targets = batch["label"].to(trainer.device)
    intent = batch["intent_vector"].to(trainer.device)

    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=trainer.use_amp):
        with torch.no_grad():
            opt_feats = trainer.optical_encoder(opt)
            sar_feats = trainer.sar_encoder(sar)
        fused = trainer.fusion_neck(opt_feats["stride_8"], sar_feats["stride_8"])
        logits, probs = trainer.task_head(fused, intent, optical_raw=opt, sar_raw=sar)
        if logits.shape[2:] != targets.shape[1:]:
            logits = torch.nn.functional.interpolate(logits, size=targets.shape[1:], mode="bilinear", align_corners=False)
        ce = trainer.ce_loss_fn(logits, targets)
        dice = trainer.dice_loss_fn(logits, targets)
        loss = ce + 0.5 * dice

    assert torch.isfinite(loss), f"Encountered NaN/Inf loss in step {step}: {loss}"
    trainer.scaler.scale(loss).backward()
    trainer.scaler.step(optimizer)
    trainer.scaler.update()
    print(f"  Step {step+1}/3 passed -> Loss: {loss.item():.4f}")

# Full validation scan across 2,310 tiles
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
val_loss, val_metrics = trainer.validate(val_loader)
print(f"  Validation Scan -> Loss: {val_loss:.4f} | Pixel Acc: {val_metrics['pixel_acc']:.4f} | Initial mIoU: {val_metrics['mIoU']:.4f}")
assert torch.isfinite(torch.tensor(val_loss)), "Validation loss is NaN/Inf!"
print(">>> STAGE 2 STABILITY PASSED SUCCESSFULLY! <<<\n")

# -------------------------------------------------------------------------
# STAGE 3: Full 20-Epoch GPU Training
# -------------------------------------------------------------------------
print("[STAGE 3/4] Starting Full GPU Training (15 Epochs Stage 1 + 5 Epochs Stage 2)...")
best_ckpt = trainer.run_colab_training(epochs=15, batch_size=16, lr=1e-3, fine_tune_epochs=5)
print(f"\n>>> TRAINING COMPLETED! Best checkpoint: {best_ckpt} <<<\n")

# -------------------------------------------------------------------------
# STAGE 4: Held-Out Test Evaluation & Modality Ablation
# -------------------------------------------------------------------------
print("[STAGE 4/4] Evaluating Held-Out Test Split (2,970 Tiles)...")
from specialists.optical_sar.evaluate import evaluate_checkpoint

eval_results = evaluate_checkpoint(
    checkpoint_path=str(best_ckpt),
    data_dir=dataset_path,
    output_dir="specialists/optical_sar/eval_results",
    batch_size=16,
)

print("\n" + "=" * 70)
print("FINAL QUANTITATIVE EVALUATION RESULTS")
print("=" * 70)
import json
print(json.dumps(eval_results, indent=2))
print("=" * 70)
print("ALL TASKS COMPLETED SUCCESSFULLY!")
