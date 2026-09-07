"""Master End-to-End Execution Script for Google Colab.
Runs verification, stability diagnostics, 50-epoch retraining, and test evaluation in one single pass.
"""

import os
import sys
import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader

if "/content/SatQuery" in sys.path or Path("/content/SatQuery").exists():
    os.chdir("/content/SatQuery")
    if "/content/SatQuery" not in sys.path:
        sys.path.insert(0, "/content/SatQuery")

print("=" * 80)
print("SATQUERY DIVISION 4: MASTER END-TO-END TRAINING & EVALUATION (COLAB GPU)")
print(f"Start Time    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device    : {torch.cuda.get_device_name(0)}")
print("=" * 80)

# -------------------------------------------------------------------------
# STAGE 1: Dataset Verification & Batch Dimensional Contract Checks
# -------------------------------------------------------------------------
print("\n[STAGE 1/3] Verifying Dataset & Batch Dimensional Contract...")
from specialists.optical_sar.dataset import OpticalSarPairedDataset
from specialists.optical_sar.query_intent import QueryIntentInterpreter, FiLMQueryModulator

dataset_path = "data/official_whu_opt_sar"
train_ds = OpticalSarPairedDataset(dataset_path, split="train", num_classes=8, augment=True)
val_ds = OpticalSarPairedDataset(dataset_path, split="val", num_classes=8, augment=False)
test_ds = OpticalSarPairedDataset(dataset_path, split="test", num_classes=8, augment=False)

print(f"  - Train tiles: {len(train_ds):,d}")
print(f"  - Val tiles  : {len(val_ds):,d}")
print(f"  - Test tiles : {len(test_ds):,d}")

assert len(train_ds) > 0, "Train dataset is empty!"
assert len(val_ds) > 0, "Validation dataset is empty!"
assert len(test_ds) > 0, "Test dataset is empty!"

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
print("[STAGE 2/3] Running Numerical Stability Diagnostics...")
from specialists.optical_sar.train_colab import ColabTrainer, compute_dynamic_class_weights
trainer = ColabTrainer(dataset_dir=dataset_path)

# 3-step optimizer check
trainer.setup_loss(torch.ones(8))
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
    with torch.amp.autocast("cuda", enabled=trainer.use_amp):
        opt_feats = trainer.optical_encoder(opt)
        sar_feats = trainer.sar_encoder(sar)
        fused = trainer.fusion_neck(opt_feats["stride_8"], sar_feats["stride_8"])
        logits, _ = trainer.task_head(fused, intent)
        if logits.shape[2:] != targets.shape[1:]:
            logits = torch.nn.functional.interpolate(logits, size=targets.shape[1:], mode="bilinear", align_corners=False)
        ce = trainer.ce_loss_fn(logits.float(), targets)
        dice = trainer.dice_loss_fn(logits.float(), targets)
        loss = ce + 0.5 * dice

    assert torch.isfinite(loss), f"Encountered NaN/Inf loss in step {step}: {loss}"
    if trainer.use_amp:
        trainer.scaler.scale(loss).backward()
        trainer.scaler.step(optimizer)
        trainer.scaler.update()
    else:
        loss.backward()
        optimizer.step()
    print(f"  Step {step+1}/3 passed -> Loss: {loss.item():.4f}")

# Validation scan check
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
val_loss, val_metrics, _ = trainer.validate(val_loader)
print(f"  Validation Scan -> Loss: {val_loss:.4f} | OA: {val_metrics['overall_accuracy']*100:.2f}% | Initial mIoU: {val_metrics['mIoU']*100:.2f}%")
assert torch.isfinite(torch.tensor(val_loss)), "Validation loss is NaN/Inf!"
print(">>> STAGE 2 STABILITY PASSED SUCCESSFULLY! <<<\n")

# -------------------------------------------------------------------------
# STAGE 3: Full 50-Epoch GPU Training & Post-Training Evaluation Suite
# -------------------------------------------------------------------------
print("[STAGE 3/3] Starting Full GPU Retraining Pipeline (50 Epochs, Cosine Annealing, Early Stopping)...")
best_ckpt = trainer.run_colab_training(
    epochs=50,
    warmup_epochs=5,
    batch_size=16,
    lr_head=1e-3,
    ft_lr_head=5e-4,
    ft_lr_backbone=2e-5,
    num_workers=2,
    patience=10,
    seed=42,
)
print(f"\n>>> COMPLETE PIPELINE RUN FINISHED! Best Checkpoint Saved: {best_ckpt} <<<\n")
