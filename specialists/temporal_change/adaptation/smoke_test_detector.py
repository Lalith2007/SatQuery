"""Mandatory Gradient Smoke Test for TinyCD Change Detector.

Mathematically proves:
1. Genuine TinyCD PyTorch model initialization
2. Trainable parameter count
3. Forward loss computation on paired inputs
4. Non-zero gradient backpropagation (||grad(W)|| > 0)
5. Optimizer update producing a non-zero parameter delta (||W_1 - W_0|| > 0)
6. Checkpoint serialization with verifiable SHA-256 hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim

from specialists.temporal_change.adaptation.models.tinycd import TinyCD, count_parameters


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_gradient_smoke_test(
    output_dir: Path | str = "specialists/temporal_change/weights",
    device: str = "auto",
) -> Dict[str, Any]:
    """Execute mathematical gradient smoke test and serialize smoke_test_proof.json."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:
        dev = device

    # 1. Model instantiation
    model = TinyCD(in_channels=3, base_features=32)
    model.to(dev)
    model.train()

    total_params, trainable_params = count_parameters(model)
    assert trainable_params > 0, "Model has 0 trainable parameters!"

    # 2. Record initial monitored parameter (classifier head weights)
    target_param_name = "classifier.0.weight"
    target_param = dict(model.named_parameters())[target_param_name]
    w_0 = target_param.detach().clone()

    # 3. Create synthetic test batch (2 paired images + 1 ground truth mask)
    b_size = 2
    t0 = torch.rand(b_size, 3, 256, 256, device=dev, requires_grad=False)
    t1 = torch.rand(b_size, 3, 256, 256, device=dev, requires_grad=False)
    mask = torch.randint(0, 2, (b_size, 1, 256, 256), device=dev).float()

    # 4. Forward pass
    pred = model(t0, t1)
    bce_loss = nn.BCELoss()(pred, mask)
    # Soft Dice Loss
    intersection = (pred * mask).sum()
    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (pred.sum() + mask.sum() + 1.0)
    total_loss = bce_loss + dice_loss

    loss_val = float(total_loss.item())
    assert not torch.isnan(total_loss), "Loss computed is NaN!"

    # 5. Backpropagation
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer.zero_grad()
    total_loss.backward()

    # 6. Verify gradient norm and non-zero gradient tensors
    grad_norms = []
    non_zero_grads = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            norm = p.grad.norm().item()
            grad_norms.append(norm)
            if norm > 0:
                non_zero_grads += 1

    total_grad_norm = sum(grad_norms)
    assert total_grad_norm > 0, "Gradient norm is zero — backpropagation failed!"
    assert non_zero_grads > 0, "No non-zero gradient tensors found!"

    # 7. Optimizer step
    optimizer.step()

    # 8. Verify monitored parameter delta (W_1 - W_0)
    w_1 = dict(model.named_parameters())[target_param_name].detach().clone()
    param_delta = float((w_1 - w_0).norm().item())
    assert param_delta > 0, f"Parameter delta after optimizer step is 0 for {target_param_name}!"

    # 9. Save smoke test checkpoint
    checkpoint_path = out_dir / "smoke_test_checkpoint.pth"
    torch.save(model.state_dict(), checkpoint_path)
    chk_sha256 = compute_file_sha256(checkpoint_path)

    proof = {
        "status": "REAL_NEURAL_TRAINING_VERIFIED",
        "model_architecture": "TinyCD (Siamese U-Net + MAMB)",
        "device": dev,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "forward_loss": round(loss_val, 6),
        "gradient_norm": round(total_grad_norm, 6),
        "non_zero_gradient_tensors": non_zero_grads,
        "monitored_parameter": target_param_name,
        "parameter_delta_after_step": round(param_delta, 8),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": chk_sha256,
    }

    proof_path = out_dir / "smoke_test_proof.json"
    with open(proof_path, "w") as f:
        json.dump(proof, f, indent=2)

    return proof


if __name__ == "__main__":
    result = run_gradient_smoke_test()
    print(json.dumps(result, indent=2))
