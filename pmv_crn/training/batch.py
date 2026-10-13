from __future__ import annotations
from typing import Dict
import torch
from pmv_crn.models.network import OptimizedDualBranchCMRNet

def forward_model_batch(model: OptimizedDualBranchCMRNet, batch: Dict[str, object], device: torch.device) -> Dict[str, torch.Tensor]:
    kwargs = {'stat_features': batch['stat_features'].to(device)}
    return model(batch['x_ts'].to(device), batch['x_img'].to(device), batch['ts_concepts'].to(device), batch['img_concepts'].to(device), **kwargs)
