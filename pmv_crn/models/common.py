from __future__ import annotations
from dataclasses import dataclass
import torch
TEMPORAL_CONCEPT_NAMES = ['level_trend', 'volatility', 'burst_shape', 'alarm_memory', 'complexity', 'multiview_coupling', 'risk_accumulation', 'lead_lag_response']
IMAGE_CONCEPT_NAMES = ['recurrence_block', 'trend_cluster', 'energy_cluster', 'lag_band', 'cross_source_coupling', 'absolute_level_recurrence', 'state_boundary', 'block_contrast']
BINARY_CLASS_NAMES = ['normal', 'danger']

def _group_count(channels: int, max_groups: int=8) -> int:
    for groups in range(min(max_groups, int(channels)), 0, -1):
        if int(channels) % groups == 0:
            return groups
    return 1

@dataclass
class BranchCMROutput:
    concepts: torch.Tensor
    rule_logits: torch.Tensor
    rule_weights: torch.Tensor
    rule_probs: torch.Tensor
    y_rule: torch.Tensor
