from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from pmv_crn.models.common import BranchCMROutput

class CMRRuleMemory(nn.Module):

    def __init__(self, emb_dim: int, n_concepts: int, n_classes: int=2, n_rules: int=8, temperature: float=0.7, dropout: float=0.15, prior_blend: float=0.15):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.n_rules = n_rules
        self.temperature = temperature
        self.prior_blend = prior_blend
        self.concept_head = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.LayerNorm(emb_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(emb_dim, n_concepts))
        self.selector = nn.Sequential(nn.Linear(emb_dim + n_concepts, emb_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(emb_dim, n_rules))
        self.rule_relevance_logits = nn.Parameter(torch.randn(n_rules, n_classes, n_concepts) * 0.02)
        self.rule_polarity_logits = nn.Parameter(torch.randn(n_rules, n_classes, n_concepts) * 0.02)
        self.rule_class_bias = nn.Parameter(torch.zeros(n_rules, n_classes))

    def _rule_truth(self, concepts: torch.Tensor) -> torch.Tensor:
        c = concepts[:, None, None, :]
        relevance = torch.sigmoid(self.rule_relevance_logits)[None, :, :, :]
        polarity = torch.sigmoid(self.rule_polarity_logits)[None, :, :, :]
        sign_terms = polarity * c + (1.0 - polarity) * (1.0 - c)
        filtered_terms = 1.0 - relevance * (1.0 - sign_terms)
        log_truth = torch.log(filtered_terms.clamp_min(1e-06)).sum(dim=-1)
        truth = torch.sigmoid(log_truth + self.rule_class_bias[None, :, :])
        return truth

    def forward(self, embedding: torch.Tensor, concept_prior: Optional[torch.Tensor]=None) -> BranchCMROutput:
        concepts_pred = torch.sigmoid(self.concept_head(embedding))
        if concept_prior is not None and self.prior_blend > 0:
            prior = concept_prior.to(device=embedding.device, dtype=embedding.dtype).clamp(0.0, 1.0)
            concepts = (1.0 - self.prior_blend) * concepts_pred + self.prior_blend * prior
        else:
            concepts = concepts_pred
        selector_input = torch.cat([embedding, concepts], dim=-1)
        rule_logits = self.selector(selector_input)
        rule_weights = F.softmax(rule_logits / self.temperature, dim=-1)
        rule_probs = self._rule_truth(concepts)
        y_rule_raw = torch.einsum('br,brc->bc', rule_weights, rule_probs)
        y_rule = y_rule_raw / y_rule_raw.sum(dim=-1, keepdim=True).clamp_min(1e-06)
        return BranchCMROutput(concepts, rule_logits, rule_weights, rule_probs, y_rule)

    def regularization_loss(self) -> torch.Tensor:
        relevance = torch.sigmoid(self.rule_relevance_logits)
        sparsity = relevance.mean()
        flat = relevance.flatten(1)
        diversity = torch.pdist(flat, p=2).mean() if self.n_rules > 1 else flat.new_tensor(0.0)
        return sparsity - 0.05 * diversity
