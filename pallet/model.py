"""PalletNet: PointDiT with 6 continuous output channels (rectified flow, x-prediction, as upstream) plus NUM_CLASSES
per-pixel class logits produced by the same final layer. The upstream PointDiT module is used unchanged: it is built
with in_channels=6 and its final linear is widened to 6 + 4 outputs after construction.

Training objective (mirrors denoiser.py): t ~ logit-normal(P_mean, P_std), a fraction zero_t_ratio of t set to 0 (the
1-step inference condition), z = t*x + (1-t)*e with invalid pixels noise-filled, loss = (x - x_pred)^2 / (1-t)^2 on valid
channels (== the upstream v-space MSE), plus cls_weight * cross-entropy on the logits (ignore 255 = padding).
Inference: deterministic Euler from zeros; steps=1 gives x_pred = net(0, t=0, image).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import PointDiT_models  # repo root

from . import targets as T


class PalletNet(nn.Module):
    def __init__(self, model="PointDiT-B/16", feature_embedding_type="dinov3_vitb16", input_size=256,
                 P_mean=-0.8, P_std=0.8, t_eps=0.05, noise_scale=1.0, zero_t_ratio=0.1, steps=1,
                 cls_weight=1.0, channel_weights=None):
        super().__init__()
        self.net = PointDiT_models[model](input_size=input_size, in_channels=T.NUM_CONT,
                                          feature_embedding_type=feature_embedding_type)
        hidden, ps = self.net.hidden_size, self.net.patch_size
        self.net.out_channels = T.NUM_CONT + T.NUM_CLASSES
        self.net.final_layer.linear = nn.Linear(hidden, ps * ps * self.net.out_channels, bias=True)
        nn.init.zeros_(self.net.final_layer.linear.weight)          # upstream zero-inits this layer too
        nn.init.zeros_(self.net.final_layer.linear.bias)
        self.P_mean, self.P_std, self.t_eps, self.noise_scale = P_mean, P_std, t_eps, noise_scale
        self.zero_t_ratio, self.steps, self.cls_weight = zero_t_ratio, steps, cls_weight
        cw = torch.ones(T.NUM_CONT) if channel_weights is None else torch.as_tensor(channel_weights, dtype=torch.float32)
        self.register_buffer("channel_weights", cw.view(1, T.NUM_CONT, 1, 1), persistent=False)

    # -- weights -----------------------------------------------------------------------------------------------
    def load_pretrained(self, path: str, use_ema: bool = True) -> dict:
        """Load a released PointDiT checkpoint; the 3-channel patch-embed conv and the final linear are left at init.
        Returns {"dropped": [...], "missing": [...], "unexpected": [...]} for logging."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = (ck.get("model_ema1") or ck.get("ema")) if use_ema and ("model_ema1" in ck or "ema" in ck) else ck["model"]   # released ckpt or pallet/train.py ckpt
        own = self.state_dict()
        dropped = [k for k, v in sd.items() if k in own and own[k].shape != v.shape]
        sd = {k: v for k, v in sd.items() if k not in dropped}
        res = self.load_state_dict(sd, strict=False)
        missing = [k for k in res.missing_keys if not k.startswith("net.y_embedder.")]   # DINOv3 comes from its own file
        bad = [k for k in missing if k not in dropped]
        assert not bad and not res.unexpected_keys, (bad, res.unexpected_keys)
        return {"dropped": dropped, "missing": missing, "unexpected": list(res.unexpected_keys)}

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("net.y_embedder.")]

    # -- training ----------------------------------------------------------------------------------------------
    def sample_t(self, n: int, device) -> torch.Tensor:
        t = torch.sigmoid(torch.randn(n, device=device) * self.P_std + self.P_mean)
        if self.zero_t_ratio > 0:
            t = torch.where(torch.rand(n, device=device) < self.zero_t_ratio, torch.zeros_like(t), t)
        return t

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        x, valid, cls = batch["target"], batch["valid_mask"], batch["cls"]
        img = batch["image"] * 2.0 - 1.0
        B = x.size(0)
        t = self.sample_t(B, x.device).view(B, 1, 1, 1)
        e = torch.randn_like(x) * self.noise_scale
        z = t * torch.where(valid, x, e) + (1.0 - t) * e
        out = self.net(z, t.flatten(), img)
        x_pred, logits = out[:, :T.NUM_CONT].float(), out[:, T.NUM_CONT:].float()
        w = self.channel_weights / (1.0 - t).clamp_min(self.t_eps) ** 2
        se = (x_pred - x) ** 2 * w
        flow = se[valid].mean() if valid.any() else se.sum() * 0.0
        ce = F.cross_entropy(logits, cls, ignore_index=T.CLS_IGNORE)
        loss = flow + self.cls_weight * ce
        with torch.no_grad():                                               # per-channel RMSE in raw units, for logs
            err = ((x_pred - x) * T.TARGET_SCALE_T.to(x.device).view(1, -1, 1, 1)) ** 2
            per_ch = torch.stack([err[:, c][valid[:, c]].mean().sqrt() if valid[:, c].any() else err.new_zeros(())
                                  for c in range(T.NUM_CONT)])
        return loss, {"flow": flow.detach(), "ce": ce.detach(), "rmse_raw": per_ch, "t": t.flatten()}

    # -- inference ---------------------------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, image: torch.Tensor, steps: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """image [B,3,H,W] in [0,1] -> (x_pred [B,6,H,W] encoded targets, logits [B,4,H,W])."""
        steps = steps or self.steps
        img = image * 2.0 - 1.0
        B, _, H, W = img.shape
        ps = self.net.patch_size
        y = self.net.extract_y_embedding(img, H // ps, W // ps)
        z = torch.zeros(B, T.NUM_CONT, H, W, device=img.device, dtype=img.dtype)
        ts = torch.linspace(0.0, 1.0, steps + 1, device=img.device)
        for i in range(steps):
            t = ts[i]
            out = self.net(z, t.expand(B), img, cached_y_emb=y)
            x_pred, logits = out[:, :T.NUM_CONT], out[:, T.NUM_CONT:]
            z = x_pred if i == steps - 1 else z + (ts[i + 1] - t) * (x_pred - z) / (1.0 - t)
        return z.float(), logits.float()
