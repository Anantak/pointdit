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
                 cls_weight=1.0, channel_weights=None, consist_weight=0.0):
        super().__init__()
        self.net = PointDiT_models[model](input_size=input_size, in_channels=T.NUM_CONT,
                                          feature_embedding_type=feature_embedding_type)
        # PointDiT.__init__ ends with initialize_weights(), which Xavier-initialises EVERY nn.Linear, including the DINOv3 encoder it
        # has just loaded through torch.hub; upstream main.py reloads the encoder afterwards. Do the same here (strict), otherwise
        # the "frozen DINOv3" is a frozen random network (this was the case for every run up to 2026-10-24, see the handover).
        self.dinov3_loaded = self.reload_dinov3(feature_embedding_type)
        hidden, ps = self.net.hidden_size, self.net.patch_size
        self.net.out_channels = T.NUM_CONT + T.NUM_CLASSES
        self.net.final_layer.linear = nn.Linear(hidden, ps * ps * self.net.out_channels, bias=True)
        nn.init.zeros_(self.net.final_layer.linear.weight)          # upstream zero-inits this layer too
        nn.init.zeros_(self.net.final_layer.linear.bias)
        self.P_mean, self.P_std, self.t_eps, self.noise_scale = P_mean, P_std, t_eps, noise_scale
        self.zero_t_ratio, self.steps, self.cls_weight, self.consist_weight = zero_t_ratio, steps, cls_weight, consist_weight
        cw = torch.ones(T.NUM_CONT) if channel_weights is None else torch.as_tensor(channel_weights, dtype=torch.float32)
        self.register_buffer("channel_weights", cw.view(1, T.NUM_CONT, 1, 1), persistent=False)

    # -- weights -----------------------------------------------------------------------------------------------
    DINOV3_FILES = {"dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
                    "dinov3_vith16plus": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"}

    def reload_dinov3(self, feature_embedding_type: str) -> bool:
        """Load the gated DINOv3 weights into net.y_embedder from DINOV3_WEIGHTS_DIR (or pretrained/dinov3). Returns False if absent."""
        import os
        from pathlib import Path
        d = Path(os.environ.get("DINOV3_WEIGHTS_DIR", Path(__file__).resolve().parents[1] / "pretrained" / "dinov3"))
        f = d / self.DINOV3_FILES.get(feature_embedding_type, "")
        if not f.is_file():
            return False
        self.net.y_embedder.load_state_dict(torch.load(f, map_location="cpu"), strict=True)
        return True

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

    @classmethod
    def load_inference(cls, path: str, device="cuda"):
        """Build a PalletNet from an exported inference checkpoint (pallet/export_checkpoint.py): EMA weights, DINOv3 stripped."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        meta = ck["meta"]
        net = cls(model=meta["model"], feature_embedding_type=meta["feature_embedding_type"])
        sd = {k: (v.float() if v.is_floating_point() else v) for k, v in ck["model"].items()}
        res = net.load_state_dict(sd, strict=False)
        if meta.get("dinov3_stripped", True):
            assert net.dinov3_loaded, "DINOv3 weights not found: set DINOV3_WEIGHTS_DIR"
            bad = [k for k in res.missing_keys if not k.startswith("net.y_embedder.")]
        else:
            bad = list(res.missing_keys)
        assert not bad and not res.unexpected_keys, (bad, res.unexpected_keys)
        return net.to(device).eval(), meta

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("net.y_embedder.")]

    # -- training ----------------------------------------------------------------------------------------------
    def sample_t(self, n: int, device) -> torch.Tensor:
        t = torch.sigmoid(torch.randn(n, device=device) * self.P_std + self.P_mean)
        if self.zero_t_ratio > 0:
            t = torch.where(torch.rand(n, device=device) < self.zero_t_ratio, torch.zeros_like(t), t)
        return t

    def forward(self, batch: dict, prior_p: float = 0.0) -> tuple[torch.Tensor, dict]:
        """Training step. With prior_p > 0 and batch["prior"] present, a fraction prior_p of the samples that have a prior are
        noised from the PRIOR instead of the target (z = t * prior_filled + (1 - t) * e), so the net learns to correct a
        propagated previous estimate at any t; the loss is always against the current target."""
        x, valid, cls = batch["target"], batch["valid_mask"], batch["cls"]
        img = batch["image"] * 2.0 - 1.0
        B = x.size(0)
        t = self.sample_t(B, x.device).view(B, 1, 1, 1)
        e = torch.randn_like(x) * self.noise_scale
        x_in = torch.where(valid, x, e)
        n_prior = 0
        if prior_p > 0 and "prior" in batch:
            use = batch["has_prior"].view(B, 1, 1, 1) & (torch.rand(B, 1, 1, 1, device=x.device) < prior_p)
            x_in = torch.where(use, torch.where(batch["prior_mask"], batch["prior"], e), x_in)
            n_prior = int(use.sum())
        z = t * x_in + (1.0 - t) * e
        out = self.net(z, t.flatten(), img)
        x_pred, logits = out[:, :T.NUM_CONT].float(), out[:, T.NUM_CONT:].float()
        w = self.channel_weights / (1.0 - t).clamp_min(self.t_eps) ** 2
        se = (x_pred - x) ** 2 * w
        flow = se[valid].mean() if valid.any() else se.sum() * 0.0
        ce = F.cross_entropy(logits, cls, ignore_index=T.CLS_IGNORE)
        loss = flow + self.cls_weight * ce
        consist = x.new_zeros(())
        if self.consist_weight > 0:                                         # |P_pred - (c + r_pred * R d)| on hit pixels (metres)
            consist = self.consistency(x_pred, batch, valid)
            loss = loss + self.consist_weight * consist
        with torch.no_grad():                                               # per-channel RMSE in raw units, for logs
            err = ((x_pred - x) * T.TARGET_SCALE_T.to(x.device).view(1, -1, 1, 1)) ** 2
            per_ch = torch.stack([err[:, c][valid[:, c]].mean().sqrt() if valid[:, c].any() else err.new_zeros(())
                                  for c in range(T.NUM_CONT)])
        return loss, {"flow": flow.detach(), "ce": ce.detach(), "consist": consist.detach(), "rmse_raw": per_ch, "t": t.flatten(), "n_prior": n_prior}

    @staticmethod
    def pallet_rays(batch: dict) -> torch.Tensor:
        """Unit pixel rays of the (padded, possibly cropped) image in the PALLET frame, from intrinsics and the GT pose: [B,3,H,W]."""
        B, H, W = batch["cls"].shape
        fx, fy, cx, cy = batch["intrinsics"].T.float()
        u = torch.arange(W, device=fx.device).float() + 0.5; v = torch.arange(H, device=fx.device).float() + 0.5
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        d = torch.stack([(uu[None] - cx[:, None, None]) / fx[:, None, None], (vv[None] - cy[:, None, None]) / fy[:, None, None],
                         torch.ones(B, H, W, device=fx.device)], 1)
        d = d / d.norm(dim=1, keepdim=True)
        R = batch["pallet_P_cam"][:, :9].float().view(B, 3, 3)
        return torch.einsum("bij,bjhw->bihw", R, d)

    def consistency(self, x_pred: torch.Tensor, batch: dict, valid: torch.Tensor) -> torch.Tensor:
        """Mean |P_pred - (c + range_pred * d_pallet)| over pixels where P and range are supervised, in metres (decoded units)."""
        scale, off = T.TARGET_SCALE_T.to(x_pred.device).view(1, -1, 1, 1), T.TARGET_OFFSET_T.to(x_pred.device).view(1, -1, 1, 1)
        raw = x_pred * scale + off
        P, r = raw[:, 0:3], raw[:, 3:4]
        c = batch["cam_in_pallet"].float().view(-1, 3, 1, 1)
        on_ray = c + r * self.pallet_rays(batch)
        m = valid[:, 0:1] & valid[:, 3:4]
        err = (P - on_ray).norm(dim=1, keepdim=True)
        return err[m].mean() if m.any() else err.sum() * 0.0

    # -- inference ---------------------------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, image: torch.Tensor, steps: int | None = None, x_prior: torch.Tensor | None = None,
                 prior_mask: torch.Tensor | None = None, t0: float = 0.0, prior_fill: str = "noise",
                 generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """image [B,3,H,W] in [0,1] -> (x_pred [B,6,H,W] encoded targets, logits [B,4,H,W]).
        Warm start: with x_prior [B,6,H,W] (encoded) and prior_mask [B,6,H,W], integration starts at time t0 instead of 0.
        prior_fill: "noise" = z0 = t0 * (prior where known, e elsewhere) + (1 - t0) * e, exactly the training distribution;
        "mixed" = known entries t0 * prior (zero-noise mean), unknown entries pure noise e;
        "zeros" = t0 * prior with 0 where unknown (WRONG at t0 > 0: an encoded 0 is "a point at the pallet origin"; kept for reference).
        t0 = 0 / no prior reproduces deterministic single-frame inference from z = 0."""
        steps = steps or self.steps
        img = image * 2.0 - 1.0
        B, _, H, W = img.shape
        ps = self.net.patch_size
        y = self.net.extract_y_embedding(img, H // ps, W // ps)
        if x_prior is None or t0 <= 0.0:
            z, t_start = torch.zeros(B, T.NUM_CONT, H, W, device=img.device, dtype=img.dtype), 0.0
        else:
            prior = x_prior.to(img.dtype)
            mask = prior_mask if prior_mask is not None else torch.ones_like(prior, dtype=torch.bool)
            e = torch.randn(prior.shape, device=prior.device, dtype=prior.dtype, generator=generator) * self.noise_scale
            if prior_fill == "noise":
                z = t0 * torch.where(mask, prior, e) + (1.0 - t0) * e
            elif prior_fill == "mixed":
                z = torch.where(mask, t0 * prior, e)
            elif prior_fill == "zeros":
                z = t0 * torch.where(mask, prior, torch.zeros_like(prior))
            else:
                raise ValueError(prior_fill)
            t_start = float(t0)
        ts = torch.linspace(t_start, 1.0, steps + 1, device=img.device)
        for i in range(steps):
            t = ts[i]
            out = self.net(z, t.expand(B), img, cached_y_emb=y)
            x_pred, logits = out[:, :T.NUM_CONT], out[:, T.NUM_CONT:]
            z = x_pred if i == steps - 1 else z + (ts[i + 1] - t) * (x_pred - z) / (1.0 - t)
        return z.float(), logits.float()
