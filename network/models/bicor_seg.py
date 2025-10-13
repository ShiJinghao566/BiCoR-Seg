import warnings
warnings.filterwarnings("ignore")

import math
from typing import Tuple, Type, List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_base


def compute_contrastive_loss(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    B, num_classes, _ = x.shape
    x = x.reshape(-1, num_classes, x.shape[-1])
    class_means = torch.mean(x, dim=0)
    diff_intra = x - class_means.unsqueeze(0)
    squared_diff_intra = torch.sum(diff_intra**2, dim=2)
    intra_loss = torch.mean(squared_diff_intra)
    diff_inter = class_means.unsqueeze(1) - class_means.unsqueeze(0)
    squared_diff_inter = torch.sum(diff_inter**2, dim=2)
    triu_indices = torch.triu_indices(num_classes, num_classes, offset=1)
    inter_distances = squared_diff_inter[triu_indices[0], triu_indices[1]]
    inter_loss = torch.mean(inter_distances)
    inter_loss = torch.clamp(inter_loss, min=1e-4)
    loss = intra_loss / (inter_loss + eps)
    return loss


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class FeatureMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class DictionaryInitializer(nn.Module):
    def __init__(self, num_classes: int, dict_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.dict_dim = dict_dim
        self.class_embed = nn.Embedding(num_classes, dict_dim)
        nn.init.trunc_normal_(self.class_embed.weight, std=0.02)


class DictFeatInteractorLayer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        dict_dim: int,
        *,
        inner_steps: int = 1,
        lambda_film: float = 0.5,
        tau: float = 1.0,
        eps: float = 1e-6,
        select_mode: str = "topk",
        topk_ratio: float = 1,
        min_pixels: int = 64,
        use_percentile: bool = False,
        percentile: float = 0.98,
        thresh: float = 0.5,
    ):
        super().__init__()
        self.N = num_classes
        self.Cf = feat_dim
        self.Cd = dict_dim
        self.Steps = max(1, inner_steps)
        self.lambda_film = lambda_film
        self.tau = tau
        self.eps = eps
        self.select_mode = select_mode
        self.topk_ratio = topk_ratio
        self.min_pixels = min_pixels
        self.use_percentile = use_percentile
        self.percentile = percentile
        self.thresh = thresh
        self.dict_to_feat = nn.Linear(self.Cd, self.Cf, bias=False)
        self.feat_to_dict = nn.Conv2d(self.Cf, self.Cd, kernel_size=1, bias=False)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.Cd * 2, self.Cd),
            nn.GELU(),
            nn.Linear(self.Cd, self.Cd),
            nn.Sigmoid(),
        )
        self.cand_mlp = nn.Sequential(
            nn.Linear(self.Cd, self.Cd),
            nn.GELU(),
            nn.Linear(self.Cd, self.Cd),
        )
        self.gamma_proj = nn.Linear(self.Cd, self.Cf)
        self.beta_proj  = nn.Linear(self.Cd, self.Cf)
        self.feat_ln = nn.LayerNorm(self.Cf)
        self.dict_ln = nn.LayerNorm(self.Cd)

    @torch.no_grad()
    def _build_mask_weights(self, H_flat: torch.Tensor) -> torch.Tensor:
        B, N, S = H_flat.shape
        if self.select_mode == "topk":
            k = max(self.min_pixels, int(round(self.topk_ratio * S)))
            k = min(k, S)
            vals, idx = torch.topk(H_flat, k=k, dim=-1)
            mask = torch.zeros_like(H_flat)
            mask.scatter_(-1, idx, 1.0)
            w_flat = H_flat * mask
        elif self.select_mode == "threshold":
            if self.use_percentile:
                thr = torch.quantile(H_flat, q=self.percentile, dim=-1, keepdim=True)
            else:
                thr = torch.full_like(H_flat[..., :1], fill_value=self.thresh)
            mask = (H_flat >= thr).float()
            w_flat = H_flat * mask
        else:
            w_flat = H_flat
        sum_w = w_flat.sum(dim=-1, keepdim=True)
        fallback = (sum_w <= self.eps)
        w_all = H_flat / (H_flat.sum(dim=-1, keepdim=True) + self.eps)
        w_flat = torch.where(fallback, w_all, w_flat / (sum_w + self.eps))
        return w_flat

    def forward(self, D: torch.Tensor, E: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, Cd = D.shape
        _, Cf, H, W = E.shape
        assert N == self.N and Cf == self.Cf and Cd == self.Cd
        H_logits_last = None
        for _ in range(self.Steps):
            Dn = self.dict_ln(D)
            En = self.feat_ln(E.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, Cf, H, W)
            D_proj = self.dict_to_feat(Dn)
            score = torch.einsum('bnc,bchw->bnhw', D_proj, En)
            score = score / math.sqrt(self.Cf)
            logits = score / self.tau
            H_logits_last = logits
            H_sig = torch.sigmoid(score)
            H_soft = torch.softmax(logits, dim=1)
            V = self.feat_to_dict(En)
            H_flat = H_sig.view(B, N, -1)
            with torch.no_grad():
                w_flat = self._build_mask_weights(H_flat)
            w = w_flat.view(B, N, H, W)
            Cn = torch.einsum('bnhw,bchw->bnc', w, V)
            gate_in = torch.cat([Dn, Cn], dim=-1)
            G = self.gate_mlp(gate_in)
            Cand = self.cand_mlp(Cn)
            D = G * Cand + (1.0 - G) * D
            gamma = 1.0 + torch.tanh(self.gamma_proj(D))
            beta  = self.beta_proj(D)
            E_exp = E.unsqueeze(1)
            gamma_b = gamma.view(B, N, Cf, 1, 1)
            beta_b  = beta.view(B, N, Cf, 1, 1)
            E_class = gamma_b * E_exp + beta_b
            E_new = (H_soft.unsqueeze(2) * E_class).sum(dim=1)
            E = (1.0 - self.lambda_film) * E + self.lambda_film * E_new
        return D, E, H_logits_last


class DictFeatInteractorStack(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_classes: int,
        feat_dim: int,
        dict_dim: int,
        *,
        layer_configs: Optional[List[Dict]] = None,
        default_inner_steps: int = 1,
        default_select_mode: str = "topk",
        default_topk_ratio: float = 1,
        default_min_pixels: int = 64,
        default_use_percentile: bool = False,
        default_percentile: float = 0.98,
        default_thresh: float = 0.5,
        lambda_film: float = 0.5,
        tau: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert num_layers >= 1
        self.layers = nn.ModuleList()
        if layer_configs is None:
            layer_configs = []
            for _ in range(num_layers):
                layer_configs.append(dict(
                    inner_steps=default_inner_steps,
                    select_mode=default_select_mode,
                    topk_ratio=default_topk_ratio,
                    min_pixels=default_min_pixels,
                    use_percentile=default_use_percentile,
                    percentile=default_percentile,
                    thresh=default_thresh,
                ))
        else:
            assert len(layer_configs) == num_layers
        for i in range(num_layers):
            cfg = layer_configs[i]
            layer = DictFeatInteractorLayer(
                num_classes=num_classes,
                feat_dim=feat_dim,
                dict_dim=dict_dim,
                inner_steps=cfg.get("inner_steps", default_inner_steps),
                lambda_film=lambda_film,
                tau=tau,
                eps=eps,
                select_mode=cfg.get("select_mode", default_select_mode),
                topk_ratio=cfg.get("topk_ratio", default_topk_ratio),
                min_pixels=cfg.get("min_pixels", default_min_pixels),
                use_percentile=cfg.get("use_percentile", default_use_percentile),
                percentile=cfg.get("percentile", default_percentile),
                thresh=cfg.get("thresh", default_thresh),
            )
            self.layers.append(layer)

    def forward(self, D0: torch.Tensor, E0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        D, E = D0, E0
        H_logits_per_layer: List[torch.Tensor] = []
        D_per_layer: List[torch.Tensor] = []
        for layer in self.layers:
            D, E, H_logits_last = layer(D, E)
            H_logits_per_layer.append(H_logits_last)
            D_per_layer.append(D)
        return D, E, H_logits_per_layer, D_per_layer


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        token_length: int,
        transformer_dim: int = 256,
        activation: Type[nn.Module] = nn.GELU,
        use_aux: bool = True,
        has_contrastive_loss: bool = False,
        num_layers: int = 2,
        layer_configs: Optional[List[Dict]] = None,
        num_stages: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.has_contrastive_loss = has_contrastive_loss
        self.use_aux = use_aux
        self.transformer_dim = transformer_dim
        self.token_length = token_length
        self.dict_init = DictionaryInitializer(token_length, transformer_dim)
        if num_stages is not None:
            num_layers = num_stages
        self.interactor_stack = DictFeatInteractorStack(
            num_layers=num_layers,
            num_classes=token_length,
            feat_dim=transformer_dim,
            dict_dim=transformer_dim,
            layer_configs=layer_configs,
        )
        assert transformer_dim % 16 == 0
        self.upsampler = nn.Sequential(
            nn.PixelShuffle(2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.PixelShuffle(2),
            activation(),
        )
        self.classifier_proj = nn.Linear(transformer_dim, transformer_dim // 16, bias=False)
        self.static_classifier = nn.Linear(transformer_dim, transformer_dim // 16, bias=False)

    def _inference_dict_interaction(self, image_embeddings: torch.Tensor):
        batch_size = image_embeddings.shape[0]
        static_D = self.dict_init.class_embed.weight
        D0 = static_D.unsqueeze(0).expand(batch_size, -1, -1)
        D_L, E_L, H_list, D_list = self.interactor_stack(D0, image_embeddings)
        up = self.upsampler(E_L)
        W_cls = self.classifier_proj(D_L)
        seg_output = torch.einsum('bnc,bchw->bnhw', W_cls, up)
        return seg_output, D_L, H_list, D_list

    def forward(self, image_embeddings: torch.Tensor):
        seg_output, D_last, H_list, D_list = self._inference_dict_interaction(
            image_embeddings=image_embeddings
        )
        if not self.training:
            return seg_output
        aux_output = None
        if self.use_aux:
            with torch.no_grad():
                static_D = self.dict_init.class_embed.weight.unsqueeze(0).expand(D_last.shape[0], -1, -1)
            up = self.upsampler(image_embeddings)
            static_W = self.static_classifier(static_D)
            aux_output = torch.einsum('bnc,bchw->bnhw', static_W, up)
        loss_con = None
        if self.has_contrastive_loss:
            loss_con = compute_contrastive_loss(D_last)
        return seg_output, aux_output, loss_con, H_list, D_list


class DynamicDictionaryLearning(nn.Module):
    def __init__(
        self,
        model: str,
        token_length: int,
        use_aux: bool = True,
        l: int = 2,
        embedding_dim: int = 256,
        has_contrastive_loss: bool = False,
        froze_backbone: bool = False,
        layer_configs: Optional[List[Dict]] = None,
    ):
        super().__init__()
        assert model == "convnext_base"
        self.model = model
        convnext = convnext_base(pretrained=True)
        self.backbone = nn.Sequential(*(list(convnext.children())[:-1]))
        self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
        self.multi_scale_features = []
        for name, module in self.backbone.named_modules():
            if name in self.target_layer_names:
                module.register_forward_hook(self.save_features_hook(name))
        hidden_sizes = [128, 256, 512, 1024]
        self.hidden_sizes = hidden_sizes
        num_encoder_blocks = 4
        decoder_hidden_size = embedding_dim
        mlps = []
        for i in range(num_encoder_blocks):
            mlp = FeatureMLP(input_dim=hidden_sizes[i], output_dim=embedding_dim)
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)
        self.decoder = Decoder(
            token_length=token_length,
            transformer_dim=embedding_dim,
            use_aux=use_aux,
            has_contrastive_loss=has_contrastive_loss,
            num_layers=l,
            layer_configs=layer_configs,
        )
        self.aggregator = nn.Conv2d(
            in_channels=decoder_hidden_size * num_encoder_blocks,
            out_channels=decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        if froze_backbone:
            self.backbone.requires_grad_(False)
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def save_features_hook(self, name):
        def hook(module, input, output):
            self.multi_scale_features.append(output)
        return hook

    def forward(self, x: torch.Tensor):
        self.multi_scale_features.clear()
        _ = self.backbone(x)
        batch_size = self.multi_scale_features[-1].shape[0]
        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(self.multi_scale_features, self.linear_c):
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(
                batch_size, -1, height, width
            )
            encoder_hidden_state = F.interpolate(
                encoder_hidden_state,
                size=self.multi_scale_features[0].size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            all_hidden_states += (encoder_hidden_state,)
        fused_states = self.aggregator(torch.cat(all_hidden_states[::-1], dim=1))
        out = self.decoder(
            image_embeddings=fused_states,
        )
        return out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_dim = 64
    token_length = 7
    layer_configs = [
        dict(inner_steps=1, select_mode="topk", topk_ratio=1,   min_pixels=64),
        dict(inner_steps=1, select_mode="topk", topk_ratio=1,   min_pixels=64),
        dict(inner_steps=1, select_mode="topk", topk_ratio=0.3, min_pixels=64),
    ]
    net = DynamicDictionaryLearning(
        model="convnext_base",
        token_length=token_length,
        l=3,
        embedding_dim=embedding_dim,
        has_contrastive_loss=True,
        layer_configs=layer_configs,
    ).to(device)
    fake_img = torch.randn(1, 3, 512, 512).to(device)
    net.train()
    out = net(fake_img)
