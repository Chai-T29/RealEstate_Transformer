import torch
import torch.nn as nn
import torch.nn.functional as func
import pytorch_lightning as pl
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta


def grad_hook(module, grad_input, grad_output):
    """Grad hook.
    
    Args:
        module: Module instance.
        grad_input: Gradients of inputs.
        grad_output: Gradients of outputs.
    Returns:
        None.
    """
    if grad_output and grad_output[0] is not None:
        grad = grad_output[0]
        print(f"[{module.__class__.__name__}] grad norm: {grad.norm().item():.6f}, "
              f"mean: {grad.mean().item():.6f}, max: {grad.max().item():.6f}")


def check_for_nans(name: str, tensor: torch.Tensor):
    """Check tensor for NaNs/Infs.
    
    Args:
        name (str): Identifier.
        tensor (torch.Tensor): Tensor of any shape.
    Returns:
        None.
    """
    print(f"[DEBUG] Checking {name}: shape={tuple(tensor.shape)}", end="")
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    if has_nan or has_inf:
        print(f" --> NaN={has_nan}, Inf={has_inf}")
    else:
        print(" --> OK")


def print_stats(name: str, tensor: torch.Tensor):
    """Print basic tensor stats.
    
    Args:
        name (str): Identifier.
        tensor (torch.Tensor): Tensor of any shape.
    Returns:
        None.
    """
    tensor_flat = tensor.flatten()
    print(f"{name} stats: min: {tensor.min().item():.6f}, max: {tensor.max().item():.6f}, "
          f"mean: {tensor.mean().item():.6f}, median: {tensor_flat.median().item():.6f}, std: {tensor.std().item():.6f}")


class FocalHuberLoss(nn.Module):
    def __init__(self, delta=1.0, gamma=1.0, reduction='mean'):
        """
        Args:
            delta (float): Threshold.
            gamma (float): Focusing parameter.
            reduction (str): 'mean' or 'sum'.
        """
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        """
        Args:
            input (torch.Tensor): Predictions.
            target (torch.Tensor): Ground truth.
        Returns:
            torch.Tensor: Loss value.
        """
        error = input - target
        abs_error = torch.abs(error)
        huber = torch.where(
            abs_error < self.delta,
            0.5 * error ** 2,
            self.delta * (abs_error - 0.5 * self.delta)
        )
        modulating_factor = (abs_error / self.delta + 1e-6) ** self.gamma
        focal_huber = modulating_factor * huber

        if self.reduction == 'mean':
            return focal_huber.mean()
        elif self.reduction == 'sum':
            return focal_huber.sum()
        else:
            return focal_huber


class HousingTransformer(pl.LightningModule):
    def __init__(
        self,
        features,
        property_types,
        window_size,
        embed_dim,
        dim_feedforward,
        num_layers_encoder,
        num_heads_encoder,
        num_layers_decoder,
        num_heads_decoder,
        lr=1e-4,
        lr_patience=5,
        lr_decay=0.1,
        dropout=0.1,
        betas=(0.9, 0.999),
        loss_type="huber",
        delta=1.0,
        gamma=0.0,
        shift=10.0,
        prediction_window=[1, 3, 12],
        check_gradients=False,
        apply_spatial_attention=True,
        print_output_stats=False,
        data_imputed=True
    ):
        """
        Args:
            features (list): Feature names (F features).
            property_types (list): Property types (P types).
            window_size (int): Temporal window length (T).
            embed_dim (int): Embedding dimension.
            dim_feedforward (int): Feedforward layer dim.
            num_layers_encoder (int): Encoder layers.
            num_heads_encoder (int): Encoder heads.
            num_layers_decoder (int): Decoder layers.
            num_heads_decoder (int): Decoder heads.
            lr (float): Learning rate.
            lr_patience (int): Scheduler patience.
            lr_decay (float): Decay factor.
            dropout (float): Dropout rate.
            betas (tuple): Optimizer betas.
            loss_type (str): "huber" or "focal".
            delta (float): Delta for loss.
            gamma (float): Gamma for focal loss.
            shift (float): Shift value.
            prediction_window (list/int): Prediction time offsets.
            check_gradients (bool): Register grad hooks.
            apply_spatial_attention (bool): Apply spatial attention.
            print_output_stats (bool): Print outputs stats.
            data_imputed (bool): Mode for imputed data.
        Returns:
            None.
        """
        super().__init__()
        self.save_hyperparameters()
        self.features = features
        num_features = len(self.features)         # F features
        self.property_types = property_types
        num_property_types = len(self.property_types)  # P types
        self.window_size = window_size             # T time steps
        self.embed_dim = embed_dim
        self.prediction_window = prediction_window if isinstance(prediction_window, list) else [prediction_window]
        self.pred_window_len = len(self.prediction_window)
        self.lr = lr
        self.lr_patience = lr_patience
        self.lr_decay = lr_decay
        self.betas = betas
        self.loss_fn = (FocalHuberLoss(delta=delta, gamma=gamma, reduction='mean')
                        if loss_type == "focal"
                        else nn.HuberLoss(reduction="mean", delta=delta))
        self.delta = delta
        self.shift = shift
        self.apply_spatial_attention = apply_spatial_attention
        self.print_output_stats = print_output_stats
        self.data_imputed = data_imputed
        self.test_step_outputs = []
        self.eps = 1e-5

        # Compress property types: [B, T, P] -> [B, T, F]
        self.compress = nn.Sequential(
            nn.Linear(num_property_types, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 1)
        )
        # Zipcode embedding: [B] -> [B, F] -> expanded to [B, T, F]
        self.zip_emb = nn.Embedding(24488, num_features)
        # Feature embedding: [B, T, 2F+1] -> [B, T, embed_dim]
        self.emb = nn.Sequential(
            nn.Linear(2 * num_features + 1, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, embed_dim)
        )
        # Encoder positional embedding: [T] -> [T, embed_dim]
        self.encoder_pos_emb = nn.Embedding(window_size, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads_encoder,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=nn.ReLU(),
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers_encoder)
        # Optional spatial attention: [B, embed_dim, T] -> [B, embed_dim, T]
        if self.apply_spatial_attention:
            self.spatial_attention = nn.MultiheadAttention(
                embed_dim=window_size,
                num_heads=1,
                dropout=dropout,
                batch_first=True
            )
        # Decoder projection: [B, T, embed_dim] -> [B, T, P+1]
        self.dec_projection = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, num_property_types + 1)
        )
        # Decoder positional embedding: embed positions to dimension P+1
        max_pred = max(self.prediction_window)
        self.decoder_pos_emb = nn.Embedding(max_pred + 1, num_property_types + 1)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=num_property_types + 1,
            nhead=num_heads_decoder,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=nn.ReLU(),
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers_decoder)

        if check_gradients:
            self.register_full_backward_hook(grad_hook)
            self.compress[-1].register_full_backward_hook(grad_hook)
            self.emb[-1].register_full_backward_hook(grad_hook)
            self.transformer_encoder.register_full_backward_hook(grad_hook)
            self.transformer_decoder.register_full_backward_hook(grad_hook)
            self.dec_projection.register_full_backward_hook(grad_hook)
    
    def zero_aware_norm(self, x: torch.Tensor, store_stats_for_idx: int = None, shift: float = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, T, F, P].
            store_stats_for_idx (int, optional): Feature index.
            shift (float, optional): Shift value.
        Returns:
            torch.Tensor: Normalized tensor [B, T, F, P].
        """
        if shift is None:
            shift = self.shift
        mask = (x != 0)
        if mask.sum() == 0:
            raise ValueError("Skipping batch: no nonzero values in housing input.")
        reduction_dims = [d for d in range(x.ndim) if d != 2]  # Reduce all but feature dim.
        count = mask.sum(dim=reduction_dims)
        sum_val = (x * mask).sum(dim=reduction_dims)
        mean_val = sum_val / (count + self.eps)
        sum_sq = ((x * mask) ** 2).sum(dim=reduction_dims)
        var_val = sum_sq / (count + self.eps) - mean_val ** 2
        std_val = var_val.sqrt()
        
        if store_stats_for_idx is not None:
            if shift == 0.0:
                self.last_mean = mean_val[store_stats_for_idx].unsqueeze(0)
                self.last_std = std_val[store_stats_for_idx].unsqueeze(0)
            else:
                self.mean_exp = mean_val[store_stats_for_idx].unsqueeze(0)
                self.std_exp = std_val[store_stats_for_idx].unsqueeze(0)
        
        expand_shape = [1] * x.ndim
        expand_shape[2] = -1  # Expand along F dimension.
        mean_val_exp = mean_val.view(expand_shape)
        std_val_exp = std_val.view(expand_shape)
        
        x_norm = torch.where(mask, (x - mean_val_exp) / (std_val_exp + self.eps) + shift, x)
        x_norm = torch.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)
        return x_norm

    def rental_zero_aware_norm(self, rental: torch.Tensor, shift: float = None) -> torch.Tensor:
        """
        Args:
            rental (torch.Tensor): [B, T].
            shift (float, optional): Shift value.
        Returns:
            torch.Tensor: Normalized rental tensor [B, T].
        """
        if shift is None:
            shift = self.shift
        mask = (rental != 0)
        if mask.sum() == 0:
            raise ValueError("Skipping batch: no nonzero values in rental input.")
        reduction_dims = [0, 1]
        count = mask.sum(dim=reduction_dims)
        sum_val = (rental * mask).sum(dim=reduction_dims)
        mean_val = sum_val / (count + self.eps)
        sum_sq = ((rental * mask) ** 2).sum(dim=reduction_dims)
        var_val = sum_sq / (count + self.eps) - mean_val ** 2
        std_val = var_val.sqrt()
        
        if self.data_imputed:
            self.rental_last_mean = mean_val.unsqueeze(0)
            self.rental_last_std = std_val.unsqueeze(0)
        else:
            self.rental_mean_exp = mean_val.unsqueeze(0)
            self.rental_std_exp = std_val.unsqueeze(0)
        
        mean_val_exp = mean_val.view(1, 1)
        std_val_exp = std_val.view(1, 1)
        rental_norm = torch.where(mask, (rental - mean_val_exp) / (std_val_exp + self.eps) + shift, rental)
        rental_norm = torch.nan_to_num(rental_norm, nan=0.0, posinf=0.0, neginf=0.0)
        return rental_norm

    def forward(self, batch):
        """
        Args:
            batch (dict): Contains:
                "housing_x": [B, T, F, P],
                "housing_y": [B, pred_window_len, P] (optional),
                "rental_x": [B, T],
                "rental_y": [B, pred_window_len],
                "zipcode_idx": [B],
                "time": Timestamp info.
        Returns:
            If teacher forcing branch:
                raw_preds: [B, (pred_window_len + 1), P+1],
                (vis_preds_housing: [B, pred_window_len, P], vis_preds_rental: [B, pred_window_len, 1]),
                y_normed: [B, pred_window_len+1, P+1].
            Else:
                raw_preds: [B, pred_window_len, P+1],
                (vis_preds_housing, vis_preds_rental).
        """
        housing_x = batch["housing_x"]  # [B, T, F, P]
        housing_y = batch.get("housing_y", None)
        rental_x = batch["rental_x"]      # [B, T]
        B, T, F, P = housing_x.shape
        
        ms_price_idx = (self.features == "median_sale_price").nonzero()[0].item()
    
        if self.data_imputed:
            housing_x = self.zero_aware_norm(housing_x, store_stats_for_idx=ms_price_idx, shift=0.0)
        else:
            housing_x = self.zero_aware_norm(housing_x, store_stats_for_idx=ms_price_idx, shift=self.shift)
    
        if self.data_imputed:
            rental_x = self.rental_zero_aware_norm(rental_x, shift=0.0)
        else:
            rental_x = self.rental_zero_aware_norm(rental_x, shift=self.shift)
    
        x_compressed = self.compress(housing_x).squeeze(-1)  # [B, T, F]
        
        zipcode = batch["zipcode_idx"]
        zip_emb = self.zip_emb(zipcode)  # [B, F]
        zip_emb = zip_emb.unsqueeze(1).expand(B, T, -1)  # [B, T, F]
        
        rental_x_unsq = rental_x.unsqueeze(-1)  # [B, T, 1]
        x_cat = torch.cat([x_compressed, zip_emb, rental_x_unsq], dim=2)  # [B, T, 2F+1]
        emb_x = self.emb(x_cat)  # [B, T, embed_dim]
    
        pos_indices = torch.arange(T, device=housing_x.device, dtype=torch.long)
        pos_enc = self.encoder_pos_emb(pos_indices).unsqueeze(0).expand(B, T, self.embed_dim)
        emb_x = emb_x + pos_enc
    
        enc = self.transformer_encoder(emb_x)  # [B, T, embed_dim]
        
        if self.apply_spatial_attention:
            enc = enc.permute(0, 2, 1)  # [B, embed_dim, T]
            attn_out, _ = self.spatial_attention(query=enc, key=enc, value=enc)
            enc = attn_out.permute(0, 2, 1)  # [B, T, embed_dim]
    
        memory = self.dec_projection(enc)  # [B, T, P+1]
    
        if housing_y is not None:
            # Teacher Forcing Branch
            original_housing = housing_y.squeeze(2)  # [B, pred_window_len, P]
            nonzero_mask_housing = original_housing != 0
            if self.data_imputed:
                scaled_housing = (original_housing - self.last_mean) / (self.last_std + self.eps)
                norm_housing_y = torch.where(nonzero_mask_housing, scaled_housing,
                                             torch.randn_like(scaled_housing, requires_grad=True) * 0.1)
            else:
                scaled_housing = (original_housing - self.mean_exp) / (self.std_exp + self.eps) + self.shift
                norm_housing_y = torch.where(nonzero_mask_housing, scaled_housing, original_housing)
    
            rental_y = batch["rental_y"].unsqueeze(2)  # [B, pred_window_len, 1]
            nonzero_mask_rental = rental_y != 0
            if self.data_imputed:
                scaled_rental = (rental_y - self.rental_last_mean) / (self.rental_last_std + self.eps)
                norm_rental_y = torch.where(nonzero_mask_rental, scaled_rental,
                                            torch.randn_like(scaled_rental, requires_grad=True) * 0.1)
            else:
                scaled_rental = (rental_y - self.rental_mean_exp) / (self.rental_std_exp + self.eps) + self.shift
                norm_rental_y = torch.where(nonzero_mask_rental, scaled_rental, rental_y)
    
            y_normed = torch.cat([norm_housing_y, norm_rental_y], dim=2)  # [B, pred_window_len, P+1]
    
            housing_prior = batch["housing_x"][:, -1, ms_price_idx, :].unsqueeze(1)  # [B, 1, P]
            if self.data_imputed:
                housing_prior_norm = (housing_prior - self.last_mean) / (self.last_std + self.eps)
            else:
                housing_prior_norm = (housing_prior - self.mean_exp) / (self.std_exp + self.eps) + self.shift
    
            rental_prior = rental_x[:, -1].unsqueeze(1)  # [B, 1]
            if self.data_imputed:
                rental_prior_norm = (rental_prior - self.rental_last_mean) / (self.rental_last_std + self.eps)
            else:
                rental_prior_norm = (rental_prior - self.rental_mean_exp) / (self.rental_std_exp + self.eps) + self.shift
    
            prior_token_norm = torch.cat([housing_prior_norm, rental_prior_norm.unsqueeze(2)], dim=2)  # [B, 1, P+1]
            decoder_input = torch.cat([prior_token_norm, y_normed], dim=1)  # [B, pred_window_len+1, P+1]
            teacher_forcing_pos = torch.tensor([0] + self.prediction_window, device=y_normed.device, dtype=torch.long)
            pos_dec_emb = self.decoder_pos_emb(teacher_forcing_pos).unsqueeze(0)
            decoder_input = decoder_input + pos_dec_emb
                
            tgt_len = decoder_input.size(1)
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt_len).to(decoder_input.device)
            raw_preds = self.transformer_decoder(decoder_input, memory, tgt_mask=tgt_mask)
            
            if self.data_imputed:
                scaled_back_housing = raw_preds[..., :P] * (self.last_std + self.eps) + self.last_mean
                scaled_back_rental = raw_preds[..., P:] * (self.rental_last_std + self.eps) + self.rental_last_mean
                vis_preds_housing = torch.where(raw_preds[..., :P] > 0, scaled_back_housing, torch.tensor(0.0, device=raw_preds.device))
                vis_preds_rental = torch.where(raw_preds[..., P:] > 0, scaled_back_rental, torch.tensor(0.0, device=raw_preds.device))
            else:
                scaled_back_housing = (raw_preds[..., :P] - self.shift) * (self.std_exp + self.eps) + self.mean_exp
                scaled_back_rental = (raw_preds[..., P:] - self.shift) * (self.rental_std_exp + self.eps) + self.rental_mean_exp
                vis_preds_housing = torch.where((raw_preds[..., :P] - self.shift) > 0, scaled_back_housing, torch.tensor(0.0, device=raw_preds.device))
                vis_preds_rental = torch.where((raw_preds[..., P:] - self.shift) > 0, scaled_back_rental, torch.tensor(0.0, device=raw_preds.device))
    
            y_normed = torch.cat([y_normed, torch.zeros((B, 1, P + 1), device=y_normed.device, dtype=y_normed.dtype)], dim=1)
            return raw_preds, (vis_preds_housing[:, :-1, :], vis_preds_rental[:, :-1, :]), y_normed
    
        else:
            # AR Branch
            housing_prior = batch["housing_x"][:, -1, ms_price_idx, :].unsqueeze(1)  # [B, 1, P]
            if self.data_imputed:
                housing_prior_norm = (housing_prior - self.last_mean) / (self.last_std + self.eps)
            else:
                housing_prior_norm = (housing_prior - self.mean_exp) / (self.std_exp + self.eps) + self.shift
    
            rental_prior = rental_x[:, -1].unsqueeze(1)  # [B, 1]
            if self.data_imputed:
                rental_prior_norm = (rental_prior - self.rental_last_mean) / (self.rental_last_std + self.eps)
            else:
                rental_prior_norm = (rental_prior - self.rental_mean_exp) / (self.rental_std_exp + self.eps) + self.shift
    
            prior_token = torch.cat([housing_prior_norm, rental_prior_norm.unsqueeze(2)], dim=2)  # [B, 1, P+1]
    
            B = prior_token.size(0)
            out_seq = torch.zeros(B, self.pred_window_len + 1, prior_token.size(2), device=prior_token.device, dtype=prior_token.dtype)
            raw_preds = torch.zeros_like(out_seq)
    
            pos_indices = torch.tensor([0] + self.prediction_window, device=prior_token.device, dtype=torch.long)
            pos_emb = self.decoder_pos_emb(pos_indices).unsqueeze(0).expand(B, -1, -1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(pos_indices)).to(prior_token.device)
    
            for t in range(self.pred_window_len + 1):
                out_seq[:, t, :] = prior_token.squeeze(1)
                decoder_input = out_seq + pos_emb[:, t:t+1, :]
                decoder_out = self.transformer_decoder(decoder_input, memory, tgt_mask=tgt_mask)
                prior_token = decoder_out[:, t:t+1, :]
                if not self.data_imputed:
                    prior_token = torch.where((prior_token - self.shift) > 0, prior_token, torch.tensor(0.0, device=prior_token.device))
                raw_preds[:, t, :] = prior_token.squeeze(1)
            
            raw_preds = raw_preds[:, :-1, :]
    
            if self.data_imputed:
                vis_preds_housing = raw_preds[..., :P] * (self.last_std + self.eps) + self.last_mean
                vis_preds_rental = raw_preds[..., P:] * (self.rental_last_std + self.eps) + self.rental_last_mean
            else:
                scaled_back_housing = (raw_preds[..., :P] - self.shift) * (self.std_exp + self.eps) + self.mean_exp
                scaled_back_rental = (raw_preds[..., P:] - self.shift) * (self.rental_std_exp + self.eps) + self.rental_mean_exp
                vis_preds_housing = torch.where((raw_preds[..., :P] - self.shift) > 0, scaled_back_housing, torch.tensor(0.0, device=raw_preds.device))
                vis_preds_rental = torch.where((raw_preds[..., P:] - self.shift) > 0, scaled_back_rental, torch.tensor(0.0, device=raw_preds.device))
    
            return raw_preds, (vis_preds_housing, vis_preds_rental)
        
    def training_step(self, batch, batch_idx):
        """
        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.
        Returns:
            torch.Tensor: Training loss.
        """
        raw_preds, (housing_vis_preds, rental_vis_preds), combined_y_normed = self(batch)
        loss_main = self.loss_fn(raw_preds, combined_y_normed)
        self.log("train_loss", loss_main, prog_bar=True)

        if self.print_output_stats and batch_idx % 200 == 0:
            orig_housing = batch["housing_y"].squeeze(2)  # [B, pred_window_len, P]
            orig_rental = batch["rental_y"].unsqueeze(2)     # [B, pred_window_len, 1]
            print_stats("combined_y_normed", combined_y_normed)
            print_stats("raw_preds", raw_preds)
            print_stats("Housing - y_actual", orig_housing)
            print_stats("Rental  - y_actual", orig_rental)
            print_stats("Housing - vis_preds", housing_vis_preds)
            print_stats("Rental  - vis_preds", rental_vis_preds)

        return loss_main

    def validation_step(self, batch, batch_idx):
        """
        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.
        Returns:
            torch.Tensor: Validation loss.
        """
        raw_preds, (housing_vis_preds, rental_vis_preds) = self({k: batch[k] for k in batch.keys() if k != "housing_y"})
        housing_y = batch["housing_y"]
        rental_y = batch["rental_y"].unsqueeze(2)
        original_housing = housing_y.squeeze(2)

        nonzero_mask_housing = original_housing != 0
        if self.data_imputed:
            scaled_housing = (original_housing - self.last_mean) / (self.last_std + self.eps)
            norm_housing_y = torch.where(nonzero_mask_housing, scaled_housing,
                                         torch.randn_like(scaled_housing, requires_grad=True) * 0.1)
        else:
            scaled_housing = (original_housing - self.mean_exp) / (self.std_exp + self.eps) + self.shift
            norm_housing_y = torch.where(nonzero_mask_housing, scaled_housing, original_housing)

        nonzero_mask_rental = rental_y != 0
        if self.data_imputed:
            scaled_rental = (rental_y - self.rental_last_mean) / (self.rental_last_std + self.eps)
            norm_rental_y = torch.where(nonzero_mask_rental, scaled_rental,
                                         torch.randn_like(scaled_rental, requires_grad=True) * 0.1)
        else:
            scaled_rental = (rental_y - self.rental_mean_exp) / (self.rental_std_exp + self.eps) + self.shift
            norm_rental_y = torch.where(nonzero_mask_rental, scaled_rental, rental_y)

        combined_y_normed = torch.cat([norm_housing_y, norm_rental_y], dim=2)
        loss_main = self.loss_fn(raw_preds, combined_y_normed)
        self.log("val_loss", loss_main, prog_bar=True)
        
        if self.print_output_stats:
            print_stats("Combined y_normed", combined_y_normed)
            print_stats("Raw preds", raw_preds)
            print_stats("Housing y_actual", original_housing)
            print_stats("Rental y_actual", rental_y)
            print_stats("Housing vis_preds", housing_vis_preds)
            print_stats("Rental vis_preds", rental_vis_preds)
            
        mask_housing = original_housing != 0
        if mask_housing.any():
            if batch_idx % 100 == 0:
                housing_rmse = torch.sqrt(((housing_vis_preds[mask_housing] - original_housing[mask_housing]) ** 2).mean()).item()
                housing_mae = torch.mean(torch.abs(housing_vis_preds[mask_housing] - original_housing[mask_housing])).item()
                housing_mape = (torch.mean(torch.abs((housing_vis_preds[mask_housing] - original_housing[mask_housing]) / original_housing[mask_housing])) * 100.0).item()
                ss_res = ((housing_vis_preds[mask_housing] - original_housing[mask_housing]) ** 2).sum().item()
                ss_tot = ((original_housing[mask_housing] - original_housing[mask_housing].mean()) ** 2).sum().item() + self.eps
                housing_r2 = 1 - ss_res / ss_tot
                print(f"Validation Housing Metrics - RMSE: {housing_rmse:.6f}, MAE: {housing_mae:.6f}, R^2: {housing_r2:.6f}, MAPE: {housing_mape:.2f}%")
        
        mask_rental = rental_y != 0
        if mask_rental.any():
            if batch_idx % 100 == 0:
                rental_rmse = torch.sqrt(((rental_vis_preds[mask_rental] - rental_y[mask_rental]) ** 2).mean()).item()
                rental_mae = torch.mean(torch.abs(rental_vis_preds[mask_rental] - rental_y[mask_rental])).item()
                rental_mape = (torch.mean(torch.abs((rental_vis_preds[mask_rental] - rental_y[mask_rental]) / rental_y[mask_rental])) * 100.0).item()
                ss_res = ((rental_vis_preds[mask_rental] - rental_y[mask_rental]) ** 2).sum().item()
                ss_tot = ((rental_y[mask_rental] - rental_y[mask_rental].mean()) ** 2).sum().item() + self.eps
                rental_r2 = 1 - ss_res / ss_tot
                print(f"Validation Rental Metrics  - RMSE: {rental_rmse:.6f}, MAE: {rental_mae:.6f}, R^2: {rental_r2:.6f}, MAPE: {rental_mape:.2f}%")
            
        return loss_main

    def test_step(self, batch, batch_idx):
        """
        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.
        Returns:
            None.
        """
        raw_preds, (housing_vis_preds, rental_vis_preds) = self(batch)
        
        last_timestamp = batch["time"][-1].detach().item()
        last_datetime = datetime.fromtimestamp(last_timestamp)
        forecast_times = [last_datetime + relativedelta(months=offset)
                          for offset in self.prediction_window]
        
        if self.print_output_stats:
            print_stats("Raw preds", raw_preds)
            print_stats("Housing vis_preds", housing_vis_preds)
            print_stats("Rental vis_preds", rental_vis_preds)
        
        item = {
            "zipcodes": batch["zipcode"],
            "forecast_times": forecast_times,
            "housing_vis_preds": housing_vis_preds,
            "rental_vis_preds": rental_vis_preds,
        }
        self.test_step_outputs.append(item)
        return None

    def on_test_epoch_end(self):
        """
        Args:
            None.
        Returns:
            None.
        """
        zipcodes = []
        for o in self.test_step_outputs:
            zipcodes.extend(o["zipcodes"])
        time = self.test_step_outputs[0]["forecast_times"]
        housing_vis = torch.cat([o["housing_vis_preds"] for o in self.test_step_outputs], dim=0)
        rental_vis = torch.cat([o["rental_vis_preds"] for o in self.test_step_outputs], dim=0)
        self.test_results = {
            "zipcodes": zipcodes,
            "time": np.array(time),
            "housing_pred": housing_vis.detach().cpu().numpy(),
            "rental_pred": rental_vis.detach().cpu().numpy()
        }

    def configure_optimizers(self):
        """
        Args:
            None.
        Returns:
            dict: Optimizer and LR scheduler configuration.
        """
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=self.betas
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=self.lr_decay,
            patience=self.lr_patience,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}