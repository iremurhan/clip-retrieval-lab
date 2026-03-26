"""
src/train.py
------------
Training engine for Cross-Modal Retrieval.

Usage:
    This module is typically invoked via `run.py` or the Slurm script `scripts/train.slurm`.
    
    python run.py --config configs/config_val.yaml
    
    The Trainer class manages:
    - Training loop with Mixed Precision (AMP)
    - Evaluation (R@K, MAP@K)
    - Checkpointing (best_model.pth, last_model.pth)
    - Logging (WandB, Tensorboard-style metrics)
"""

import torch
import logging
import warnings
import wandb
import os
import time
import random
from .metrics import AverageMeter, compute_recall_at_k, compute_map_at_k, compute_eccv_metrics, build_ranked_dicts, compute_mapr_rprecision
from .utils import compute_grad_norm
from .grad_cache import GradCache

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, use_wandb=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        self.use_wandb = use_wandb
        self.log_freq = config['logging']['log_freq']
        self.checkpoint_dir = config['logging']['checkpoint_dir']
        
        # Best score tracking for checkpointing and WandB summary
        self.best_r1 = 0.0
        self.best_metrics = {
            't2i_r1': 0.0, 't2i_r5': 0.0, 't2i_r10': 0.0,
            'i2t_r1': 0.0, 'i2t_r5': 0.0, 'i2t_r10': 0.0,
            't2i_map5': 0.0, 't2i_map10': 0.0,
            'i2t_map5': 0.0, 'i2t_map10': 0.0,
        }
        
        # Initialize Mixed Precision Training (AMP)
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed Precision Training (AMP) enabled.")
        else:
            self.scaler = None
            logger.info("Mixed Precision Training (AMP) disabled (CPU mode).")
        
        # Initialize Gradient Caching (STRICT CONFIG: will raise KeyError if missing)
        # Check if gradient caching is enabled
        self.use_grad_cache = config['training'].get('use_grad_cache', False)
        if self.use_grad_cache:
            loss_type = config.get('loss', {}).get('type', 'infonce').lower()
            if loss_type == 'siglip':
                raise NotImplementedError(
                    "SigLIPLoss is not compatible with GradCache. "
                    "GradCache calls criterion with positional img_aug_embeds/txt_aug_embeds args "
                    "but SigLIPLoss.bias gradient must flow through the full-batch logits. "
                    "Disable GradCache or use InfoNCE when use_grad_cache: true."
                )
            intra_img = config['loss'].get('intra_img_weight', 0.0)
            intra_txt = config['loss'].get('intra_txt_weight', 0.0)
            if intra_img > 0 or intra_txt > 0:
                raise ValueError(
                    "Intra-modal losses are not compatible with GradCache. "
                    "Set intra_img_weight: 0 and intra_txt_weight: 0 when use_grad_cache: true, "
                    "or disable GradCache."
                )
        if self.use_grad_cache:
            # STRICT: micro_batch_size MUST exist if grad_cache is enabled
            # GradCache constructor will raise KeyError if missing
            self.grad_cache = GradCache(
                model=self.model,
                criterion=self.criterion,
                config=self.config,
                device=self.device,
                scaler=self.scaler
            )
            logger.info(f"Gradient Caching enabled with micro_batch_size={config['training']['micro_batch_size']}")
        else:
            self.grad_cache = None
            logger.info("Gradient Caching disabled (standard training mode).")

        logger.info(
            f"Effective batch size: {config['training']['batch_size']} "
            f"(micro_batch_size: {config['training']['micro_batch_size']})"
        )

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize WandB run reference and define summary metrics
        self.wandb_run = None
        if self.use_wandb and wandb.run is not None:
            self.wandb_run = wandb.run
            # Define WandB summary metrics with max tracking for validation results
            # This ensures the dashboard always shows the best scores, even if current epoch is worse
            # Flickr metrics
            wandb.define_metric("val/t2i_r1", summary="max")
            wandb.define_metric("val/t2i_r5", summary="max")
            wandb.define_metric("val/t2i_r10", summary="max")
            wandb.define_metric("val/i2t_r1", summary="max")
            wandb.define_metric("val/i2t_r5", summary="max")
            wandb.define_metric("val/i2t_r10", summary="max")
            wandb.define_metric("val/t2i_mapr", summary="max")
            wandb.define_metric("val/i2t_mapr", summary="max")
            wandb.define_metric("val/t2i_rprecision", summary="max")
            wandb.define_metric("val/i2t_rprecision", summary="max")
            # COCO / ECCV metrics
            wandb.define_metric("val/coco_5k_r1_i2t", summary="max")
            wandb.define_metric("val/coco_5k_r1_t2i", summary="max")
            wandb.define_metric("val/coco_5k_r5_i2t", summary="max")
            wandb.define_metric("val/coco_5k_r5_t2i", summary="max")
            wandb.define_metric("val/coco_5k_r10_i2t", summary="max")
            wandb.define_metric("val/coco_5k_r10_t2i", summary="max")
            wandb.define_metric("val/coco_1k_r1_i2t", summary="max")
            wandb.define_metric("val/coco_1k_r1_t2i", summary="max")
            wandb.define_metric("val/coco_1k_r5_i2t", summary="max")
            wandb.define_metric("val/coco_1k_r5_t2i", summary="max")
            wandb.define_metric("val/coco_1k_r10_i2t", summary="max")
            wandb.define_metric("val/coco_1k_r10_t2i", summary="max")
            wandb.define_metric("val/eccv_r1_i2t", summary="max")
            wandb.define_metric("val/eccv_r1_t2i", summary="max")
            wandb.define_metric("val/eccv_map_at_r_i2t", summary="max")
            wandb.define_metric("val/eccv_map_at_r_t2i", summary="max")
            wandb.define_metric("val/eccv_rprecision_i2t", summary="max")
            wandb.define_metric("val/eccv_rprecision_t2i", summary="max")
            wandb.define_metric("val/cxc_r1_i2t", summary="max")
            wandb.define_metric("val/cxc_r1_t2i", summary="max")


    def load_checkpoint(self, checkpoint_path):
        """
        Loads full training state from a checkpoint file.
        Returns start_epoch.
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load Weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state if available (for AMP)
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info("GradScaler state loaded from checkpoint.")
        
        # Load State Info
        if 'best_r1' not in checkpoint or 'epoch' not in checkpoint:
            logger.error(f"Checkpoint file {checkpoint_path} is missing critical keys ('epoch' or 'best_r1').")
            logger.error("Cannot resume training safely. Aborting.")
            raise KeyError("Invalid checkpoint format for resuming.")

        self.best_r1 = checkpoint['best_r1']
        start_epoch = checkpoint['epoch']
        
        logger.info(f"Resuming successfully from epoch {start_epoch} with Best R@1: {self.best_r1:.2f}")
        
        return start_epoch

    def save_checkpoint(self, epoch, is_best=False):
        """
        Save checkpoint. Only two files are ever written: last_model.pth (for resume)
        and best_model.pth (only when is_best=True). No epoch-based checkpoints.
        Disk errors are caught and logged; training continues.
        """
        last_path = os.path.join(self.checkpoint_dir, "last_model.pth")
        best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        # NOTE: Python/numpy/torch random state is intentionally not saved.
        # Resuming from checkpoint will have different augmentation than a fresh run
        # at the same epoch. This is a known, accepted limitation.
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_r1': self.best_r1,
            'config': self.config
        }
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Always update last_model.pth for resume
        try:
            torch.save(checkpoint, last_path)
            logger.info(f"Checkpoint saved: last_model.pth (epoch {epoch})")
        except (OSError, RuntimeError) as e:
            # Do not crash training on disk errors; just log and continue.
            logger.error(f"DISK ERROR: Could not save last_model.pth at Epoch {epoch}. {e}")
            return

        # Optionally update best_model.pth for best performance
        if is_best:
            try:
                torch.save(checkpoint, best_path)
                logger.info(f"Saved best model to {best_path} (R@1: {self.best_r1:.2f})")
            except (OSError, RuntimeError) as e:
                # Same here: log and keep training.
                logger.error(f"DISK ERROR: Could not save best_model.pth at Epoch {epoch}. {e}")

    def train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        batch_time = AverageMeter()
        
        num_batches = len(self.train_loader)
        batch_size = self.config['data']['batch_size']
        intra_img_weight = self.config['loss'].get('intra_img_weight', 0.0)
        intra_txt_weight = self.config['loss'].get('intra_txt_weight', 0.0)

        # Initialize grad_norm for logging
        grad_norm = 0.0
        end_time = time.time()
        
        for step, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            neg_input_ids = batch.get('negative_input_ids')
            neg_attention_mask = batch.get('negative_attention_mask')
            if neg_input_ids is not None:
                neg_input_ids = neg_input_ids.to(self.device)
                neg_attention_mask = neg_attention_mask.to(self.device)
            
            # ============================================================
            # Forward Pass: Use Gradient Caching if enabled, else standard
            # ============================================================
            if self.use_grad_cache:
                # GRADIENT CACHING MODE
                # GradCache handles the forward/backward internally
                # Note: Currently GradCache does NOT support neg_txt_embeds or intra-modal
                # TODO: Add support for these features if needed
                if neg_input_ids is not None:
                    logger.warning("Hard negatives (neg_input_ids) are not supported with GradCache yet. Ignoring.")
                
                # Zero gradients before GradCache forward (which accumulates gradients)
                self.optimizer.zero_grad(set_to_none=True)
                
                # GradCache performs forward and backward internally
                loss_dict = self.grad_cache.forward(images, input_ids, attention_mask)
                loss = loss_dict["loss_total"]
                
                # Compute grad norm and step optimizer
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = compute_grad_norm(self.model)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    grad_norm = compute_grad_norm(self.model)
                    self.optimizer.step()
            
            else:
                # STANDARD MODE (Original Implementation)
                if self.use_amp:
                    self.optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type='cuda'):
                        img_embeds = self.model.encode_image(images)
                        txt_embeds = self.model.encode_text(input_ids, attention_mask)
                        img_aug_embeds = None
                        txt_aug_embeds = None
                        if intra_img_weight > 0:
                            with torch.no_grad():
                                img_aug_embeds = self.model.encode_image(batch['image_aug'].to(self.device))
                        if intra_txt_weight > 0:
                            with torch.no_grad():
                                txt_aug_embeds = self.model.encode_text(input_ids, attention_mask)

                        loss_dict = self.criterion(
                            img_embeds, txt_embeds,
                            self.model.clip.logit_scale,
                            img_aug_embeds, txt_aug_embeds
                        )
                        loss = loss_dict["loss_total"]

                        if step == 0:
                            # Sanity: embeddings must be unit-norm after L2 normalization
                            with torch.no_grad():
                                img_norms = torch.norm(img_embeds, dim=1)
                                txt_norms = torch.norm(txt_embeds, dim=1)
                                if not torch.allclose(img_norms, torch.ones_like(img_norms), atol=1e-5):
                                    raise RuntimeError(
                                        f"Image embeddings are not unit-norm at step 0. "
                                        f"Max deviation: {(img_norms - 1).abs().max().item():.6f}"
                                    )
                                if not torch.allclose(txt_norms, torch.ones_like(txt_norms), atol=1e-5):
                                    raise RuntimeError(
                                        f"Text embeddings are not unit-norm at step 0. "
                                        f"Max deviation: {(txt_norms - 1).abs().max().item():.6f}"
                                    )
                                # Sanity: intra-modal loss is exactly 0 when weight is 0
                                if getattr(self.criterion, 'w_img', 0.0) == 0.0:
                                    assert loss_dict["loss_intra_img"].item() == 0.0, \
                                        "loss_intra_img is non-zero but intra_img_weight=0"
                                if getattr(self.criterion, 'w_txt', 0.0) == 0.0:
                                    assert loss_dict["loss_intra_txt"].item() == 0.0, \
                                        "loss_intra_txt is non-zero but intra_txt_weight=0"

                    # Backward with gradient scaling
                    self.scaler.scale(loss).backward()
                    
                    # Unscale gradients to compute true grad norm
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = compute_grad_norm(self.model)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard precision training
                    self.optimizer.zero_grad(set_to_none=True)
                    img_embeds = self.model.encode_image(images)
                    txt_embeds = self.model.encode_text(input_ids, attention_mask)
                    img_aug_embeds = None
                    txt_aug_embeds = None
                    if intra_img_weight > 0:
                        with torch.no_grad():
                            img_aug_embeds = self.model.encode_image(batch['image_aug'].to(self.device))
                    if intra_txt_weight > 0:
                        with torch.no_grad():
                            txt_aug_embeds = self.model.encode_text(input_ids, attention_mask)

                    # loss is now a dict
                    loss_dict = self.criterion(
                        img_embeds, txt_embeds,
                        self.model.clip.logit_scale,
                        img_aug_embeds, txt_aug_embeds
                    )
                    loss = loss_dict["loss_total"]

                    if step == 0:
                        # Sanity: embeddings must be unit-norm after L2 normalization
                        with torch.no_grad():
                            img_norms = torch.norm(img_embeds, dim=1)
                            txt_norms = torch.norm(txt_embeds, dim=1)
                            if not torch.allclose(img_norms, torch.ones_like(img_norms), atol=1e-5):
                                raise RuntimeError(
                                    f"Image embeddings are not unit-norm at step 0. "
                                    f"Max deviation: {(img_norms - 1).abs().max().item():.6f}"
                                )
                            if not torch.allclose(txt_norms, torch.ones_like(txt_norms), atol=1e-5):
                                raise RuntimeError(
                                    f"Text embeddings are not unit-norm at step 0. "
                                    f"Max deviation: {(txt_norms - 1).abs().max().item():.6f}"
                                )
                            # Sanity: intra-modal loss is exactly 0 when weight is 0
                            if getattr(self.criterion, 'w_img', 0.0) == 0.0:
                                assert loss_dict["loss_intra_img"].item() == 0.0, \
                                    "loss_intra_img is non-zero but intra_img_weight=0"
                            if getattr(self.criterion, 'w_txt', 0.0) == 0.0:
                                assert loss_dict["loss_intra_txt"].item() == 0.0, \
                                    "loss_intra_txt is non-zero but intra_txt_weight=0"

                    # Backward
                    loss.backward()
                    
                    # Compute grad norm before optimizer step
                    grad_norm = compute_grad_norm(self.model)
                    
                    self.optimizer.step()
            
            self.scheduler.step()
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Loss is {loss.item()} at epoch {epoch} step {step}. "
                    "Training cannot continue."
                )

            # Measure batch time
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if step % self.log_freq == 0:
                fractional_epoch = epoch + (step / num_batches)
                samples_per_sec = batch_size / batch_time.val if batch_time.val > 0 else 0
                
                lr_dict = {}
                for i, param_group in enumerate(self.optimizer.param_groups):
                    group_name = param_group.get('name', f'group_{i}')
                    lr_dict[f"train/lr_{group_name}"] = param_group['lr']
                
                primary_lr = self.optimizer.param_groups[0]['lr']
                
                logger.info(
                    f"Epoch {epoch+1} [{step}/{num_batches}] "
                    f"Loss: {losses.avg:.4f} | LR: {primary_lr:.6f} | "
                    f"GradNorm: {grad_norm:.2f} | Speed: {samples_per_sec:.1f} samples/s"
                )

                with torch.no_grad():
                    scale_val = self.model.clip.logit_scale.exp().item()
                    if scale_val >= 99.0:
                        logger.warning(
                            f"logit_scale.exp()={scale_val:.2f} has hit the clamp ceiling (100). "
                            "This indicates temperature instability."
                        )

                if self.use_wandb:
                    try:
                        log_dict = {
                            "epoch": epoch,
                            "train/loss_total": loss_dict["loss_total"].item(),
                            "train/loss_inter": loss_dict["loss_inter"].item(),
                            "train/loss_intra_img": loss_dict["loss_intra_img"].item(),
                            "train/loss_intra_txt": loss_dict["loss_intra_txt"].item(),
                            "train/epoch": fractional_epoch,
                            "train/step": step,
                            "train/grad_norm": grad_norm,
                            "train/samples_per_sec": samples_per_sec,
                            "train/batch_time": batch_time.val,
                        }
                        log_dict.update(lr_dict)
                        wandb.log(log_dict)
                    except Exception as e:
                        logger.warning(f"Failed to log to W&B: {e}", exc_info=True)
        
        return losses.avg

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        img_embeds_list = []
        txt_embeds_list = []
        image_ids_list = []
        sentids_list = []

        epoch_log = epoch if isinstance(epoch, str) else epoch
        logger.info(f"Starting Evaluation ({epoch_log})...")

        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            img_emb, txt_emb = self.model(images, input_ids, attention_mask)
            img_embeds_list.append(img_emb.cpu())
            txt_embeds_list.append(txt_emb.cpu())
            image_ids_list.append(batch['image_id'].cpu())
            sentids_list.append(batch['sentid'].cpu())

        img_embeds = torch.cat(img_embeds_list, dim=0)   # [N_captions, D]
        txt_embeds = torch.cat(txt_embeds_list, dim=0)   # [N_captions, D]
        image_ids = torch.cat(image_ids_list, dim=0)     # [N_captions]
        sentids = torch.cat(sentids_list, dim=0)         # [N_captions]

        # Deduplicate images, preserving insertion order
        seen_image_ids = set()
        first_occurrence_indices = []
        unique_image_ids_list = []
        for idx in range(len(image_ids)):
            img_id = image_ids[idx].item()
            if img_id not in seen_image_ids:
                seen_image_ids.add(img_id)
                first_occurrence_indices.append(idx)
                unique_image_ids_list.append(img_id)

        unique_image_ids = torch.tensor(unique_image_ids_list, dtype=image_ids.dtype)
        img_embeds_unique = img_embeds[first_occurrence_indices]  # [N_images, D]

        images_path = self.config.get('data', {}).get('images_path', '')
        is_coco = 'coco' in images_path.lower()
        is_flickr = 'flickr' in images_path.lower()

        if not is_coco and not is_flickr:
            raise ValueError(
                f"Cannot determine dataset from images_path='{images_path}'. "
                "Path must contain 'coco' or 'flickr'."
            )

        # ------------------------------------------------------------------
        # Standard retrieval metrics run on all datasets/splits.
        # COCO test additionally logs eccv_caption metrics.
        # ------------------------------------------------------------------
        current_split = self.val_loader.dataset.split
        dataset_label = "Flickr" if is_flickr else ("COCO Test" if current_split == 'test' else "COCO (Val)")

        r_t2i, r_i2t = compute_recall_at_k(img_embeds_unique, txt_embeds, image_ids, unique_image_ids)

        # Build ground-truth dicts for mAP@R / R-Precision
        # Keys are sentids (stable Karpathy IDs); ranked list values are also sentids / image_ids.
        # gt_i2t: image_id -> set of sentids belonging to that image (R=5)
        # gt_t2i: sentid -> set of image_ids for that caption (R=1)
        caption_ids_rank = [sid.item() for sid in sentids]
        image_ids_rank = [uid.item() for uid in unique_image_ids]

        gt_i2t = {}
        for uid in unique_image_ids:
            uid_val = uid.item()
            gt_i2t[uid_val] = set(
                caption_ids_rank[cap_idx]
                for cap_idx in range(len(image_ids))
                if image_ids[cap_idx].item() == uid_val
            )

        gt_t2i = {}
        for cap_idx, sid in enumerate(caption_ids_rank):
            img_id = image_ids[cap_idx].item()
            gt_t2i[sid] = {img_id}

        sim_matrix = torch.matmul(txt_embeds, img_embeds_unique.t()).numpy()  # [N_captions, N_images]
        i2t_ranked, t2i_ranked = build_ranked_dicts(sim_matrix, image_ids_rank, caption_ids_rank)
        mapr_scores = compute_mapr_rprecision(i2t_ranked, t2i_ranked, gt_i2t, gt_t2i)

        log_data = {
            "val/t2i_r1":         r_t2i[1],
            "val/t2i_r5":         r_t2i[5],
            "val/t2i_r10":        r_t2i[10],
            "val/i2t_r1":         r_i2t[1],
            "val/i2t_r5":         r_i2t[5],
            "val/i2t_r10":        r_i2t[10],
            "val/t2i_mapr":       mapr_scores['mapr_t2i'],
            "val/i2t_mapr":       mapr_scores['mapr_i2t'],
            "val/t2i_rprecision": mapr_scores['rprecision_t2i'],
            "val/i2t_rprecision": mapr_scores['rprecision_i2t'],
        }

        logger.info(
            f"Epoch {epoch_log} {dataset_label} Results:\n"
            f"  T2I: R@1: {r_t2i[1]:.2f} | R@5: {r_t2i[5]:.2f} | R@10: {r_t2i[10]:.2f} | "
            f"mAP@R: {mapr_scores['mapr_t2i']:.2f} | R-Prec: {mapr_scores['rprecision_t2i']:.2f}\n"
            f"  I2T: R@1: {r_i2t[1]:.2f} | R@5: {r_i2t[5]:.2f} | R@10: {r_i2t[10]:.2f} | "
            f"mAP@R: {mapr_scores['mapr_i2t']:.2f} | R-Prec: {mapr_scores['rprecision_i2t']:.2f}"
        )

        eccv_scores = {}
        if is_coco and current_split == 'test':
            caption_ids_eccv = [sid.item() for sid in sentids]
            image_ids_eccv = [uid.item() for uid in unique_image_ids]
            eccv_scores = compute_eccv_metrics(sim_matrix, image_ids_eccv, caption_ids_eccv)

        if is_coco and eccv_scores:
            # eccv_caption output structure:
            #   coco_5k_recalls / coco_1k_recalls: {direction: {K: value}}   (0-1 scale)
            #   eccv_r1 / eccv_map_at_r / eccv_rprecision: {direction: value} (0-1 scale)
            #   cxc_recalls: {direction: {K: value}}                          (0-1 scale)
            coco5k = eccv_scores.get('coco_5k_recalls', {})
            coco1k = eccv_scores.get('coco_1k_recalls', {})
            eccv_r1 = eccv_scores.get('eccv_r1', {})
            eccv_mapr = eccv_scores.get('eccv_map_at_r', {})
            eccv_rprec = eccv_scores.get('eccv_rprecision', {})
            cxc = eccv_scores.get('cxc_recalls', {})

            def _recall(d, direction, k):
                return d.get(direction, {}).get(k, 0.0) * 100.0

            def _scalar(d, direction):
                return d.get(direction, 0.0) * 100.0

            log_data.update({
                # COCO-5K recalls
                "val/coco_5k_r1_i2t":  _recall(coco5k, 'i2t', 1),
                "val/coco_5k_r1_t2i":  _recall(coco5k, 't2i', 1),
                "val/coco_5k_r5_i2t":  _recall(coco5k, 'i2t', 5),
                "val/coco_5k_r5_t2i":  _recall(coco5k, 't2i', 5),
                "val/coco_5k_r10_i2t": _recall(coco5k, 'i2t', 10),
                "val/coco_5k_r10_t2i": _recall(coco5k, 't2i', 10),
                # COCO-1K recalls
                "val/coco_1k_r1_i2t":  _recall(coco1k, 'i2t', 1),
                "val/coco_1k_r1_t2i":  _recall(coco1k, 't2i', 1),
                "val/coco_1k_r5_i2t":  _recall(coco1k, 'i2t', 5),
                "val/coco_1k_r5_t2i":  _recall(coco1k, 't2i', 5),
                "val/coco_1k_r10_i2t": _recall(coco1k, 'i2t', 10),
                "val/coco_1k_r10_t2i": _recall(coco1k, 't2i', 10),
                # ECCV metrics
                "val/eccv_r1_i2t":          _scalar(eccv_r1,   'i2t'),
                "val/eccv_r1_t2i":          _scalar(eccv_r1,   't2i'),
                "val/eccv_map_at_r_i2t":    _scalar(eccv_mapr, 'i2t'),
                "val/eccv_map_at_r_t2i":    _scalar(eccv_mapr, 't2i'),
                "val/eccv_rprecision_i2t":  _scalar(eccv_rprec, 'i2t'),
                "val/eccv_rprecision_t2i":  _scalar(eccv_rprec, 't2i'),
                # CxC recalls (R@1 only for brevity; full set available via eccv_scores)
                "val/cxc_r1_i2t": _recall(cxc, 'i2t', 1),
                "val/cxc_r1_t2i": _recall(cxc, 't2i', 1),
            })

            logger.info(
                f"Epoch {epoch_log} COCO ECCV Test Results:\n"
                f"  COCO-5K  R@1: i2t={log_data['val/coco_5k_r1_i2t']:.2f} | t2i={log_data['val/coco_5k_r1_t2i']:.2f} | "
                f"R@5: i2t={log_data['val/coco_5k_r5_i2t']:.2f} | t2i={log_data['val/coco_5k_r5_t2i']:.2f} | "
                f"R@10: i2t={log_data['val/coco_5k_r10_i2t']:.2f} | t2i={log_data['val/coco_5k_r10_t2i']:.2f}\n"
                f"  COCO-1K  R@1: i2t={log_data['val/coco_1k_r1_i2t']:.2f} | t2i={log_data['val/coco_1k_r1_t2i']:.2f}\n"
                f"  ECCV R@1:     i2t={log_data['val/eccv_r1_i2t']:.2f} | t2i={log_data['val/eccv_r1_t2i']:.2f}\n"
                f"  ECCV mAP@R:   i2t={log_data['val/eccv_map_at_r_i2t']:.2f} | t2i={log_data['val/eccv_map_at_r_t2i']:.2f}\n"
                f"  ECCV R-Prec:  i2t={log_data['val/eccv_rprecision_i2t']:.2f} | t2i={log_data['val/eccv_rprecision_t2i']:.2f}"
            )

            current_metrics = {k.split('val/')[1]: v for k, v in log_data.items()}
            primary_score = log_data["val/coco_5k_r1_i2t"]
        else:
            current_metrics = {k.split('val/')[1]: v for k, v in log_data.items()}
            primary_score = r_t2i[1]

        if self.use_wandb:
            if isinstance(epoch, int):
                log_data["epoch"] = epoch
            else:
                # TEST_FINAL: write everything to run summary
                for k, v in log_data.items():
                    if k != "epoch":
                        wandb.run.summary[f"test_{k.split('/')[1]}"] = v

            try:
                wandb.log(log_data)

                for metric_name, current_value in current_metrics.items():
                    if current_value > self.best_metrics.get(metric_name, 0.0):
                        self.best_metrics[metric_name] = current_value
                        wandb.run.summary[f"best_val_{metric_name}"] = current_value

            except Exception as e:
                logger.warning(f"Failed to log to W&B: {e}", exc_info=True)

            if epoch == "TEST_FINAL":
                try:
                    self._log_qualitative_table(epoch, txt_embeds, img_embeds_unique, first_occurrence_indices)
                except Exception as e:
                    logger.warning(f"Qualitative table logging failed: {e}", exc_info=True)

        return primary_score

    def _log_qualitative_table(self, epoch, txt_embeds, img_embeds_unique, first_occurrence_indices):
        """
        Logs a WandB Table visualizing Text-to-Image retrieval on the test set.
        Only called during TEST_FINAL so results reflect the best checkpoint, not val epochs.
        Columns: image_id | sentid | Caption | Ground Truth | Rank 1 | Rank 2 | Rank 3

        Samples are selected deterministically: the 50 samples with the lowest sentid,
        so the same images are shown across all runs and seeds.
        Images are resized to 224px before upload to reduce W&B payload (~55% smaller).
        """
        from PIL import Image as PILImage
        dataset = self.val_loader.dataset

        # Deterministic sample selection: 50 lowest sentids
        indexed = sorted(enumerate(dataset.samples), key=lambda x: x[1]['sentid'])
        indices = [i for i, _ in indexed[:50]]

        columns = ["image_id", "sentid", "Caption", "Ground Truth", "Rank 1", "Rank 2", "Rank 3"]
        table = wandb.Table(columns=columns)

        for idx in indices:
            sample = dataset.samples[idx]
            caption = sample["caption"]
            image_id = sample["image_id"]
            sentid = sample["sentid"]
            filename = sample["filename"]
            filepath = sample.get("filepath", "")
            if filepath:
                img_path = os.path.join(dataset.images_root_path, filepath, filename)
            else:
                img_path = os.path.join(dataset.images_root_path, filename)

            try:
                pil_gt = PILImage.open(img_path).convert("RGB")
                pil_gt.thumbnail((224, 224), PILImage.LANCZOS)
                gt_image = wandb.Image(pil_gt)
            except Exception as e:
                logger.warning(f"Failed to load ground-truth image: {img_path} — {e}")
                gt_image = wandb.Image(torch.zeros(3, 224, 224), caption="Img Not Found")

            query_emb = txt_embeds[idx].to(self.device)
            sims = torch.matmul(img_embeds_unique.to(self.device), query_emb)
            topk_scores, topk_indices = sims.topk(3)

            retrieved_images = []
            for rank, ret_idx in enumerate(topk_indices):
                dataset_idx = first_occurrence_indices[ret_idx.item()]
                ret_sample = dataset.samples[dataset_idx]
                ret_filename = ret_sample["filename"]
                ret_filepath = ret_sample.get("filepath", "")
                if ret_filepath:
                    ret_path = os.path.join(dataset.images_root_path, ret_filepath, ret_filename)
                else:
                    ret_path = os.path.join(dataset.images_root_path, ret_filename)
                score_str = f"#{rank+1} img_id={ret_sample['image_id']} score={topk_scores[rank].item():.2f}"
                try:
                    pil_ret = PILImage.open(ret_path).convert("RGB")
                    pil_ret.thumbnail((224, 224), PILImage.LANCZOS)
                    retrieved_images.append(wandb.Image(pil_ret, caption=score_str))
                except Exception as e:
                    logger.warning(f"Failed to load retrieved image: {ret_path} — {e}")
                    retrieved_images.append(wandb.Image(torch.zeros(3, 224, 224), caption="Err"))

            table.add_data(image_id, sentid, caption, gt_image, *retrieved_images)

        wandb.log({"test/qualitative_results": table}, commit=False)

    def fit(self, start_epoch=0):
        eval_frequency = self.config['logging']['eval_freq']

        # Run evaluation before training only when explicitly requested via eval_epoch_zero.
        if start_epoch == 0 and self.config['logging'].get('eval_epoch_zero', False):
            logger.info("Running initial evaluation at Epoch 0 (eval_epoch_zero=true)...")
            score = self.evaluate(epoch=0)
            if score > self.best_r1:
                self.best_r1 = score
            logger.info(f"Epoch 0 evaluation done. Starting training loop.")

        # --- MAIN TRAINING LOOP ---
        for epoch in range(start_epoch, self.config['training']['epochs']):
            # Train one epoch
            train_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")

            # Evaluation & Best Model Check
            is_new_best = False
            is_eval_time = ((epoch + 1) % eval_frequency == 0) or (
                (epoch + 1) == self.config['training']['epochs']
            )
            if is_eval_time:
                score = self.evaluate(epoch)
                is_new_best = score > self.best_r1
                if is_new_best:
                    self.best_r1 = score
                    logger.info(f"New Best R@1: {score:.2f} found at Epoch {epoch+1}!")

            # Save Checkpoint
            # Always keep last_model.pth up to date for resume; also update best_model.pth when is_new_best=True
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.save_checkpoint(epoch + 1, is_best=is_new_best)
