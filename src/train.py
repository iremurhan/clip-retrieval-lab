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
import torch.nn.functional as F
import logging
import wandb
import os
import time
import random
from .metrics import AverageMeter, compute_recall_at_k, compute_map_at_k
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
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize WandB run reference and define summary metrics
        self.wandb_run = None
        if self.use_wandb and wandb.run is not None:
            self.wandb_run = wandb.run
            # Define WandB summary metrics with max tracking for validation results
            # This ensures the dashboard always shows the best scores, even if current epoch is worse
            wandb.define_metric("val/t2i_r1", summary="max")
            wandb.define_metric("val/t2i_r5", summary="max")
            wandb.define_metric("val/t2i_r10", summary="max")
            wandb.define_metric("val/i2t_r1", summary="max")
            wandb.define_metric("val/i2t_r5", summary="max")
            wandb.define_metric("val/i2t_r10", summary="max")
            wandb.define_metric("val/t2i_map5", summary="max")
            wandb.define_metric("val/t2i_map10", summary="max")
            wandb.define_metric("val/i2t_map5", summary="max")
            wandb.define_metric("val/i2t_map10", summary="max")


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
        use_clip_loss = self.config['loss'].get('use_clip_loss', False)
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
                # Note: Currently GradCache does NOT support use_clip_loss, neg_txt_embeds, or intra-modal
                # TODO: Add support for these features if needed
                if use_clip_loss:
                    raise NotImplementedError("GradCache does not support use_clip_loss yet. Set loss.use_clip_loss=false.")
                if neg_input_ids is not None:
                    logger.warning("Hard negatives (neg_input_ids) are not supported with GradCache yet. Ignoring.")
                if intra_img_weight > 0 or intra_txt_weight > 0:
                    logger.warning("Intra-modal losses not supported with GradCache yet. Ignoring.")
                
                # Zero gradients before GradCache forward (which accumulates gradients)
                self.optimizer.zero_grad()
                
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
                    with torch.amp.autocast(device_type='cuda'):
                        if use_clip_loss:
                            loss, _, _ = self.model.forward_with_clip_loss(images, input_ids, attention_mask)
                        else:
                            img_embeds = self.model.encode_image(images)
                            txt_embeds = self.model.encode_text(input_ids, attention_mask)
                            neg_txt_embeds = None
                            if neg_input_ids is not None:
                                neg_txt_embeds = self.model.encode_text(neg_input_ids, neg_attention_mask)
                            img_aug_embeds = None
                            txt_aug_embeds = None
                            if intra_img_weight > 0:
                                img_aug_embeds = self.model.encode_image(batch['image_aug'].to(self.device))
                            if intra_txt_weight > 0:
                                txt_aug_embeds = self.model.encode_text(input_ids, attention_mask)
                            
                            loss_dict = self.criterion(
                                img_embeds, txt_embeds,
                                img_aug_embeds, txt_aug_embeds
                            )
                            loss = loss_dict["loss_total"]
                    
                    # Backward with gradient scaling
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    
                    # Unscale gradients to compute true grad norm
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = compute_grad_norm(self.model)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard precision training
                    if use_clip_loss:
                        loss, _, _ = self.model.forward_with_clip_loss(images, input_ids, attention_mask)
                    else:
                        img_embeds = self.model.encode_image(images)
                        txt_embeds = self.model.encode_text(input_ids, attention_mask)
                        neg_txt_embeds = None
                        if neg_input_ids is not None:
                            neg_txt_embeds = self.model.encode_text(neg_input_ids, neg_attention_mask)
                        img_aug_embeds = None
                        txt_aug_embeds = None
                        if intra_img_weight > 0:
                            img_aug_embeds = self.model.encode_image(batch['image_aug'].to(self.device))
                        if intra_txt_weight > 0:
                            txt_aug_embeds = self.model.encode_text(input_ids, attention_mask)
                        
                        # loss is now a dict
                        loss_dict = self.criterion(
                            img_embeds, txt_embeds,
                            img_aug_embeds, txt_aug_embeds
                        )
                        loss = loss_dict["loss_total"]
                    
                    # Backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # Compute grad norm before optimizer step
                    grad_norm = compute_grad_norm(self.model)
                    
                    self.optimizer.step()
            
            self.scheduler.step()
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
            
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
                        logger.warning(f"Failed to log to W&B: {e}")
        
        return losses.avg

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        img_embeds_list = []
        txt_embeds_list = []
        image_ids_list = []
        
        # Handle "TEST" epoch logging string
        epoch_log = epoch if isinstance(epoch, str) else epoch
        logger.info(f"Starting Evaluation ({epoch_log})...")
        
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            img_emb, txt_emb = self.model(images, input_ids, attention_mask)
            img_embeds_list.append(img_emb.cpu())
            txt_embeds_list.append(txt_emb.cpu())
            
            # Collect image_ids for ground truth matching
            image_ids_list.append(batch['image_id'].cpu())
            
        img_embeds = torch.cat(img_embeds_list, dim=0)
        txt_embeds = torch.cat(txt_embeds_list, dim=0)
        image_ids = torch.cat(image_ids_list, dim=0)
        
        # Get unique images in insertion order (preserving order of first occurrence)
        # This ensures alignment between unique_image_ids and img_embeds_unique
        seen_image_ids = set()
        first_occurrence_indices = []
        unique_image_ids_list = []
        
        for idx in range(len(image_ids)):
            img_id = image_ids[idx].item()
            if img_id not in seen_image_ids:
                seen_image_ids.add(img_id)
                first_occurrence_indices.append(idx)
                unique_image_ids_list.append(img_id)
        
        # Convert to tensors - order matches first_occurrence_indices (insertion order)
        unique_image_ids = torch.tensor(unique_image_ids_list, dtype=image_ids.dtype)
        img_embeds_unique = img_embeds[first_occurrence_indices]
        
        # Compute Recall@K metrics
        r_t2i, r_i2t = compute_recall_at_k(img_embeds_unique, txt_embeds, image_ids, unique_image_ids)
        
        # Compute MAP@K metrics
        map_t2i, map_i2t = compute_map_at_k(img_embeds_unique, txt_embeds, image_ids, unique_image_ids, k_values=[5, 10])
        
        # Log results to console
        logger.info(
            f"Epoch {epoch_log} Results:\n"
            f"  T2I: R@1: {r_t2i[1]:.2f} | R@5: {r_t2i[5]:.2f} | R@10: {r_t2i[10]:.2f} | MAP@5: {map_t2i[5]:.2f} | MAP@10: {map_t2i[10]:.2f}\n"
            f"  I2T: R@1: {r_i2t[1]:.2f} | R@5: {r_i2t[5]:.2f} | R@10: {r_i2t[10]:.2f} | MAP@5: {map_i2t[5]:.2f} | MAP@10: {map_i2t[10]:.2f}"
        )
        
        # Current metrics dictionary
        current_metrics = {
            't2i_r1': r_t2i[1], 't2i_r5': r_t2i[5], 't2i_r10': r_t2i[10],
            'i2t_r1': r_i2t[1], 'i2t_r5': r_i2t[5], 'i2t_r10': r_i2t[10],
            't2i_map5': map_t2i[5], 't2i_map10': map_t2i[10],
            'i2t_map5': map_i2t[5], 'i2t_map10': map_i2t[10],
        }
        
        if self.use_wandb:
            # Build log data dictionary
            log_data = {
                "val/t2i_r1": r_t2i[1], "val/t2i_r5": r_t2i[5], "val/t2i_r10": r_t2i[10],
                "val/i2t_r1": r_i2t[1], "val/i2t_r5": r_i2t[5], "val/i2t_r10": r_i2t[10],
                "val/t2i_map5": map_t2i[5], "val/t2i_map10": map_t2i[10],
                "val/i2t_map5": map_i2t[5], "val/i2t_map10": map_i2t[10],
            }
            
            # If epoch is a number, add epoch info; otherwise handle test logging
            if isinstance(epoch, int):
                log_data["epoch"] = epoch
            else:
                # For test evaluation (e.g., "TEST_FINAL"), store as summary metrics
                for k, v in log_data.items():
                    wandb.run.summary[f"test_{k.split('/')[1]}"] = v
            
            try:
                wandb.log(log_data)
                
                # Update best metrics in WandB summary if new best is found
                # This ensures the run summary always shows MAX values
                for metric_name, current_value in current_metrics.items():
                    if current_value > self.best_metrics[metric_name]:
                        self.best_metrics[metric_name] = current_value
                        # Update WandB summary with best value
                        wandb.run.summary[f"best_val_{metric_name}"] = current_value
                        
            except Exception as e:
                logger.warning(f"Failed to log to W&B: {e}")

            if self.use_wandb:
                try:
                    self._log_qualitative_table(epoch, txt_embeds, img_embeds_unique, first_occurrence_indices)
                except Exception as e:
                    logger.warning(f"Qualitative table logging failed: {e}")
            
        return r_t2i[1]

    def _log_qualitative_table(self, epoch, txt_embeds, img_embeds_unique, first_occurrence_indices):
        """
        Logs a WandB Table visualizing Text-to-Image retrieval.
        Columns: Caption | Ground Truth | Rank 1 | Rank 2 | Rank 3
        """
        num_samples = len(txt_embeds)
        sample_count = min(5, num_samples)
        indices = random.sample(range(num_samples), sample_count)

        columns = ["Caption", "Ground Truth", "Rank 1", "Rank 2", "Rank 3"]
        table = wandb.Table(columns=columns)
        dataset = self.val_loader.dataset

        for idx in indices:
            sample = dataset.samples[idx]
            caption = sample["caption"]
            filename = sample["filename"]
            filepath = sample.get("filepath", "")
            if filepath:
                img_path = os.path.join(dataset.images_root_path, filepath, filename)
            else:
                img_path = os.path.join(dataset.images_root_path, filename)

            try:
                gt_image = wandb.Image(img_path)
            except Exception:
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
                score_str = f"Score: {topk_scores[rank].item():.2f}"
                try:
                    retrieved_images.append(wandb.Image(ret_path, caption=score_str))
                except Exception:
                    retrieved_images.append(wandb.Image(torch.zeros(3, 224, 224), caption="Err"))

            table.add_data(caption, gt_image, *retrieved_images)

        wandb.log({"val/qualitative_results": table}, commit=False)

    def fit(self, start_epoch=0):
        eval_frequency = self.config['logging']['eval_freq']

        # If start_epoch is 0 (i.e., not resuming), run initial evaluation.
        if start_epoch == 0:
            logger.info("Running initial evaluation at Epoch 0...")
            score = self.evaluate(epoch=0)
            if score > self.best_r1:
                self.best_r1 = score
            logger.info(f"Initial state saved. Starting training loop from Epoch {start_epoch}.")

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
