import torch
import torch.nn.functional as F
import logging
import wandb
import os
import time
from .utils import AverageMeter, compute_recall_at_k
from .loss import DistillationLoss

logger = logging.getLogger(__name__)


def compute_grad_norm(model):
    """
    Compute the total L2 gradient norm across all parameters.
    Useful for monitoring training stability and detecting gradient explosions.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


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
        self.best_r1 = 0.0
        
        # Initialize Mixed Precision Training (AMP)
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed Precision Training (AMP) enabled.")
        else:
            self.scaler = None
            logger.info("Mixed Precision Training (AMP) disabled (CPU mode).")
        
        # ---------------------------------------------------------
        # Knowledge Distillation Setup with Linear Decay
        # ---------------------------------------------------------
        self.use_distillation = False
        self.distill_loss_fn = None
        self.distill_alpha_start = 0.0
        self.total_epochs = config['training']['epochs']
        
        distillation_config = config.get('distillation', {})
        if distillation_config.get('enabled'):
            self.use_distillation = True
            self.distill_alpha_start = distillation_config.get('alpha')
            distill_temp = distillation_config.get('temperature')
            
            self.distill_loss_fn = DistillationLoss(temperature=distill_temp)
            
            logger.info(
                f"Knowledge Distillation enabled: "
                f"alpha_start={self.distill_alpha_start} (linear decay to 0), "
                f"temperature={distill_temp}"
            )
        
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

    def _compute_current_alpha(self, epoch: int) -> float:
        """
        Compute current distillation alpha using linear decay.
        
        Formula: alpha = alpha_start * (1 - epoch / (total_epochs - 1))
        
        - At epoch 0: alpha = alpha_start
        - At last epoch (total_epochs - 1): alpha = 0
        
        Args:
            epoch: Current epoch index (0-based)
        
        Returns:
            Current alpha value
        """
        if self.total_epochs <= 1:
            return self.distill_alpha_start
        
        decay_factor = 1.0 - (epoch / (self.total_epochs - 1))
        current_alpha = self.distill_alpha_start * max(0.0, decay_factor)
        return current_alpha

    def load_checkpoint(self, checkpoint_path):
        """
        Loads full training state from a checkpoint file.
        Returns start_epoch.
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 1. Load Weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state if available (for AMP)
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info("GradScaler state loaded from checkpoint.")
        
        # 2. Load State Info
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
        Save checkpoint with robust error handling for disk quota issues.
        
        Always saves last_model.pth (for resuming training).
        If is_best is True, also saves best_model.pth (for best performance).
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        # Define checkpoint paths
        last_path = os.path.join(self.checkpoint_dir, "last_model.pth")
        best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        
        # Build checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_r1': self.best_r1,
            'config': self.config
        }
        
        # Save scaler state if using AMP
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Always save last_model.pth (overwrite previous one)
        try:
            torch.save(checkpoint, last_path)
            logger.info(f"Checkpoint saved: last_model.pth (epoch {epoch})")
        except OSError as e:
            # Handle all disk errors (quota exceeded, no space, etc.)
            logger.error(f"DISK ERROR: Could not save checkpoint at Epoch {epoch}. Training continues...")
            print(f"Error detail: {e}")
            # Don't raise - let training continue
            return
        
        # If this is the best model, also save as best_model.pth
        if is_best:
            try:
                torch.save(checkpoint, best_path)
                logger.info(f"Saved best model to {best_path} (R@1: {self.best_r1:.2f})")
            except OSError as e:
                # Handle all disk errors (quota exceeded, no space, etc.)
                logger.error(f"DISK ERROR: Could not save checkpoint at Epoch {epoch}. Training continues...")
                print(f"Error detail: {e}")
                # Don't raise - let training continue

    def train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        losses_clip = AverageMeter()
        losses_distill = AverageMeter()
        batch_time = AverageMeter()
        
        num_batches = len(self.train_loader)
        batch_size = self.config['data']['batch_size']
        
        # Check if we should use CLIP's native loss (with learnable temperature)
        use_clip_loss = self.config['loss'].get('use_clip_loss', False)
        
        # Compute current alpha with linear decay
        current_alpha = self._compute_current_alpha(epoch) if self.use_distillation else 0.0
        
        if self.use_distillation:
            logger.info(f"Epoch {epoch+1}: Distillation alpha = {current_alpha:.4f}")
        
        # Initialize grad_norm for logging
        grad_norm = 0.0
        end_time = time.time()
        
        for step, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # For distillation: get batch indices and teacher targets
            batch_indices = batch['index'].to(self.device) if self.use_distillation else None
            soft_target_indices = batch.get('soft_target_indices')
            soft_target_scores = batch.get('soft_target_scores')
            
            if soft_target_indices is not None:
                soft_target_indices = soft_target_indices.to(self.device)
            if soft_target_scores is not None:
                soft_target_scores = soft_target_scores.to(self.device)
            
            # ============================================================
            # Forward Pass with Mixed Precision (AMP)
            # ============================================================
            if self.use_amp:
                # Use autocast for forward pass
                with torch.amp.autocast(device_type='cuda'):
                    # Option A: Use CLIP's Native Loss (Learnable Temperature)
                    if use_clip_loss:
                        loss_clip, _, _ = self.model.forward_with_clip_loss(images, input_ids, attention_mask)
                        # Get text embeddings for distillation
                        if self.use_distillation and soft_target_indices is not None:
                            txt_embeds = self.model.clip.get_text_features(
                                input_ids=input_ids,
                                attention_mask=attention_mask
                            )
                            txt_embeds = F.normalize(txt_embeds, p=2, dim=1)
                    else:
                        images_aug = batch['image_aug'].to(self.device)
                        
                        # Forward (Original Views)
                        img_embeds, txt_embeds = self.model(images, input_ids, attention_mask)
                        
                        img_aug_embeds = None
                        txt_aug_embeds = None

                        # A. Image Intra-Modal (Img <-> Img_Aug)
                        if self.config['loss'].get('intra_img_weight', 0.0) > 0:
                            img_aug_embeds, _ = self.model(images_aug, input_ids, attention_mask)

                        # B. Text Intra-Modal (Text <-> Text_Aug)
                        # SimCSE style: Pass same text again (dropout acts as augmentation)
                        if self.config['loss'].get('intra_txt_weight', 0.0) > 0:
                            _, txt_aug_embeds = self.model(images, input_ids, attention_mask)
                        
                        loss_clip = self.criterion(img_embeds, txt_embeds, img_aug_embeds, txt_aug_embeds)
                    
                    # Knowledge Distillation Loss
                    loss_distill = torch.tensor(0.0, device=self.device)
                    
                    if self.use_distillation and soft_target_indices is not None and current_alpha > 0:
                        student_text_logits = txt_embeds @ txt_embeds.t()
                        
                        loss_distill = self.distill_loss_fn(
                            student_logits=student_text_logits,
                            batch_indices=batch_indices,
                            teacher_indices=soft_target_indices,
                            teacher_scores=soft_target_scores
                        )
                        
                        # Combine losses with dynamic alpha
                        loss = (1 - current_alpha) * loss_clip + current_alpha * loss_distill
                    else:
                        loss = loss_clip
                
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
                    loss_clip, _, _ = self.model.forward_with_clip_loss(images, input_ids, attention_mask)
                    if self.use_distillation and soft_target_indices is not None:
                        txt_embeds = self.model.clip.get_text_features(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        txt_embeds = F.normalize(txt_embeds, p=2, dim=1)
                else:
                    images_aug = batch['image_aug'].to(self.device)
                    
                    # Forward (Original Views)
                    img_embeds, txt_embeds = self.model(images, input_ids, attention_mask)
                    
                    # Conditional Forward (Augmented Views for Intra-Modal Loss)
                    img_aug_embeds = None
                    txt_aug_embeds = None

                    # A. Image Intra-Modal (Img <-> Img_Aug)
                    if self.config['loss'].get('intra_img_weight', 0.0) > 0:
                        img_aug_embeds, _ = self.model(images_aug, input_ids, attention_mask)

                    # B. Text Intra-Modal (Text <-> Text_Aug)
                    # SimCSE style: Pass same text again (dropout acts as augmentation)
                    if self.config['loss'].get('intra_txt_weight', 0.0) > 0:
                        _, txt_aug_embeds = self.model(images, input_ids, attention_mask)
                    
                    loss_clip = self.criterion(img_embeds, txt_embeds, img_aug_embeds, txt_aug_embeds)
                
                # Knowledge Distillation Loss (Non-AMP)
                loss_distill = torch.tensor(0.0, device=self.device)
                
                if self.use_distillation and soft_target_indices is not None and current_alpha > 0:
                    student_text_logits = txt_embeds @ txt_embeds.t()
                    
                    loss_distill = self.distill_loss_fn(
                        student_logits=student_text_logits,
                        batch_indices=batch_indices,
                        teacher_indices=soft_target_indices,
                        teacher_scores=soft_target_scores
                    )
                    
                    loss = (1 - current_alpha) * loss_clip + current_alpha * loss_distill
                else:
                    loss = loss_clip
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                
                # Compute grad norm before optimizer step
                grad_norm = compute_grad_norm(self.model)
                
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
            losses_clip.update(loss_clip.item(), images.size(0))
            if self.use_distillation:
                losses_distill.update(loss_distill.item(), images.size(0))
            
            # Measure batch time
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if step % self.log_freq == 0:
                # Calculate fractional epoch for smooth WandB charts
                fractional_epoch = epoch + (step / num_batches)
                
                # Calculate throughput (samples per second)
                samples_per_sec = batch_size / batch_time.val if batch_time.val > 0 else 0
                
                # Get all learning rates from different param groups
                lr_dict = {}
                for i, param_group in enumerate(self.optimizer.param_groups):
                    group_name = param_group.get('name', f'group_{i}')
                    lr_dict[f"train/lr_{group_name}"] = param_group['lr']
                
                # Primary LR for console logging
                primary_lr = self.optimizer.param_groups[0]['lr']
                
                if self.use_distillation:
                    logger.info(
                        f"Epoch {epoch+1} [{step}/{num_batches}] "
                        f"Loss: {losses.avg:.4f} (CLIP: {losses_clip.avg:.4f}, Distill: {losses_distill.avg:.4f}) | "
                        f"α: {current_alpha:.4f} | LR: {primary_lr:.6f} | "
                        f"GradNorm: {grad_norm:.2f} | Speed: {samples_per_sec:.1f} samples/s"
                    )
                else:
                    logger.info(
                        f"Epoch {epoch+1} [{step}/{num_batches}] "
                        f"Loss: {losses.avg:.4f} | LR: {primary_lr:.6f} | "
                        f"GradNorm: {grad_norm:.2f} | Speed: {samples_per_sec:.1f} samples/s"
                    )
                
                if self.use_wandb:
                    try:
                        # Build comprehensive log dict
                        log_dict = {
                            "train/loss": losses.val,
                            "train/loss_clip": losses_clip.val,
                            "train/epoch": fractional_epoch,
                            "train/step": step,
                            "train/grad_norm": grad_norm,
                            "train/samples_per_sec": samples_per_sec,
                            "train/batch_time": batch_time.val,
                        }
                        
                        # Add all learning rates
                        log_dict.update(lr_dict)
                        
                        # Add distillation metrics if enabled
                        if self.use_distillation:
                            log_dict["train/loss_distill"] = losses_distill.val
                            log_dict["train/current_alpha"] = current_alpha
                        
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
        
        r_t2i, r_i2t = compute_recall_at_k(img_embeds_unique, txt_embeds, image_ids, unique_image_ids)
        
        # Log
        logger.info(
            f"Epoch {epoch_log} Results:\n"
            f"  T2I: R@1: {r_t2i[1]:.2f} | R@5: {r_t2i[5]:.2f} | R@10: {r_t2i[10]:.2f}\n"
            f"  I2T: R@1: {r_i2t[1]:.2f} | R@5: {r_i2t[5]:.2f} | R@10: {r_i2t[10]:.2f}"
        )
        
        if self.use_wandb:
            # Log for both int (normal epoch) and str (e.g., "TEST_FINAL")
            log_data = {
                "val/t2i_r1": r_t2i[1], "val/t2i_r5": r_t2i[5], "val/t2i_r10": r_t2i[10],
                "val/i2t_r1": r_i2t[1], "val/i2t_r5": r_i2t[5], "val/i2t_r10": r_i2t[10],
            }
            
            # If epoch is a number, add epoch info; otherwise don't use custom string as step
            if isinstance(epoch, int):
                log_data["epoch"] = epoch
            else:
                # If epoch is a string like "TEST_FINAL", we can add a prefix to distinguish it
                # or log it directly. WandB typically works on a step basis.
                # It's best to add test results as separate "summary" metrics:
                for k, v in log_data.items():
                    wandb.run.summary[f"test_{k.split('/')[1]}"] = v
            
            try:
                wandb.log(log_data)
            except Exception as e:
                logger.warning(f"Failed to log to W&B: {e}")
            
        return r_t2i[1]

    def fit(self, start_epoch=0):
        eval_frequency = self.config['logging']['eval_freq'] 
        save_frequency = self.config['logging']['save_freq']

        # If start_epoch is 0 (i.e., not resuming), run initial evaluation.
        if start_epoch == 0:
            logger.info("Running initial evaluation at Epoch 0...")
            score = self.evaluate(epoch=0)
            
            # At Epoch 0, only saving Last Model makes sense, not Best.
            # However, to simplify: If score > 0, let this be the first record.
            if score > self.best_r1:
                self.best_r1 = score
                
            logger.info(f"Initial state saved. Starting training loop from Epoch {start_epoch}.")

        # --- MAIN TRAINING LOOP ---
        for epoch in range(start_epoch, self.config['training']['epochs']):
            # 1. Train one epoch
            train_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")
            
            # Determine when to evaluate and save
            is_eval_time = ((epoch + 1) % eval_frequency == 0) or ((epoch + 1) == self.config['training']['epochs'])
            is_save_time = ((epoch + 1) % save_frequency == 0) or ((epoch + 1) == self.config['training']['epochs'])

            # Ensure checkpoint directory exists before any save operation
            os.makedirs(self.checkpoint_dir, exist_ok=True)

            # 2. EVALUATION & BEST MODEL CHECK
            is_new_best = False
            if is_eval_time:
                score = self.evaluate(epoch)
                
                # Check if this is a new best model
                is_new_best = score > self.best_r1
                if is_new_best:
                    self.best_r1 = score
                    logger.info(f"New Best R@1: {score:.2f} found at Epoch {epoch+1}!")
                    
                    # Save best model immediately (independent of save_frequency)
                    # Also saves last_model.pth as part of the save_checkpoint method
                    self.save_checkpoint(epoch + 1, is_best=True)
                elif is_save_time:
                    # If not best but it's save time, just save last_model.pth
                    self.save_checkpoint(epoch + 1, is_best=False)

            # 3. PERIODIC CHECKPOINT SAVE (Last Model)
            # Only save if we haven't already saved in the evaluation block above
            if is_save_time and not is_eval_time:
                self.save_checkpoint(epoch + 1, is_best=False)
