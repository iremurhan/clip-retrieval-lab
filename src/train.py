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
from .metrics import AverageMeter, compute_recall_at_k, compute_map_at_k, compute_eccv_metrics
from .utils import compute_grad_norm
from .grad_cache import GradCache

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, clip_tokenizer=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        self.use_wandb = wandb.run is not None
        self.tokenizer = clip_tokenizer

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
        self.use_amp = self.device.type == 'cuda'
        self.amp_dtype = torch.bfloat16 if self.use_amp else None
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda', enabled=False)
            logger.info("Mixed Precision Training (AMP) enabled with torch.bfloat16.")
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
        
        # Initialize on-the-fly paraphraser for L_text_text (replaces SimCSE dual-forward)
        self.paraphraser = None
        if config['loss'].get('intra_txt_weight', 0.0) > 0:
            para_cfg = config.get('paraphraser')
            if para_cfg is None:
                raise ValueError(
                    "intra_txt_weight > 0 requires a 'paraphraser' section in config."
                )
            from .paraphraser import OnTheFlyParaphraser
            self.paraphraser = OnTheFlyParaphraser(para_cfg, clip_tokenizer, device)
            logger.info("OnTheFlyParaphraser (Mistral-7B) initialized.")

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize WandB run reference and define summary metrics
        self.wandb_run = None
        if self.use_wandb and wandb.run is not None:
            self.wandb_run = wandb.run
            for metric in [
                "val/r1_i2t", "val/r1_t2i",
                "val/r5_i2t", "val/r5_t2i",
                "val/r10_i2t", "val/r10_t2i",
                "val/mapr_i2t", "val/mapr_t2i",
                "val/rprecision_i2t", "val/rprecision_t2i",
            ]:
                wandb.define_metric(metric, summary="max")


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

    def save_checkpoint(self, epoch):
        """
        Save checkpoint to best_model.pth (single file, overwritten every save_freq epochs).
        Atomic write via .tmp + os.replace to avoid corrupt files on failure.
        Disk errors are caught and logged; training continues.
        """
        path = os.path.join(self.checkpoint_dir, "best_model.pth")
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

        tmp_path = path + ".tmp"
        try:
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, path)
            logger.info(f"Checkpoint saved: best_model.pth (epoch {epoch})")
        except (OSError, RuntimeError) as e:
            logger.error(f"DISK ERROR: Could not save best_model.pth at Epoch {epoch}. {e}")

    def train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        batch_time = AverageMeter()
        
        num_batches = len(self.train_loader)
        batch_size = self.config['training']['batch_size']
        intra_img_weight = self.config['loss'].get('intra_img_weight', 0.0)
        intra_txt_weight = self.config['loss'].get('intra_txt_weight', 0.0)

        # Initialize grad_norm for logging
        grad_norm = 0.0
        end_time = time.time()
        
        for step, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device, non_blocking=True)
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)

            neg_input_ids = batch.get('negative_input_ids')
            neg_attention_mask = batch.get('negative_attention_mask')
            if neg_input_ids is not None:
                neg_input_ids = neg_input_ids.to(self.device, non_blocking=True)
                neg_attention_mask = neg_attention_mask.to(self.device, non_blocking=True)
            
            # ============================================================
            # Forward Pass: Use Gradient Caching if enabled, else standard
            # ============================================================
            if self.use_grad_cache:
                # GRADIENT CACHING MODE
                # GradCache handles the forward/backward internally
                if neg_input_ids is not None:
                    logger.warning("Hard negatives (neg_input_ids) are not supported with GradCache yet. Ignoring.")
                if intra_img_weight > 0:
                    logger.warning("L_i2i (image intra-modal) not supported with GradCache yet. Ignoring.")

                # Generate paraphrases for full batch before GradCache phases
                para_input_ids, para_attention_mask = None, None
                if intra_txt_weight > 0 and self.paraphraser is not None:
                    captions = batch['caption']  # list[str], len=N
                    para_input_ids, para_attention_mask = self.paraphraser.generate(captions)
                    # para_input_ids: [N, 77], para_attention_mask: [N, 77]

                # Zero gradients before GradCache forward (which accumulates gradients)
                self.optimizer.zero_grad()

                # GradCache performs forward and backward internally
                loss_dict = self.grad_cache.forward(
                    images, input_ids, attention_mask,
                    para_input_ids=para_input_ids,
                    para_attention_mask=para_attention_mask,
                )
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
                    # Zero gradients before forward to avoid stale gradient accumulation
                    self.optimizer.zero_grad()
                    with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                        img_embeds = self.model.encode_image(images)
                        txt_embeds = self.model.encode_text(input_ids, attention_mask)
                        img_aug_embeds = None
                        txt_aug_embeds = None
                        if intra_img_weight > 0:
                            img_aug_embeds = self.model.encode_image(
                                batch['image_aug'].to(self.device, non_blocking=True))
                        if intra_txt_weight > 0 and self.paraphraser is not None:
                            para_ids, para_mask = self.paraphraser.generate(
                                batch['caption'])
                            txt_aug_embeds = self.model.encode_text(para_ids, para_mask)

                        loss_dict = self.criterion(
                            img_embeds, txt_embeds,
                            self.model.clip.logit_scale,
                            img_aug_embeds, txt_aug_embeds
                        )
                        loss = loss_dict["loss_total"]

                    # Backward with gradient scaling
                    self.scaler.scale(loss).backward()
                    
                    # Unscale gradients to compute true grad norm
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = compute_grad_norm(self.model)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard precision training
                    img_embeds = self.model.encode_image(images)
                    txt_embeds = self.model.encode_text(input_ids, attention_mask)
                    neg_txt_embeds = None
                    if neg_input_ids is not None:
                        neg_txt_embeds = self.model.encode_text(neg_input_ids, neg_attention_mask)
                    img_aug_embeds = None
                    txt_aug_embeds = None
                    if intra_img_weight > 0:
                        img_aug_embeds = self.model.encode_image(batch['image_aug'].to(self.device, non_blocking=True))
                    # TODO: Re-enable when real text augmentation (e.g. synonym replacement or
                    # random token dropout) is implemented. Using identical inputs produces a
                    # trivially zero intra-text loss and is therefore disabled until then.
                    if False:  # intra_txt_weight > 0
                        txt_aug_embeds = self.model.encode_text(input_ids, attention_mask)

                    # loss is now a dict
                    loss_dict = self.criterion(
                        img_embeds, txt_embeds,
                        self.model.clip.logit_scale,
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
                fractional_epoch = (epoch + 1) + (step / num_batches)
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
                            "epoch": epoch + 1,
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
    def _extract_embeddings(self, loader):
        """
        Run the model over a dataloader and collect image/text embeddings + image IDs.
        Returns (img_embeds_unique, txt_embeds, image_ids, unique_image_ids, sentids,
                 first_occurrence_indices, unique_image_ids_list).
        """
        self.model.eval()
        img_embeds_list = []
        txt_embeds_list = []
        image_ids_list = []
        sentids_list = []

        for batch in loader:
            images = batch['image'].to(self.device, non_blocking=True)
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                img_emb, txt_emb = self.model(images, input_ids, attention_mask)
            img_embeds_list.append(img_emb.cpu())
            txt_embeds_list.append(txt_emb.cpu())
            image_ids_list.append(batch['image_id'].cpu())
            sentids_list.append(batch['sentid'].cpu())

        img_embeds = torch.cat(img_embeds_list, dim=0)
        txt_embeds = torch.cat(txt_embeds_list, dim=0)
        image_ids = torch.cat(image_ids_list, dim=0)
        sentids = torch.cat(sentids_list, dim=0)

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
        img_embeds_unique = img_embeds[first_occurrence_indices]

        return img_embeds_unique, txt_embeds, image_ids, unique_image_ids, sentids, first_occurrence_indices, unique_image_ids_list

    def _compute_standard_metrics(self, img_embeds_unique, txt_embeds, image_ids, unique_image_ids, sims, prefix="val"):
        """
        Computes standard R@1/5/10, mAP@R (mAP@5/10), R-Precision for both i2t and t2i.
        Used for val (always) and test (Flickr, or COCO fallback).
        Returns dict with keys prefixed by `prefix/`.
        """
        from .metrics import compute_mapr_rprecision, build_ranked_dicts

        r_t2i, r_i2t = compute_recall_at_k(img_embeds_unique, txt_embeds, image_ids, unique_image_ids, sims=sims)
        map_t2i, map_i2t = compute_map_at_k(img_embeds_unique, txt_embeds, image_ids, unique_image_ids, k_values=[5, 10], sims=sims)

        # mAP@R and R-Precision via build_ranked_dicts + compute_mapr_rprecision
        sims_np = sims.t().numpy()  # [N_imgs, N_txts]^T -> [N_txts, N_imgs]
        unique_image_ids_list = unique_image_ids.tolist()
        caption_ids = list(range(txt_embeds.shape[0]))
        i2t_ranked, t2i_ranked = build_ranked_dicts(sims_np, unique_image_ids_list, caption_ids)

        # Ground truth: image_id -> set of caption indices, caption_idx -> set of image_ids
        from .metrics import _build_gt_mappings
        _, caption_to_image_idx, image_to_caption_indices = _build_gt_mappings(image_ids, unique_image_ids)
        gt_i2t = {unique_image_ids_list[img_idx]: set(cap_indices) for img_idx, cap_indices in image_to_caption_indices.items()}
        gt_t2i = {cap_idx: {unique_image_ids_list[caption_to_image_idx[cap_idx].item()]} for cap_idx in range(len(caption_to_image_idx))}

        mapr_rprec = compute_mapr_rprecision(i2t_ranked, t2i_ranked, gt_i2t, gt_t2i)

        return {
            f"{prefix}/r1_i2t":         r_i2t[1],
            f"{prefix}/r5_i2t":         r_i2t[5],
            f"{prefix}/r10_i2t":        r_i2t[10],
            f"{prefix}/r1_t2i":         r_t2i[1],
            f"{prefix}/r5_t2i":         r_t2i[5],
            f"{prefix}/r10_t2i":        r_t2i[10],
            f"{prefix}/mapr_i2t":       mapr_rprec['mapr_i2t'],
            f"{prefix}/mapr_t2i":       mapr_rprec['mapr_t2i'],
            f"{prefix}/rprecision_i2t": mapr_rprec['rprecision_i2t'],
            f"{prefix}/rprecision_t2i": mapr_rprec['rprecision_t2i'],
        }

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        logger.info(f"Starting Evaluation (epoch={epoch})...")

        img_embeds_unique, txt_embeds, image_ids, unique_image_ids, _, first_occurrence_indices, _ = \
            self._extract_embeddings(self.val_loader)

        # [N_imgs, D] x [D, N_txts] -> [N_imgs, N_txts]
        sims = torch.matmul(img_embeds_unique, txt_embeds.t())

        metrics = self._compute_standard_metrics(img_embeds_unique, txt_embeds, image_ids, unique_image_ids, sims, prefix="val")

        logger.info(
            f"Epoch {epoch} Results:\n"
            f"  I2T: R@1: {metrics['val/r1_i2t']:.2f} | R@5: {metrics['val/r5_i2t']:.2f} | R@10: {metrics['val/r10_i2t']:.2f} | "
            f"mAP@R: {metrics['val/mapr_i2t']:.2f} | R-Prec: {metrics['val/rprecision_i2t']:.2f}\n"
            f"  T2I: R@1: {metrics['val/r1_t2i']:.2f} | R@5: {metrics['val/r5_t2i']:.2f} | R@10: {metrics['val/r10_t2i']:.2f} | "
            f"mAP@R: {metrics['val/mapr_t2i']:.2f} | R-Prec: {metrics['val/rprecision_t2i']:.2f}"
        )

        if self.use_wandb:
            log_data = dict(metrics)
            if isinstance(epoch, int):
                log_data["epoch"] = epoch + 1
            try:
                wandb.log(log_data)
            except Exception as e:
                logger.warning(f"Failed to log to W&B: {e}")

            is_final_epoch = (epoch + 1) == self.config['training']['epochs']
            is_qualitative_epoch = (isinstance(epoch, int) and
                                    (epoch + 1) % 5 == 0 or is_final_epoch)
            if is_qualitative_epoch:
                try:
                    self._log_qualitative_table(epoch, txt_embeds, img_embeds_unique, first_occurrence_indices)
                except Exception as e:
                    logger.warning(f"Qualitative table logging failed: {e}")

        return metrics['val/r1_t2i']

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
        save_freq = self.config['logging']['save_freq']
        improved_since_last_save = False

        # If start_epoch is 0 (i.e., not resuming) and eval_epoch_zero is enabled, run initial evaluation.
        if start_epoch == 0 and self.config['logging'].get('eval_epoch_zero', False):
            logger.info("Running initial evaluation at Epoch 0...")
            score = self.evaluate(epoch=-1)
            if score > self.best_r1:
                self.best_r1 = score
            logger.info(f"Initial state saved. Starting training loop from Epoch {start_epoch}.")

        # --- MAIN TRAINING LOOP ---
        for epoch in range(start_epoch, self.config['training']['epochs']):
            # Train one epoch
            train_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")

            # Evaluation & Best Model Check
            is_eval_time = ((epoch + 1) % eval_frequency == 0) or (
                (epoch + 1) == self.config['training']['epochs']
            )
            if is_eval_time:
                score = self.evaluate(epoch)
                if score > self.best_r1:
                    self.best_r1 = score
                    improved_since_last_save = True
                    logger.info(f"New Best R@1: {score:.2f} found at Epoch {epoch+1}!")

            # Save only when save_freq is hit AND there has been improvement since last save
            is_save_time = (epoch + 1) % save_freq == 0 or (epoch + 1) == self.config['training']['epochs']
            if is_save_time and improved_since_last_save:
                self.save_checkpoint(epoch + 1)
                improved_since_last_save = False

        # --- TEST EVALUATION ---
        logger.info("Training complete. Running final test evaluation...")
        self._evaluate_test()

    def _evaluate_test(self):
        """
        Runs evaluation on the test split using the best saved checkpoint.
        Logs results to WandB as summary metrics under 'test/' prefix.
        ECCV metrics are computed here for COCO only.
        Val evaluation never uses ECCV — only test does.
        """
        best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        if not os.path.exists(best_path):
            logger.warning("No best_model.pth found. Skipping test evaluation.")
            return

        logger.info(f"Loading best checkpoint from {best_path} for test evaluation...")

        # Save current (last epoch) weights so we can restore after test eval
        last_epoch_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        checkpoint = torch.load(best_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Build test dataloader
        from .data import create_image_text_dataloader
        test_loader = create_image_text_dataloader(self.config, self.tokenizer, split='test')

        img_embeds_unique, txt_embeds, image_ids, unique_image_ids, sentids, _, unique_image_ids_list = \
            self._extract_embeddings(test_loader)

        # [N_imgs, D] x [D, N_txts] -> [N_imgs, N_txts]
        sims = torch.matmul(img_embeds_unique, txt_embeds.t())

        is_coco = self.config['data']['dataset'] == 'coco'

        if is_coco:
            # ECCV metrics — only on test, only for COCO
            sims_np = sims.t().numpy()  # [N_txts, N_imgs]
            eccv_scores = compute_eccv_metrics(
                sims_np,
                image_ids=unique_image_ids_list,
                caption_ids=sentids.tolist(),
                dataset='coco',
            )
            if eccv_scores:
                test_metrics = {
                    "test/coco_5k_r1_i2t":      eccv_scores.get('coco_5k_r1', {}).get('i2t', 0),
                    "test/coco_5k_r1_t2i":      eccv_scores.get('coco_5k_r1', {}).get('t2i', 0),
                    "test/coco_5k_r5_i2t":      eccv_scores.get('coco_5k_recalls', {}).get('i2t', {}).get(5, 0),
                    "test/coco_5k_r5_t2i":      eccv_scores.get('coco_5k_recalls', {}).get('t2i', {}).get(5, 0),
                    "test/coco_5k_r10_i2t":     eccv_scores.get('coco_5k_recalls', {}).get('i2t', {}).get(10, 0),
                    "test/coco_5k_r10_t2i":     eccv_scores.get('coco_5k_recalls', {}).get('t2i', {}).get(10, 0),
                    "test/coco_1k_r1_i2t":      eccv_scores.get('coco_1k_r1', {}).get('i2t', 0),
                    "test/coco_1k_r1_t2i":      eccv_scores.get('coco_1k_r1', {}).get('t2i', 0),
                    "test/eccv_map_at_r_i2t":   eccv_scores.get('eccv_map_at_r', {}).get('i2t', 0),
                    "test/eccv_map_at_r_t2i":   eccv_scores.get('eccv_map_at_r', {}).get('t2i', 0),
                    "test/eccv_rprecision_i2t":  eccv_scores.get('eccv_rprecision', {}).get('i2t', 0),
                    "test/eccv_rprecision_t2i":  eccv_scores.get('eccv_rprecision', {}).get('t2i', 0),
                    "test/cxc_r1_i2t":          eccv_scores.get('cxc_recalls', {}).get('i2t', {}).get(1, 0),
                    "test/cxc_r1_t2i":          eccv_scores.get('cxc_recalls', {}).get('t2i', {}).get(1, 0),
                }
            else:
                logger.warning("ECCV metrics returned empty. Falling back to standard R@K for test.")
                test_metrics = self._compute_standard_metrics(
                    img_embeds_unique, txt_embeds, image_ids, unique_image_ids, sims, prefix="test"
                )
        else:
            # Flickr — standard R@K + mAP@R + R-Precision
            test_metrics = self._compute_standard_metrics(
                img_embeds_unique, txt_embeds, image_ids, unique_image_ids, sims, prefix="test"
            )

        # Log to WandB as summary (not as a time-series metric)
        if self.use_wandb and wandb.run is not None:
            for k, v in test_metrics.items():
                wandb.run.summary[k] = v

        logger.info("Test Results:")
        for k, v in test_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        # Restore last epoch weights
        self.model.load_state_dict(last_epoch_state)
