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

import gc
import math
import torch
import logging
import wandb
import os
import time
import random
from .metrics import AverageMeter, compute_recall_at_k, compute_eccv_metrics
import torch.nn.functional as F
from .utils import compute_grad_norm, chunked_matmul
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
        
        # Best score tracking for checkpointing
        self.best_r1 = 0.0
        
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
        self.use_grad_cache = config['training']['use_grad_cache']
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
        
        # Initialize paraphraser for L_text_text = InfoNCE(T_orig, T_paraphrased)
        self.paraphraser = None
        para_cfg = config['paraphraser']
        para_type = para_cfg['type']
        para_paths = para_cfg['paths']
        intra_txt_weight = config['loss']['intra_txt_weight']
        if intra_txt_weight > 0:
            if para_type not in para_paths:
                raise ValueError(
                    f"paraphraser.type='{para_type}' is not defined in paraphraser.paths "
                    f"(available: {sorted(para_paths.keys())})."
                )
            rewrites_path = para_paths[para_type]
            from .paraphraser import PrecomputedLLMParaphraser
            self.paraphraser = PrecomputedLLMParaphraser(
                rewrites_path=rewrites_path,
                tokenizer=clip_tokenizer,
                device=device,
                max_length=config['data']['max_length'],
                seed=config['training']['seed'],
            )
            logger.info(f"Using LLM pre-computed paraphraser (LaCLIP-style, type={para_type}).")
        else:
            logger.info("Paraphraser disabled (intra_txt_weight=0).")

        # Initialize hard negative generator for B2 (POS-tag swapping)
        self.use_hard_negatives = config.get('loss', {}).get('hard_negatives', False)
        if self.use_hard_negatives:
            from .data import HardNegativeGenerator
            self.hard_neg_generator = HardNegativeGenerator(
                seed=config['training']['seed']
            )
            logger.info("Hard negative generator initialized (spaCy POS-tag swapping).")
        else:
            self.hard_neg_generator = None

        # Initialize multi-label classification auxiliary loss (B4 — COCO only)
        self.cls_weight = config['loss'].get('cls_weight', 0.0)
        if self.cls_weight > 0:
            dataset_name = config['data']['dataset'].lower()
            if dataset_name != 'coco':
                raise ValueError(
                    f"loss.cls_weight={self.cls_weight} (B4 multi-label aux loss) "
                    f"requires data.dataset=coco, got dataset={dataset_name!r}. "
                    "B4 uses COCO category labels only."
                )
            cls_label_path = config['loss'].get('cls_label_path')
            if not cls_label_path or not os.path.isfile(cls_label_path):
                raise FileNotFoundError(
                    f"cls_weight={self.cls_weight} but cls_label_path is missing or invalid: "
                    f"'{cls_label_path}'. Run tools/build_coco_multilabel.py first."
                )
            self.cls_label_map = torch.load(cls_label_path, map_location=self.device)
            num_cls = config['loss']['cls_num_classes']
            self._cls_zero_vec = torch.zeros(num_cls, dtype=torch.float32, device=self.device)
            logger.info(
                f"Multi-label classification enabled: weight={self.cls_weight}, "
                f"{len(self.cls_label_map)} images loaded from {cls_label_path}"
            )
        else:
            self.cls_label_map = None

        # B5a/B5b/B5c gating: cached at init so the hot loop is a single bool check.
        self.seg_mode = config.get('model', {}).get('seg_mode')
        self.use_seg_ids = self.seg_mode is not None
        if self.use_seg_ids:
            if self.cls_weight > 0:
                raise RuntimeError(
                    "B4 (loss.cls_weight > 0) and B5 (model.seg_mode set) are mutually exclusive: "
                    "cls_head reads pooler_output while seg-injection rewrites hidden_states. "
                    "Either disable cls_weight or unset seg_mode."
                )
            if self.use_grad_cache:
                raise RuntimeError(
                    "B5 (model.seg_mode set) is incompatible with training.use_grad_cache=true. "
                    "GradCache does not yet thread seg_ids; either disable grad_cache or extend "
                    "src/grad_cache.py to forward seg_ids."
                )
            logger.info(
                f"B5 active in Trainer: model.seg_mode={self.seg_mode}, "
                f"vocab={config['model'].get('seg_vocab_size')}, "
                f"feat_dim={config['model'].get('seg_feature_dim')}"
            )

        # SIGTERM flag — set externally via signal handler in run.py
        self._sigterm_received = False

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

        # Explicitly move state_dict to CPU to avoid torch.save creating
        # a second full-model CPU copy while GPU tensors are still alive.
        cpu_model_sd = {k: v.cpu() for k, v in self.model.state_dict().items()}
        cpu_optim_sd = self.optimizer.state_dict()  # already CPU
        cpu_sched_sd = self.scheduler.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': cpu_model_sd,
            'optimizer_state_dict': cpu_optim_sd,
            'scheduler_state_dict': cpu_sched_sd,
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
            return True
        except (OSError, RuntimeError) as e:
            logger.error(f"DISK ERROR: Could not save best_model.pth at Epoch {epoch}. {e}")
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return False
        finally:
            del checkpoint, cpu_model_sd
            gc.collect()

    def train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        batch_time = AverageMeter()
        
        num_batches = len(self.train_loader)
        batch_size = self.config['training']['batch_size']
        intra_img_weight = self.config['loss']['intra_img_weight']
        intra_txt_weight = self.config['loss']['intra_txt_weight']

        # Initialize grad_norm for logging
        grad_norm = 0.0
        end_time = time.time()
        
        for step, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device, non_blocking=True)
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)

            # B5 inter-modal seg_ids; None for every other variant. Augmented
            # views (image_aug_a/b) intentionally drop seg_ids — RandomResizedCrop
            # breaks spatial alignment with the precomputed map.
            seg_ids = None
            if self.use_seg_ids:
                if 'seg_ids' not in batch:
                    raise KeyError(
                        "B5 active (model.seg_mode set) but batch is missing 'seg_ids'. "
                        "Check the dataloader factory."
                    )
                seg_ids = batch['seg_ids'].to(self.device, non_blocking=True)

            neg_input_ids = batch.get('negative_input_ids')
            neg_attention_mask = batch.get('negative_attention_mask')
            if neg_input_ids is not None:
                neg_input_ids = neg_input_ids.to(self.device, non_blocking=True)
                neg_attention_mask = neg_attention_mask.to(self.device, non_blocking=True)

            if step == 0 and epoch == 0:
                loss_type = self.config['loss']['type']
                w_img = self.config['loss']['intra_img_weight']
                w_txt = self.config['loss']['intra_txt_weight']
                # B0v3 pair design: dataset returns image_aug_a + image_aug_b for the
                # intra-modal image-image pair, NOT a single image_aug.
                pair_available = ('image_aug_a' in batch) and ('image_aug_b' in batch)
                txt_aug_available = self.paraphraser is not None
                logger.info(
                    f"Loss config at step 0: type={loss_type}, "
                    f"intra_img_weight={w_img}, intra_txt_weight={w_txt}, "
                    f"img_aug_pair_available={pair_available}, "
                    f"txt_aug_available={txt_aug_available}"
                )
                if w_img > 0 and not pair_available:
                    raise RuntimeError(
                        f"intra_img_weight={w_img} but image_aug_a/image_aug_b are not in batch. "
                        "Check that k_photometric_augs > 0 and the dataset returns the B0v3 pair."
                    )
                if w_txt > 0 and not txt_aug_available:
                    raise RuntimeError(
                        f"intra_txt_weight={w_txt} but paraphraser is not initialized."
                    )

            # ============================================================
            # Forward Pass: Use Gradient Caching if enabled, else standard
            # ============================================================
            if self.use_grad_cache:
                # GRADIENT CACHING MODE
                # GradCache currently does NOT support B0v3 pair-based intra-modal
                # (would require threading both aug views and both paraphrase pairs
                # through the cache phases). Fail loud rather than silently degrade.
                if intra_img_weight > 0 or intra_txt_weight > 0:
                    raise RuntimeError(
                        "GradCache does not yet support B0v3 pair-based intra-modal "
                        "(intra_img_weight>0 or intra_txt_weight>0). "
                        "Either disable training.use_grad_cache or extend src/grad_cache.py "
                        "to thread image_aug_a/image_aug_b + paraphrase pairs."
                    )
                if neg_input_ids is not None:
                    logger.warning("Hard negatives (neg_input_ids) are not supported with GradCache yet. Ignoring.")
                if self.use_hard_negatives:
                    logger.warning("Syntactic hard negatives (loss.hard_negatives) are not supported with GradCache yet. Ignoring.")

                # Zero gradients before GradCache forward (which accumulates gradients)
                self.optimizer.zero_grad()

                # GradCache performs forward and backward internally (inter-modal only)
                loss_dict = self.grad_cache.forward(
                    images, input_ids, attention_mask,
                    para_input_ids=None,
                    para_attention_mask=None,
                    image_aug=None,
                )
                loss = loss_dict["loss_total"]
                
                # Compute grad norm and step optimizer
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = compute_grad_norm(self.model)
                    if math.isfinite(grad_norm):
                        self.scaler.step(self.optimizer)
                    else:
                        logger.critical(
                            f"Skipping optimizer step at epoch {epoch} step {step}: "
                            f"grad_norm={grad_norm}. Weights not updated."
                        )
                    self.scaler.update()
                else:
                    grad_norm = compute_grad_norm(self.model)
                    if math.isfinite(grad_norm):
                        self.optimizer.step()
                    else:
                        logger.critical(
                            f"Skipping optimizer step at epoch {epoch} step {step}: "
                            f"grad_norm={grad_norm}. Weights not updated."
                        )
            
            else:
                # STANDARD MODE (Original Implementation)
                if self.use_amp:
                    # Zero gradients before forward to avoid stale gradient accumulation
                    self.optimizer.zero_grad()
                    with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                        cls_logits = None
                        if self.cls_weight > 0 and self.model.cls_head is not None:
                            img_embeds, cls_logits = self.model.encode_image_with_cls(images)
                        else:
                            img_embeds = self.model.encode_image(images, seg_ids=seg_ids)
                        txt_embeds = self.model.encode_text(input_ids, attention_mask)
                        img_aug_a_embeds = None
                        img_aug_b_embeds = None
                        txt_aug_a_embeds = None
                        txt_aug_b_embeds = None
                        neg_txt_embeds = None

                        # B0v3: encode the two augmented image views for image-image
                        # intra-modal pair. Both views are produced independently by
                        # transform_aug applied twice in __getitem__.
                        if intra_img_weight > 0:
                            with torch.no_grad():
                                img_aug_a_embeds = self.model.encode_image(
                                    batch['image_aug_a'].to(self.device, non_blocking=True))
                                img_aug_b_embeds = self.model.encode_image(
                                    batch['image_aug_b'].to(self.device, non_blocking=True))

                        # Merge paraphrase pair + hard-neg text into a single no_grad
                        # encode_text call to avoid redundant kernel launches.
                        nograd_ids_parts = []
                        nograd_mask_parts = []
                        _para_n_a = 0
                        _para_n_b = 0
                        _neg_n = 0

                        if intra_txt_weight > 0 and self.paraphraser is not None:
                            pa_ids, pa_mask, pb_ids, pb_mask = self.paraphraser.generate_pair(
                                batch['sentid'].tolist())
                            nograd_ids_parts.append(pa_ids)
                            nograd_mask_parts.append(pa_mask)
                            _para_n_a = pa_ids.shape[0]
                            nograd_ids_parts.append(pb_ids)
                            nograd_mask_parts.append(pb_mask)
                            _para_n_b = pb_ids.shape[0]

                        if self.hard_neg_generator is not None:
                            captions = batch['caption']
                            hard_neg_captions = self.hard_neg_generator.generate(captions)
                            tokenized_neg = self.tokenizer(
                                hard_neg_captions,
                                padding='max_length',
                                truncation=True,
                                max_length=self.config['data']['max_length'],
                                return_tensors='pt'
                            )
                            neg_ids = tokenized_neg['input_ids'].to(self.device)
                            neg_mask = tokenized_neg['attention_mask'].to(self.device)
                            nograd_ids_parts.append(neg_ids)
                            nograd_mask_parts.append(neg_mask)
                            _neg_n = neg_ids.shape[0]

                        # Single fused no_grad encode_text for all auxiliary texts
                        if nograd_ids_parts:
                            merged_ids = torch.cat(nograd_ids_parts, dim=0)
                            merged_mask = torch.cat(nograd_mask_parts, dim=0)
                            with torch.no_grad():
                                merged_embeds = self.model.encode_text(merged_ids, merged_mask)
                            offset = 0
                            if _para_n_a > 0:
                                txt_aug_a_embeds = merged_embeds[offset:offset + _para_n_a]
                                offset += _para_n_a
                            if _para_n_b > 0:
                                txt_aug_b_embeds = merged_embeds[offset:offset + _para_n_b]
                                offset += _para_n_b
                            if _neg_n > 0:
                                neg_txt_embeds = merged_embeds[offset:offset + _neg_n]

                        if step == 0 and epoch == 0:
                            loss_type = self.config['loss']['type']
                            w_img = self.config['loss']['intra_img_weight']
                            w_txt = self.config['loss']['intra_txt_weight']
                            logger.info(
                                f"Loss config at step 0: type={loss_type}, "
                                f"intra_img_weight={w_img}, intra_txt_weight={w_txt}, "
                                f"img_aug_pair_available={img_aug_a_embeds is not None and img_aug_b_embeds is not None}, "
                                f"txt_aug_pair_available={txt_aug_a_embeds is not None and txt_aug_b_embeds is not None}"
                            )
                            if w_img > 0 and (img_aug_a_embeds is None or img_aug_b_embeds is None):
                                raise RuntimeError(
                                    f"intra_img_weight={w_img} but image_aug pair is None. "
                                    "Check that k_photometric_augs > 0 and dataset returns image_aug_a/image_aug_b."
                                )
                            if w_txt > 0 and (txt_aug_a_embeds is None or txt_aug_b_embeds is None):
                                raise RuntimeError(
                                    f"intra_txt_weight={w_txt} but paraphrase pair embeds are None. "
                                    "Check that paraphraser is initialized and rewrites file has >= 2 entries per sentid."
                                )
                            if self.use_hard_negatives:
                                if neg_txt_embeds is None:
                                    raise RuntimeError(
                                        "loss.hard_negatives=true but neg_txt_embeds is None at step 0. "
                                        "Check HardNegativeGenerator initialization."
                                    )
                                captions_v = batch['caption']
                                hard_negs_v = self.hard_neg_generator.generate(captions_v)
                                n_different = sum(h != c for h, c in zip(hard_negs_v, captions_v))
                                pct_different = n_different / len(captions_v)
                                if pct_different < 0.5:
                                    logger.warning(
                                        f"Only {pct_different:.1%} of hard negatives differ from "
                                        f"source captions. Hard negative quality may be low."
                                    )
                                else:
                                    logger.info(
                                        f"Hard negatives verified: {pct_different:.1%} differ from source. "
                                        f"Example: '{captions_v[0]}' -> '{hard_negs_v[0]}'"
                                    )

                        loss_dict = self.criterion(
                            img_embeds, txt_embeds,
                            self.model.clip.logit_scale,
                            img_aug_a_embeds=img_aug_a_embeds,
                            img_aug_b_embeds=img_aug_b_embeds,
                            txt_aug_a_embeds=txt_aug_a_embeds,
                            txt_aug_b_embeds=txt_aug_b_embeds,
                            neg_txt_embeds=neg_txt_embeds,
                        )
                        loss = loss_dict["loss_total"]

                        # Multi-label classification auxiliary loss
                        if cls_logits is not None:
                            image_ids = batch['image_id']
                            label_vecs = torch.stack(
                                [self.cls_label_map.get(iid.item(), self._cls_zero_vec) for iid in image_ids]
                            )  # [B, 80]
                            loss_cls = F.binary_cross_entropy_with_logits(
                                cls_logits, label_vecs)  # scalar
                            loss = loss + self.cls_weight * loss_cls
                            loss_dict["loss_cls"] = loss_cls
                            loss_dict["loss_total"] = loss

                    # Backward with gradient scaling
                    self.scaler.scale(loss).backward()
                    
                    # Unscale gradients to compute true grad norm
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = compute_grad_norm(self.model)
                    if math.isfinite(grad_norm):
                        self.scaler.step(self.optimizer)
                    else:
                        logger.critical(
                            f"Skipping optimizer step at epoch {epoch} step {step}: "
                            f"grad_norm={grad_norm}. Weights not updated."
                        )
                    self.scaler.update()
                else:
                    # Standard precision training
                    self.optimizer.zero_grad()
                    cls_logits = None
                    if self.cls_weight > 0 and self.model.cls_head is not None:
                        img_embeds, cls_logits = self.model.encode_image_with_cls(images)
                    else:
                        img_embeds = self.model.encode_image(images, seg_ids=seg_ids)
                    txt_embeds = self.model.encode_text(input_ids, attention_mask)
                    img_aug_a_embeds = None
                    img_aug_b_embeds = None
                    txt_aug_a_embeds = None
                    txt_aug_b_embeds = None
                    neg_txt_embeds = None

                    # B0v3: encode the two augmented image views for image-image
                    # intra-modal pair.
                    if intra_img_weight > 0:
                        with torch.no_grad():
                            img_aug_a_embeds = self.model.encode_image(
                                batch['image_aug_a'].to(self.device, non_blocking=True))
                            img_aug_b_embeds = self.model.encode_image(
                                batch['image_aug_b'].to(self.device, non_blocking=True))

                    # Merge paraphrase pair + hard-neg text into a single no_grad
                    # encode_text call to avoid redundant kernel launches.
                    nograd_ids_parts = []
                    nograd_mask_parts = []
                    _para_n_a = 0
                    _para_n_b = 0
                    _neg_n = 0

                    if intra_txt_weight > 0 and self.paraphraser is not None:
                        pa_ids, pa_mask, pb_ids, pb_mask = self.paraphraser.generate_pair(
                            batch['sentid'].tolist())
                        nograd_ids_parts.append(pa_ids)
                        nograd_mask_parts.append(pa_mask)
                        _para_n_a = pa_ids.shape[0]
                        nograd_ids_parts.append(pb_ids)
                        nograd_mask_parts.append(pb_mask)
                        _para_n_b = pb_ids.shape[0]

                    if self.hard_neg_generator is not None:
                        captions = batch['caption']
                        hard_neg_captions = self.hard_neg_generator.generate(captions)
                        tokenized_neg = self.tokenizer(
                            hard_neg_captions,
                            padding='max_length',
                            truncation=True,
                            max_length=self.config['data']['max_length'],
                            return_tensors='pt'
                        )
                        neg_ids = tokenized_neg['input_ids'].to(self.device)
                        neg_mask = tokenized_neg['attention_mask'].to(self.device)
                        nograd_ids_parts.append(neg_ids)
                        nograd_mask_parts.append(neg_mask)
                        _neg_n = neg_ids.shape[0]

                    # Single fused no_grad encode_text for all auxiliary texts
                    if nograd_ids_parts:
                        merged_ids = torch.cat(nograd_ids_parts, dim=0)
                        merged_mask = torch.cat(nograd_mask_parts, dim=0)
                        with torch.no_grad():
                            merged_embeds = self.model.encode_text(merged_ids, merged_mask)
                        offset = 0
                        if _para_n_a > 0:
                            txt_aug_a_embeds = merged_embeds[offset:offset + _para_n_a]
                            offset += _para_n_a
                        if _para_n_b > 0:
                            txt_aug_b_embeds = merged_embeds[offset:offset + _para_n_b]
                            offset += _para_n_b
                        if _neg_n > 0:
                            neg_txt_embeds = merged_embeds[offset:offset + _neg_n]

                    if step == 0 and epoch == 0:
                        loss_type = self.config['loss']['type']
                        w_img = self.config['loss']['intra_img_weight']
                        w_txt = self.config['loss']['intra_txt_weight']
                        logger.info(
                            f"Loss config at step 0: type={loss_type}, "
                            f"intra_img_weight={w_img}, intra_txt_weight={w_txt}, "
                            f"img_aug_pair_available={img_aug_a_embeds is not None and img_aug_b_embeds is not None}, "
                            f"txt_aug_pair_available={txt_aug_a_embeds is not None and txt_aug_b_embeds is not None}"
                        )
                        if w_img > 0 and (img_aug_a_embeds is None or img_aug_b_embeds is None):
                            raise RuntimeError(
                                f"intra_img_weight={w_img} but image_aug pair is None. "
                                "Check that k_photometric_augs > 0 and dataset returns image_aug_a/image_aug_b."
                            )
                        if w_txt > 0 and (txt_aug_a_embeds is None or txt_aug_b_embeds is None):
                            raise RuntimeError(
                                f"intra_txt_weight={w_txt} but paraphrase pair embeds are None. "
                                "Check that paraphraser is initialized and rewrites file has >= 2 entries per sentid."
                            )
                        if self.use_hard_negatives:
                            if neg_txt_embeds is None:
                                raise RuntimeError(
                                    "loss.hard_negatives=true but neg_txt_embeds is None at step 0. "
                                    "Check HardNegativeGenerator initialization."
                                )
                            captions_v = batch['caption']
                            hard_negs_v = self.hard_neg_generator.generate(captions_v)
                            n_different = sum(h != c for h, c in zip(hard_negs_v, captions_v))
                            pct_different = n_different / len(captions_v)
                            if pct_different < 0.5:
                                logger.warning(
                                    f"Only {pct_different:.1%} of hard negatives differ from "
                                    f"source captions. Hard negative quality may be low."
                                )
                            else:
                                logger.info(
                                    f"Hard negatives verified: {pct_different:.1%} differ from source. "
                                    f"Example: '{captions_v[0]}' -> '{hard_negs_v[0]}'"
                                )

                    loss_dict = self.criterion(
                        img_embeds, txt_embeds,
                        self.model.clip.logit_scale,
                        img_aug_a_embeds=img_aug_a_embeds,
                        img_aug_b_embeds=img_aug_b_embeds,
                        txt_aug_a_embeds=txt_aug_a_embeds,
                        txt_aug_b_embeds=txt_aug_b_embeds,
                        neg_txt_embeds=neg_txt_embeds,
                    )
                    loss = loss_dict["loss_total"]

                    # Multi-label classification auxiliary loss
                    if cls_logits is not None:
                        image_ids = batch['image_id']
                        label_vecs = torch.stack(
                            [self.cls_label_map.get(iid.item(), self._cls_zero_vec) for iid in image_ids]
                        )  # [B, 80]
                        loss_cls = F.binary_cross_entropy_with_logits(
                            cls_logits, label_vecs)  # scalar
                        loss = loss + self.cls_weight * loss_cls
                        loss_dict["loss_cls"] = loss_cls
                        loss_dict["loss_total"] = loss

                    loss.backward()

                    # Compute grad norm before optimizer step
                    grad_norm = compute_grad_norm(self.model)
                    if math.isfinite(grad_norm):
                        self.optimizer.step()
                    else:
                        logger.critical(
                            f"Skipping optimizer step at epoch {epoch} step {step}: "
                            f"grad_norm={grad_norm}. Weights not updated."
                        )
            
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
                        if "loss_cls" in loss_dict:
                            log_dict["train/loss_cls"] = loss_dict["loss_cls"].item()
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

            # B5: thread per-patch seg_ids through eval batches too. Augmented
            # views aren't generated during eval (transform_aug=transform), so
            # only the inter-modal forward needs seg_ids here.
            seg_ids = None
            if self.use_seg_ids:
                if 'seg_ids' not in batch:
                    raise KeyError(
                        "B5 active but eval batch is missing 'seg_ids'. "
                        "Check the dataloader factory."
                    )
                seg_ids = batch['seg_ids'].to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                img_emb, txt_emb = self.model(images, input_ids, attention_mask, seg_ids=seg_ids)
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

        # [N_imgs, D] x [D, N_txts] -> [N_imgs, N_txts]  (chunked to limit peak RAM)
        sims = chunked_matmul(img_embeds_unique, txt_embeds)

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

        # Explicitly free evaluation artifacts before checkpoint save
        del img_embeds_unique, txt_embeds, sims, image_ids, unique_image_ids
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return metrics['val/r1_t2i']

    def _log_qualitative_table(self, epoch, txt_embeds, img_embeds_unique, first_occurrence_indices):
        """
        Logs a WandB Table visualizing Text-to-Image retrieval.
        Columns: Caption | Ground Truth | Rank 1 | Rank 2 | Rank 3
        """
        num_samples = len(txt_embeds)
        sample_count = min(5, num_samples)
        rng = random.Random(epoch if isinstance(epoch, int) else 0)
        indices = rng.sample(range(num_samples), sample_count)

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
        if start_epoch == 0 and self.config['logging']['eval_epoch_zero']:
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
                if self.save_checkpoint(epoch + 1):
                    improved_since_last_save = False

            if self._sigterm_received:
                logger.critical("SIGTERM: Emergency checkpoint save and exit.")
                self.save_checkpoint(epoch + 1)
                raise SystemExit(0)

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

        checkpoint = torch.load(best_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        gc.collect()
        self.model.eval()

        # Build test dataloader
        from .data import create_image_text_dataloader
        test_loader = create_image_text_dataloader(self.config, self.tokenizer, split='test')

        img_embeds_unique, txt_embeds, image_ids, unique_image_ids, sentids, _, unique_image_ids_list = \
            self._extract_embeddings(test_loader)

        # [N_imgs, D] x [D, N_txts] -> [N_imgs, N_txts]  (chunked to limit peak RAM)
        sims = chunked_matmul(img_embeds_unique, txt_embeds)

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
                # Validate expected keys exist before extraction
                _expected_keys = [
                    'coco_5k_r1', 'coco_5k_r5', 'coco_5k_r10',
                    'coco_1k_r1', 'coco_1k_r5', 'coco_1k_r10',
                    'cxc_r1', 'cxc_r5', 'cxc_r10',
                    'eccv_map_at_r', 'eccv_rprecision',
                ]
                _missing = [k for k in _expected_keys if k not in eccv_scores]
                if _missing:
                    raise RuntimeError(
                        f"eccv_caption returned unexpected format. "
                        f"Missing keys: {_missing}. "
                        f"Got keys: {sorted(eccv_scores.keys())}"
                    )
                logger.info(f"ECCV scores keys validated: {sorted(eccv_scores.keys())}")
                test_metrics = {
                    # COCO 5K
                    "test/coco_5k_r1_i2t":      eccv_scores.get('coco_5k_r1', {}).get('i2t', 0),
                    "test/coco_5k_r1_t2i":      eccv_scores.get('coco_5k_r1', {}).get('t2i', 0),
                    "test/coco_5k_r5_i2t":      eccv_scores.get('coco_5k_r5', {}).get('i2t', 0),
                    "test/coco_5k_r5_t2i":      eccv_scores.get('coco_5k_r5', {}).get('t2i', 0),
                    "test/coco_5k_r10_i2t":     eccv_scores.get('coco_5k_r10', {}).get('i2t', 0),
                    "test/coco_5k_r10_t2i":     eccv_scores.get('coco_5k_r10', {}).get('t2i', 0),
                    # COCO 1K
                    "test/coco_1k_r1_i2t":      eccv_scores.get('coco_1k_r1', {}).get('i2t', 0),
                    "test/coco_1k_r1_t2i":      eccv_scores.get('coco_1k_r1', {}).get('t2i', 0),
                    "test/coco_1k_r5_i2t":      eccv_scores.get('coco_1k_r5', {}).get('i2t', 0),
                    "test/coco_1k_r5_t2i":      eccv_scores.get('coco_1k_r5', {}).get('t2i', 0),
                    "test/coco_1k_r10_i2t":     eccv_scores.get('coco_1k_r10', {}).get('i2t', 0),
                    "test/coco_1k_r10_t2i":     eccv_scores.get('coco_1k_r10', {}).get('t2i', 0),
                    # ECCV
                    "test/eccv_map_at_r_i2t":   eccv_scores.get('eccv_map_at_r', {}).get('i2t', 0),
                    "test/eccv_map_at_r_t2i":   eccv_scores.get('eccv_map_at_r', {}).get('t2i', 0),
                    "test/eccv_rprecision_i2t":  eccv_scores.get('eccv_rprecision', {}).get('i2t', 0),
                    "test/eccv_rprecision_t2i":  eccv_scores.get('eccv_rprecision', {}).get('t2i', 0),
                    # CxC
                    "test/cxc_r1_i2t":          eccv_scores.get('cxc_r1', {}).get('i2t', 0),
                    "test/cxc_r1_t2i":          eccv_scores.get('cxc_r1', {}).get('t2i', 0),
                    "test/cxc_r5_i2t":          eccv_scores.get('cxc_r5', {}).get('i2t', 0),
                    "test/cxc_r5_t2i":          eccv_scores.get('cxc_r5', {}).get('t2i', 0),
                    "test/cxc_r10_i2t":         eccv_scores.get('cxc_r10', {}).get('i2t', 0),
                    "test/cxc_r10_t2i":         eccv_scores.get('cxc_r10', {}).get('t2i', 0),
                }
                # Also compute standard R@K metrics (matches Flickr test behavior)
                standard = self._compute_standard_metrics(
                    img_embeds_unique, txt_embeds, image_ids, unique_image_ids, sims, prefix="test"
                )
                test_metrics.update(standard)
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
