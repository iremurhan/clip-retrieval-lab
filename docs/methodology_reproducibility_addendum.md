# Methodology Reproducibility Addendum

## Scope Restrictions And Dataset-Specific Side Inputs

B4 is evaluated only on COCO because its auxiliary classifier uses the 80 COCO object categories and Flickr30K does not provide exhaustive image-level category labels. B5b is not excluded on Flickr30K: Flickr30K has no ground-truth COCO category annotations, so its B5b semantic segment ids are pseudo-labels rather than annotations. In B5b, SAM segments are first generated for each image. On COCO, each SAM segment is assigned a COCO-80 category by IoU matching to COCO instance annotations with threshold 0.3. On Flickr30K, each SAM segment crop is classified zero-shot with CLIP over prompts of the form "a photo of a {category}" for the same 80 COCO categories; the top-1 class is kept only when its probability is at least 0.3, otherwise the segment is mapped to background. Flickr30K SAM segments whose bounding box is smaller than 4 px on either side are also mapped to background. Thus B5b results on Flickr30K should be described as using CLIP-derived pseudo-semantic labels, not as using ground-truth category supervision.

## SAM Preprocessing

SAM masks are precomputed offline with the SAM ViT-B checkpoint `sam_vit_b_01ec64.pth` and `SamAutomaticMaskGenerator`. The only non-default mask-generator parameter explicitly set in code is `points_per_side=32`; other generator parameters, including NMS thresholds and minimum mask area, use the Segment Anything library defaults. Each generated mask set is collapsed into a single integer segment map per image and saved as a compressed `.npz`; B5a, B5b, and B5c then convert these pixel-space segment maps to 24 x 24 patch-level side inputs for CLIP ViT-L/14 at 336 px.

B5c uses a five-dimensional continuous feature vector for the dominant segment of each patch: normalized centroid x, normalized centroid y, area fraction, clipped log aspect ratio, and fraction of CLIP patches occupied by the segment. Background patches receive the zero vector.

For B5d/B5e multi-stream variants, the SAM image encoder is run offline and remains frozen. The stored side input is the dense SAM ViT-B image-encoder feature map, originally 256 x 64 x 64, adaptively average-pooled to 256 x 8 x 8 and saved in float16. During CLIP training, the additional trainable module is only the fusion block: gated residual fusion, cross-attention, concatenation projection, or SAM skip fusion depending on the variant. The SAM encoder is not executed in the training forward pass, which keeps memory compatible with the default batch size.

## Text And Negative-Caption Augmentation

LLM paraphrases are generated offline from the Karpathy train/restval captions using HuggingFace `AutoModelForCausalLM` in bfloat16 with `device_map="auto"`. The active configuration uses `meta-llama/Meta-Llama-3-8B-Instruct` when `paraphraser.type=llama`; the legacy default script also supports `mistralai/Mistral-7B-Instruct-v0.2`. The prompt instructs the model to rewrite the caption with different sentence structure while preserving all objects, attributes, relationships, and actions, and to output only the rewritten caption. Generation uses six sampled candidates per caption, `temperature=0.9`, `top_p=0.95`, `top_k=50`, and `max_new_tokens=128`; after normalized deduplication, two distinct rewrites are stored per caption. Training samples two distinct precomputed rewrites per caption on demand for the text-text intra-modal loss; no LLM is run during training.

Hard negatives are generated online for B2 by spaCy POS tagging. For each positive caption, one hard-negative caption is produced by replacing eligible non-stopword nouns, verbs, and adjectives with words of the same POS collected from the current mini-batch. Each eligible token is replaced with probability 0.5 when a different same-POS candidate exists; if no token is replaced, word order is shuffled as a fallback. The replacement vocabulary is therefore batch-local rather than a fixed external lexicon.

## Training And Loss Details

Unless explicitly noted, training uses batch size 128 for all variants, including the multi-stream variants, without gradient accumulation. This is possible because B5d/B5e consume precomputed SAM encoder features instead of running an additional SAM ViT during the training forward pass. The default run length is 10 epochs with AdamW, cosine scheduling, and two warmup epochs. CLIP's pretrained logit scale is reset to `2.6593`, corresponding to temperature 0.07. For SigLIP, the learnable bias is initialized to `-10.0`; the implementation reuses the model logit-scale parameter rather than initializing it to `log(10)`.

## Hardware And Software

Training jobs request one GPU, 8 CPU cores per GPU, and 64 GB host memory per GPU under Slurm, using the container image `biremurhan/image-text-contrast:v0.16`. Local W&B metadata in the experiment logs identifies the GPU as NVIDIA A40 for the inspected runs. The training container is based on `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`; project dependencies include `transformers==4.48.0`, `numpy==1.26.4`, `PyYAML==6.0.2`, and `wandb==0.25.1`. The code uses HuggingFace Transformers CLIP (`openai/clip-vit-large-patch14-336`), not OpenCLIP, unless a separate untracked experiment changed this in the thesis text.

The seed setup fixes Python, NumPy, and PyTorch random seeds; calls `torch.cuda.manual_seed_all`; sets `torch.backends.cudnn.deterministic=True`; sets `torch.backends.cudnn.benchmark=False`; and sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`. DataLoader workers are seeded as `seed + worker_id`, and the training DataLoader uses a seeded PyTorch generator for shuffling. These settings improve reproducibility but do not guarantee bit-exact determinism for every CUDA kernel.

Training-time reporting should be filled from Slurm accounting or final W&B runtimes for the submitted jobs. The local completed logs I inspected are consistent with single-GPU runs taking roughly 30-36 hours on Flickr30K and roughly 5-6 days on COCO for several full 10-epoch B5/B0plus-style runs, but these numbers should be verified against the final job table before being stated in the thesis.

## Statistical Reporting

Reported means and standard deviations are computed across available random seeds for each run and dataset; single-seed entries are marked separately in the generated tables/figures. No paired significance test or bootstrap confidence interval is currently implemented in the retrieval reporting scripts. If the chapter claims statistical testing, add a paired bootstrap over test-set queries for R@K; otherwise state explicitly that the tables report seed mean and sample standard deviation only.
