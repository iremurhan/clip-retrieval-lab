# Implementation Summary: Issue #30 (Text-Guided Cross-Attention Fusion)

## 1. Changelog by File
* **`src/model.py`**: Added `CrossAttentionFusion` module (Q=text pooler output, K=V=ViT image patches) and updated `encode_image` to accept optional `input_ids`/`attention_mask`, returning a `(image_embeds, attn_probs)` tuple from both text-guided and standard pooling paths.
* **`src/grad_cache.py`**: Both the no-grad cache phase and the gradient recomputation phase now forward `txt_input_chunk` and `txt_mask_chunk` to `encode_image`, enabling text-guided fusion throughout gradient caching.
* **`src/train.py`**: Added `_log_attention_maps(epoch)` method to `Trainer`, which runs the first 15 validation samples through the model, extracts `attn_probs` from `encode_image`, and logs the visualizations as `wandb.Image` objects under `"val/attention_maps"`.
* **`src/utils.py`**: Added `visualize_text_guided_attention(image, attn_probs, caption)` which denormalizes a CLIP image tensor, upsamples the patch attention grid via `scipy.ndimage.zoom`, and returns a side-by-side matplotlib figure of the original image and its attention heatmap.

## 2. Verification
* **Attention Math**: `CrossAttentionFusion.forward` computes `scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale` followed by `F.softmax(scores, dim=-1)` and `torch.matmul(attn_probs_full, V)` — exact `softmax(Q*K^T / sqrt(d)) * V` formulation, implemented explicitly without `nn.MultiheadAttention`.
* **Memory Management**: `_log_attention_maps` calls `plt.close(fig)` immediately after each figure is serialized to a `BytesIO` buffer, preventing matplotlib figure accumulation across the 15 logged samples.
