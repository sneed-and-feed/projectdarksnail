import os
import sys
import time
import argparse
import ctypes
from typing import Any, Optional, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    Image = None
    ImageDraw = None
    HAS_PIL = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    requests = None
    HAS_REQUESTS = False

# Try to load HF_TOKEN from Google Colab secrets if available
try:
    from google.colab import userdata
    token = userdata.get('HF_TOKEN')
    if token:
        os.environ["HF_TOKEN"] = token
except Exception:
    pass

# ==============================================================================
# 1. Low-level Llama.cpp Ctypes Mocking / Loading
# ==============================================================================

HAS_LLAMA_CPP = False
try:
    import llama_cpp
    import llama_cpp.llama_cpp as ll_cpp
    HAS_LLAMA_CPP = True
except ImportError:
    pass

# Define mock types if llama_cpp is not available
if not HAS_LLAMA_CPP:
    # Creating a dummy namespace to prevent NameErrors
    class MockLlamaCpp:
        @staticmethod
        def llama_backend_init():
            print("[Mock C API] llama_backend_init()")

        @staticmethod
        def llama_backend_free():
            print("[Mock C API] llama_backend_free()")

        @staticmethod
        def llama_model_n_embd(model):
            return 3072  # Gemma-4-12B hidden size

        @staticmethod
        def llama_decode(ctx, batch):
            print(f"[Mock C API] llama_decode(ctx, batch) with n_tokens={batch.n_tokens}, is_embeddings={batch.embd is not None}")
            if batch.embd is not None:
                # Print sample embeddings shape/info
                print(f"  -> Decoded {batch.n_tokens} custom visual embeddings into KV cache.")
            else:
                sample_tokens = [batch.token[i] for i in range(min(5, batch.n_tokens))]
                print(f"  -> Decoded tokens: {sample_tokens} ... at positions: {[batch.pos[i] for i in range(min(5, batch.n_tokens))]}")
            return 0

        @staticmethod
        def llama_batch_init(n_tokens, embd, n_seq_max):
            class MockBatch:
                def __init__(self):
                    self.n_tokens = n_tokens
                    self.token = (ctypes.c_int32 * n_tokens)() if embd == 0 else None
                    self.embd = (ctypes.c_float * (n_tokens * embd))() if embd > 0 else None
                    self.pos = (ctypes.c_int32 * n_tokens)()
                    self.n_seq_id = (ctypes.c_int32 * n_tokens)()
                    self.logits = (ctypes.c_int8 * n_tokens)()
                    # seq_id is pointer to pointer. We represent it as pointer to array
                    self.seq_id = (ctypes.POINTER(ctypes.c_int32) * (n_tokens + 1))()
                    for i in range(n_tokens):
                        self.seq_id[i] = (ctypes.c_int32 * n_seq_max)()
                    self.seq_id[n_tokens] = None

            print(f"[Mock C API] llama_batch_init(n_tokens={n_tokens}, embd={embd}, n_seq_max={n_seq_max})")
            return MockBatch()

        @staticmethod
        def llama_batch_free(batch):
            print(f"[Mock C API] llama_batch_free()")

        @staticmethod
        def llama_get_logits_ith(ctx, i):
            # Return a pointer to a dummy vocab logits array of size 518000
            # We seed a mock response for GUI action generation (e.g. click at coordinates)
            vocab_size = 518000
            dummy_logits = (ctypes.c_float * vocab_size)()
            # Make some tokens highly probable
            # Let's mock a sequence of outputs like: 'CLICK', '(', '600', ',', '165', ')'
            # We'll just put high values at index 0 for simplicity, or random values
            for idx in range(vocab_size):
                dummy_logits[idx] = -10.0
            dummy_logits[100] = 10.0  # mock token
            return ctypes.pointer(dummy_logits)

    ll_cpp = MockLlamaCpp()

    class MockLlama:
        def __init__(self, model_path, vocab_only=False, **kwargs):
            self.model_path = model_path
            print(f"[Mock Llama] Loaded model {model_path} (vocab_only={vocab_only})")
            self.model = ctypes.c_void_p(12345)
            self.ctx = ctypes.c_void_p(67890)
            self._n_vocab = 518000

        def tokenize(self, text, add_bos=False, special=True):
            # Return some fake tokens
            text_str = text.decode("utf-8", errors="ignore")
            print(f"[Mock Tokenizer] Tokenize: '{text_str}'")
            if text == b"<|image|>":
                return [517766]
            elif text == b"<image|>":
                return [517770]
            elif text == b"<bos>":
                return [2]
            elif text == b"<eos>":
                return [1]
            elif text == b"<|turn>":
                return [106]
            elif text == b"<turn|>":
                return [216]
            # General fallback
            return [ord(c) + 1000 for c in text_str]

        def detokenize(self, tokens):
            # Return string back
            chars = []
            for t in tokens:
                if t == 517766:
                    chars.append("<|image|>")
                elif t == 517770:
                    chars.append("<image|>")
                elif t >= 1000:
                    chars.append(chr(t - 1000))
                else:
                    chars.append(f"[Tok {t}]")
            res_str = "".join(chars)
            return res_str.encode("utf-8")

        def n_vocab(self):
            return self._n_vocab

    llama_cpp = type('llama_cpp_dummy', (), {'Llama': MockLlama})()

# ==============================================================================
# 2. Vision Models (ViT and Vision Projection)
# ==============================================================================

class VisionProjection(nn.Module):
    def __init__(self, vit_dim: int = 768, llm_dim: int = 2048):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vit_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MultimodalEncoder:
    def __init__(self, device: str, llm_dim: int):
        self.device = device
        self.llm_dim = llm_dim
        
        # Load ViT
        self.load_pretrained = (device == "cuda")
        if self.load_pretrained:
            try:
                from transformers import ViTImageProcessor, ViTModel
                print("[Vision] Loading pretrained google/vit-base-patch16-224...")
                self.vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
                self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
                print("[Vision] Pretrained ViT loaded successfully.")
            except Exception as e:
                print(f"[Vision] Pretrained ViT load failed ({e}). Falling back to dummy.")
                self.load_pretrained = False

        if not self.load_pretrained:
            from transformers import ViTConfig, ViTImageProcessor, ViTModel
            print("[Vision] Using tiny randomly-initialized ViT for CPU/local mode.")
            vit_config = ViTConfig(
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=128
            )
            self.vit_processor = ViTImageProcessor()
            self.vit_model = ViTModel(vit_config)
            
        self.vit_model.to(self.device)
        self.vit_model.eval()
        for param in self.vit_model.parameters():
            param.requires_grad = False
            
        self.vit_dim = self.vit_model.config.hidden_size
        self.projection = VisionProjection(vit_dim=self.vit_dim, llm_dim=self.llm_dim).to(self.device)
        self.projection.eval()
        
    def extract_and_project(self, image: Any) -> np.ndarray:
        """Processes PIL image, extracts patch features, and projects to LLM hidden dim."""
        if not HAS_PIL:
            raise RuntimeError("Pillow is required for image extraction. Run `pip install pillow`.")
        inputs = self.vit_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.vit_model(**inputs)
            # Remove [CLS] token, leaving 196 patches
            features = outputs.last_hidden_state[:, 1:, :]  # (1, 196, vit_dim)
            projected = self.projection(features)           # (1, 196, llm_dim)
            
        return projected.squeeze(0).cpu().numpy()  # Return shape (196, llm_dim) as numpy float32 array

# ==============================================================================
# 3. Three-Stage Decoder Implementation
# ==============================================================================

def decode_tokens(ctx_ptr, token_ids, pos_start, seq_id=0, need_logits=False):
    """Decodes a list of token IDs at specified positions using llama_decode."""
    n_tokens = len(token_ids)
    if n_tokens == 0:
        return
        
    batch = ll_cpp.llama_batch_init(n_tokens, 0, 1)
    batch.n_tokens = n_tokens
    
    for i, tok in enumerate(token_ids):
        batch.token[i] = tok
        batch.pos[i] = pos_start + i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = seq_id
        # Enable logits ONLY for the last token in the batch if requested
        batch.logits[i] = 1 if (need_logits and i == n_tokens - 1) else 0
        
    res = ll_cpp.llama_decode(ctx_ptr, batch)
    ll_cpp.llama_batch_free(batch)
    
    if res != 0:
        raise RuntimeError(f"llama_decode failed with exit code {res}")


def decode_embeddings(ctx_ptr, embeddings, pos_start, seq_id=0):
    """Decodes a numpy array of custom embeddings at specified positions."""
    # embeddings is of shape (n_tokens, n_embd)
    n_tokens, n_embd = embeddings.shape
    embeddings_flat = embeddings.astype(np.float32, copy=False).flatten()
    
    batch = ll_cpp.llama_batch_init(n_tokens, n_embd, 1)
    batch.n_tokens = n_tokens
    
    # Copy floats directly from numpy buffer into batch.embd using ctypes memmove (zero copy)
    ctypes.memmove(batch.embd, embeddings_flat.ctypes.data, embeddings_flat.nbytes)
    
    for i in range(n_tokens):
        batch.pos[i] = pos_start + i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = seq_id
        batch.logits[i] = 0  # Embeddings don't need logits
        
    res = ll_cpp.llama_decode(ctx_ptr, batch)
    ll_cpp.llama_batch_free(batch)
    
    if res != 0:
        raise RuntimeError(f"llama_decode (embeddings) failed with exit code {res}")


def sample_token(logits, temperature=0.7):
    """Simple temperature-based sampling from logits."""
    # Softmax with temperature
    if temperature == 0:
        return int(np.argmax(logits))
    logits = np.array(logits, dtype=np.float64)
    probs = np.exp(logits / temperature)
    probs /= np.sum(probs)
    return int(np.random.choice(len(logits), p=probs))

def find_tokens_bulk(model, text_list):
    """Finds multiple token IDs in a single pass over the vocabulary."""
    results = {text: None for text in text_list}
    remaining = set(text_list)
    
    # Fast path: try tokenize first for each target
    for text in list(remaining):
        try:
            tokens = model.tokenize(text, add_bos=False, special=True)
            if len(tokens) == 1:
                results[text] = tokens[0]
                remaining.remove(text)
        except Exception:
            pass
            
    if not remaining:
        return results
        
    # Single-pass vocab scan fallback
    vocab_size = model.n_vocab()
    for token_id in range(vocab_size):
        try:
            piece = model.detokenize([token_id])
            if piece in remaining:
                results[piece] = token_id
                remaining.remove(piece)
                if not remaining:
                    break
        except Exception:
            continue
            
    return results


def scan_vocabulary(model):
    print("\n===== [Vocabulary Scan] =====")
    vocab_size = model.n_vocab()
    print(f"Total vocabulary size: {vocab_size}")
    
    # Test tokenization of common control strings
    test_strings = [
        b"<bos>", b"<eos>", 
        b"<|im_start|>", b"<|im_end|>", 
        b"<|turn>", b"<turn|>", 
        b"<|image|>", b"<image|>", 
        b"user", b"assistant", b"model", b"<start_of_turn>", b"<end_of_turn>"
    ]
    print("\nTokenization results for common strings:")
    for s in test_strings:
        try:
            tokens = model.tokenize(s, add_bos=False, special=True)
            detok = model.detokenize(tokens).decode("utf-8", errors="ignore")
            print(f"  Tokenizing {s.decode()} -> IDs: {tokens} -> Detok: '{detok}'")
        except Exception as e:
            print(f"  Tokenizing {s.decode()} -> Failed: {e}")
            
    # Scan vocabulary for tokens matching patterns
    patterns = ["im_start", "im_end", "turn", "user", "assistant", "model", "image", "bos", "eos"]
    print("\nScanning vocabulary for matches containing patterns:")
    matched = 0
    # To avoid printing too many, we limit to 150 matches
    for token_id in range(vocab_size):
        try:
            piece_bytes = model.detokenize([token_id])
            piece = piece_bytes.decode("utf-8", errors="ignore")
            if any(p in piece for p in patterns):
                print(f"  ID {token_id:6d}: '{piece}' (bytes: {piece_bytes})")
                matched += 1
                if matched >= 150:
                    print("  ... (limit of 150 matches reached)")
                    break
        except Exception:
            pass
    print("=============================\n")


# ==============================================================================
# 4. Main Execution Loop
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="GGUF Multimodal Injector (Path A)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to your Gemma-4 GGUF file")
    parser.add_argument("--image_url", type=str, default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png", help="URL or path to target image")
    parser.add_argument("--prompt", type=str, default="What is shown in this image?", help="Query prompt for the VLM")
    parser.add_argument("--mock", action="store_true", help="Force mock execution mode")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature for text sampling")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max output tokens to generate")
    parser.add_argument("--scan_vocab", action="store_true", help="Perform a full scan of vocabulary control tokens")
    
    args = parser.parse_args()
    
    # Force mock mode if requested or if llama-cpp-python is missing
    global HAS_LLAMA_CPP, llama_cpp, ll_cpp
    if args.mock:
        HAS_LLAMA_CPP = False
        print("[Injector] Mock mode forced by user.")
        
    device = "cuda" if (torch.cuda.is_available() and not args.mock) else "cpu"
    print(f"[Injector] Running on device: {device}")
    
    # 1. Download or locate GGUF model
    model_path = args.model_path
    if HAS_LLAMA_CPP:
        from huggingface_hub import hf_hub_download
        if not model_path:
            # Default to downloading Q6_K version to save bandwidth/disk
            filename = "adelic-gemma4-12b-Q6_K.gguf"
            print(f"[Injector] No model path specified. Downloading default {filename}...")
            try:
                model_path = hf_hub_download(
                    repo_id="sneedjak/Adelic-Gemma-4-12B-GGUF",
                    filename=filename
                )
                print(f"[Injector] Download complete: {model_path}")
            except Exception as e:
                print(f"[Injector] Download failed: {e}. Switching to mock mode.")
                HAS_LLAMA_CPP = False
                model_path = "mock_model.gguf"
        elif not os.path.exists(model_path):
            # The specified path does not exist locally; check if it's a filename we can pull from HF
            filename = os.path.basename(model_path)
            print(f"[Injector] Specified model path '{model_path}' not found locally. Attempting to download '{filename}' from Hugging Face...")
            try:
                model_path = hf_hub_download(
                    repo_id="sneedjak/Adelic-Gemma-4-12B-GGUF",
                    filename=filename
                )
                print(f"[Injector] Download complete: {model_path}")
            except Exception as e:
                print(f"[Injector] Download failed: {e}. Switching to mock mode.")
                HAS_LLAMA_CPP = False
                model_path = "mock_model.gguf"
    else:
        if not model_path:
            model_path = "mock_model.gguf"
        
    # 2. Initialize the Llama engine
    print(f"[Injector] Loading model into Llama engine...")
    if HAS_LLAMA_CPP:
        # Load high level class
        model = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=-1 if device == "cuda" else 0, # Offload all to GPU if available
            verbose=True
        )
        model_ptr = model.model
        ctx_ptr = model.ctx
        # Dynamic fallback for hidden size lookup to support different llama-cpp-python versions
        try:
            n_embd = ll_cpp.llama_model_n_embd(model_ptr)
        except AttributeError:
            try:
                n_embd = ll_cpp.llama_n_embd(model_ptr)
            except AttributeError:
                n_embd = getattr(model, "n_embd", lambda: 3072)()
        vocab_size = model.n_vocab()
    else:
        # Initialize mock model
        model = llama_cpp.Llama(model_path=model_path)
        model_ptr = model.model
        ctx_ptr = model.ctx
        try:
            n_embd = ll_cpp.llama_model_n_embd(model_ptr)
        except AttributeError:
            n_embd = 3072
        vocab_size = model.n_vocab()
        
    print(f"[Injector] Model hidden dimension (n_embd): {n_embd}")
    print(f"[Injector] Vocabulary size: {vocab_size}")
    
    if args.scan_vocab:
        scan_vocabulary(model)
    
    # 3. Initialize Vision Stack
    vision_stack = MultimodalEncoder(device=device, llm_dim=n_embd)
    
    # 4. Fetch image
    print(f"[Injector] Fetching image from {args.image_url}...")
    try:
        if args.image_url.startswith("http"):
            image = Image.open(requests.get(args.image_url, stream=True).raw).convert("RGB")
        else:
            image = Image.open(args.image_url).convert("RGB")
        print("[Injector] Image loaded successfully.")
    except Exception as e:
        print(f"[Injector] Image load failed ({e}). Creating synthetic mock image.")
        image = Image.new("RGB", (224, 224), color=(50, 100, 150))
        draw = ImageDraw.Draw(image)
        draw.text((10, 100), "Mock Screen Grab", fill=(255, 255, 255))
        
    # 5. Extract visual features and project
    print("[Injector] Running Vision Encoder and Projection Layer...")
    visual_embeddings = vision_stack.extract_and_project(image)  # Shape (196, n_embd)
    print(f"[Injector] Visual embeddings projected successfully. Shape: {visual_embeddings.shape}")
    
    # 6. Retrieve special tokens dynamically and select template
    print("[Injector] Detecting prompt format and special tokens from vocabulary...")
    
    # Define targets to lookup in vocabulary
    targets = [
        b"<|image|>", b"<image|>",
        b"<|im_start|>", b"<|im_end|>",
        b"<|turn>", b"<turn|>",
        b"<start_of_turn>", b"<end_of_turn>"
    ]
    
    # Bulk find token IDs
    resolved_tokens = find_tokens_bulk(model, targets)
    
    image_token_id = resolved_tokens[b"<|image|>"]
    eoi_token_id = resolved_tokens[b"<image|>"]
    
    if image_token_id is None:
        print("  -> Warning: <|image|> token not found in vocab. Defaulting to 517766.")
        image_token_id = 517766
    if eoi_token_id is None:
        print("  -> Warning: <image|> EOI token not found in vocab. Defaulting to 517770.")
        eoi_token_id = 517770
        
    print(f"  -> Resolved <|image|> ID: {image_token_id}")
    print(f"  -> Resolved <image|> EOI ID: {eoi_token_id}")
    
    im_start_id = resolved_tokens[b"<|im_start|>"]
    im_end_id = resolved_tokens[b"<|im_end|>"]
    turn_user_id = resolved_tokens[b"<|turn>"]
    turn_end_id = resolved_tokens[b"<turn|>"]
    gemma2_start_id = resolved_tokens[b"<start_of_turn>"]
    gemma2_end_id = resolved_tokens[b"<end_of_turn>"]
    
    print(f"  -> Template tokens found:")
    print(f"     * <|im_start|>: {im_start_id}")
    print(f"     * <|im_end|>: {im_end_id}")
    print(f"     * <|turn>: {turn_user_id}")
    print(f"     * <turn|>: {turn_end_id}")
    print(f"     * <start_of_turn>: {gemma2_start_id}")
    print(f"     * <end_of_turn>: {gemma2_end_id}")

    # 7. Tokenize Prompt Segments using manual injection for safety
    try:
        bos_token_id = model.tokenize(b"", add_bos=True)[0]
    except Exception:
        try:
            bos_token_id = find_tokens_bulk(model, [b"<bos>"])[b"<bos>"] or 2
        except Exception:
            bos_token_id = 2

    # Choose prompt structure based on detected tokens
    if im_start_id is not None and im_end_id is not None:
        print("[Injector] Selected prompt template: ChatML")
        # Prefix: <bos><|im_start|>user\n
        user_text_ids = model.tokenize(b"user\n", add_bos=False, special=True)
        prefix_ids = [bos_token_id, im_start_id] + user_text_ids
        
        # Suffix: \n{prompt}<|im_end|>\n<|im_start|>assistant\n
        prompt_ids = model.tokenize(f"\n{args.prompt}".encode("utf-8"), add_bos=False, special=True)
        nl_ids = model.tokenize(b"\n", add_bos=False, special=True)
        assistant_ids = model.tokenize(b"assistant\n", add_bos=False, special=True)
        suffix_ids = prompt_ids + [im_end_id] + nl_ids + [im_start_id] + assistant_ids
        
        eot_token_id = im_end_id
        
    elif turn_user_id is not None and turn_end_id is not None:
        print("[Injector] Selected prompt template: Gemma 4 (<|turn>)")
        # Prefix: <bos><|turn>user\n
        user_text_ids = model.tokenize(b"user\n", add_bos=False, special=True)
        prefix_ids = [bos_token_id, turn_user_id] + user_text_ids
        
        # Suffix: \n{prompt}<turn|>\n<|turn>model\n
        prompt_ids = model.tokenize(f"\n{args.prompt}".encode("utf-8"), add_bos=False, special=True)
        nl_ids = model.tokenize(b"\n", add_bos=False, special=True)
        model_text_ids = model.tokenize(b"model\n", add_bos=False, special=True)
        suffix_ids = prompt_ids + [turn_end_id] + nl_ids + [turn_user_id] + model_text_ids
        
        eot_token_id = turn_end_id
        
    elif gemma2_start_id is not None and gemma2_end_id is not None:
        print("[Injector] Selected prompt template: Gemma 1/2 (<start_of_turn>)")
        # Prefix: <bos><start_of_turn>user\n
        user_text_ids = model.tokenize(b"user\n", add_bos=False, special=True)
        prefix_ids = [bos_token_id, gemma2_start_id] + user_text_ids
        
        # Suffix: \n{prompt}<end_of_turn>\n<start_of_turn>model\n
        prompt_ids = model.tokenize(f"\n{args.prompt}".encode("utf-8"), add_bos=False, special=True)
        nl_ids = model.tokenize(b"\n", add_bos=False, special=True)
        model_text_ids = model.tokenize(b"model\n", add_bos=False, special=True)
        suffix_ids = prompt_ids + [gemma2_end_id] + nl_ids + [gemma2_start_id] + model_text_ids
        
        eot_token_id = gemma2_end_id
        
    else:
        # Default fallback to ChatML if not sure
        print("[Injector] Warning: No standard template matching. Defaulting to ChatML with fallback IDs.")
        fallback_im_start = im_start_id if im_start_id is not None else 106
        fallback_im_end = im_end_id if im_end_id is not None else 216
        
        user_text_ids = model.tokenize(b"user\n", add_bos=False, special=True)
        prefix_ids = [bos_token_id, fallback_im_start] + user_text_ids
        
        prompt_ids = model.tokenize(f"\n{args.prompt}".encode("utf-8"), add_bos=False, special=True)
        nl_ids = model.tokenize(b"\n", add_bos=False, special=True)
        assistant_ids = model.tokenize(b"assistant\n", add_bos=False, special=True)
        suffix_ids = prompt_ids + [fallback_im_end] + nl_ids + [fallback_im_start] + assistant_ids
        
        eot_token_id = fallback_im_end

    print(f"  -> Generated prefix IDs: {prefix_ids}")
    print(f"  -> Detokenized prefix: '{model.detokenize(prefix_ids).decode('utf-8', errors='ignore')}'")
    print(f"  -> Generated suffix IDs: {suffix_ids}")
    print(f"  -> Detokenized suffix: '{model.detokenize(suffix_ids).decode('utf-8', errors='ignore')}'")
    
    # 8. Run Three-Stage Decoding
    print("\n===== [Stage 1: Decoding Prefix Text] =====")
    pos = 0
    decode_tokens(ctx_ptr, prefix_ids, pos_start=pos, seq_id=0, need_logits=False)
    pos += len(prefix_ids)
    print(f"  -> Prefix tokens decoded. Next position: {pos}")
    
    print("\n===== [Stage 2: Injecting Image Embeddings] =====")
    # Inject 196 visual embeddings at consecutive positions
    decode_embeddings(ctx_ptr, visual_embeddings, pos_start=pos, seq_id=0)
    pos += visual_embeddings.shape[0]
    
    # Decode EOI token immediately after the image embeddings to close the block
    decode_tokens(ctx_ptr, [eoi_token_id], pos_start=pos, seq_id=0, need_logits=False)
    pos += 1
    print(f"  -> Visual embeddings injected. Next position: {pos}")
    
    print("\n===== [Stage 3: Decoding Suffix Text] =====")
    # Decode suffix text, enabling logits for the final token so we can generate from it
    decode_tokens(ctx_ptr, suffix_ids, pos_start=pos, seq_id=0, need_logits=True)
    pos += len(suffix_ids)
    print(f"  -> Suffix tokens decoded. Next position: {pos}")
    
    # 9. Autoregressive Text Generation Loop
    print("\n===== [Generation Started] =====")
    generated_tokens = []
    
    try:
        eos_token_id = model.tokenize(b"<eos>", add_bos=False, special=True)[0]
    except Exception:
        eos_token_id = 1
        
    for step in range(args.max_tokens):
        # Safe logits cast in mock mode
        if not HAS_LLAMA_CPP:
            # Mock generating a sample action response
            mock_action_seq = ["[Mock] Action: CLICK on Search Bar at (600, 165).", "\nReason: Found search query target."]
            if step < len(mock_action_seq):
                text = mock_action_seq[step]
            else:
                text = ""
                break
            print(text, end="", flush=True)
            time.sleep(0.3)
            continue
            
        # Retrieve logits pointer with dynamic version fallback
        try:
            # Pass -1 to retrieve the logits of the last output row in the batch
            logits_ptr = ll_cpp.llama_get_logits_ith(ctx_ptr, -1)
        except AttributeError:
            logits_ptr = ll_cpp.llama_get_logits(ctx_ptr)
            
        # Explicitly cast to POINTER(c_float) to prevent type errors if returned as c_void_p
        typed_logits_ptr = ctypes.cast(logits_ptr, ctypes.POINTER(ctypes.c_float))
        
        # Convert logits pointer to numpy array
        logits = np.ctypeslib.as_array(typed_logits_ptr, shape=(vocab_size,))
        
        # Sample next token
        next_token = sample_token(logits, temperature=args.temp)
        
        if next_token == eos_token_id or next_token == eot_token_id:
            print(" [EOS/EOT detected]", flush=True)
            break
            
        generated_tokens.append(next_token)
        
        # Decode and print generated text
        token_text = model.detokenize([next_token]).decode("utf-8", errors="ignore")
        print(token_text, end="", flush=True)
        
        # Decode next token to advance context for next step
        decode_tokens(ctx_ptr, [next_token], pos_start=pos, seq_id=0, need_logits=True)
        pos += 1
        
    print("\n===== [Generation Completed] =====\n")
    
    # Cleanup memory
    if HAS_LLAMA_CPP:
        # High-level model class in llama-cpp-python is cleaned up by garbage collection or manually
        del model
        
    print("[Injector] Completed successfully.")


if __name__ == "__main__":
    main()
