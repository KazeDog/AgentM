from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_IMAGEMOL_ENCODER_CACHE: Dict[Tuple[str, int], "ImageMolEncoder"] = {}


class ImageMolEncoder:
    encoder_name = "imagemol_interface"

    def __init__(self, checkpoint_path: Optional[str] = None, embedding_dim: int = 128):
        self.checkpoint_path = checkpoint_path
        self.embedding_dim = embedding_dim

    def encode(self, image_path: str, metadata: Optional[Dict[str, float]] = None) -> List[float]:
        raise NotImplementedError

    def grad_cam(self, image_path: str, output_path: Optional[str] = None) -> Optional[str]:
        return None


class MockImageMolEncoder(ImageMolEncoder):
    """Deterministic fallback encoder used when ImageMol weights are absent."""

    encoder_name = "mock_imagemol_hash_encoder"

    def encode(self, image_path: str, metadata: Optional[Dict[str, float]] = None) -> List[float]:
        path = Path(image_path)
        payload = path.read_bytes() if path.exists() else str(image_path).encode("utf-8")
        if metadata:
            payload += repr(sorted(metadata.items())).encode("utf-8")
        seed_bytes = hashlib.sha256(payload).digest()[:8]
        seed = int.from_bytes(seed_bytes, byteorder="big", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.normal(loc=0.0, scale=1.0, size=self.embedding_dim)
        norm = float(np.linalg.norm(vec)) or 1.0
        return (vec / norm).astype(float).tolist()


class TorchImageMolEncoder(ImageMolEncoder):
    """ImageMol ResNet18 encoder compatible with ChengF-Lab/ImageMol checkpoints.

    The public ImageMol checkpoint stores an `embedding_layer.*` state dict
    whose module layout matches `torchvision.models.resnet18(...).children()[:-1]`.
    We load that backbone and return the 512-dimensional image embedding.
    """

    encoder_name = "chengf_lab_imagemol_resnet18"

    def __init__(self, checkpoint_path: str, embedding_dim: int = 512):
        resolved = _resolve_checkpoint_path(checkpoint_path)
        super().__init__(checkpoint_path=str(resolved), embedding_dim=embedding_dim)
        self.device, self.model = self._load_model(resolved)
        self.model.eval()

    def encode(self, image_path: str, metadata: Optional[Dict[str, float]] = None) -> List[float]:
        import torch

        tensor, _ = self._load_image_tensor(image_path)
        with torch.no_grad():
            embedding = self.model(tensor.to(self.device)).view(1, -1).squeeze(0)
        embedding = embedding.detach().cpu().float().numpy()
        norm = float(np.linalg.norm(embedding)) or 1.0
        return (embedding / norm).astype(float).tolist()

    def grad_cam(self, image_path: str, output_path: Optional[str] = None) -> Optional[str]:
        try:
            import torch
            import torch.nn.functional as F
            from PIL import Image
        except Exception:
            return None

        tensor, image_rgb = self._load_image_tensor(image_path)
        tensor = tensor.to(self.device)

        target_layer = self.model[7][-1]
        activations = {}
        gradients = {}

        def forward_hook(_module, _inputs, output):
            activations["value"] = output

        def backward_hook(_module, _grad_input, grad_output):
            gradients["value"] = grad_output[0]

        handle_f = target_layer.register_forward_hook(forward_hook)
        handle_b = target_layer.register_full_backward_hook(backward_hook)
        try:
            self.model.zero_grad(set_to_none=True)
            embedding = self.model(tensor).view(1, -1)
            score = embedding.norm(dim=1).sum()
            score.backward()
            acts = activations["value"].detach()
            grads = gradients["value"].detach()
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=image_rgb.size[::-1], mode="bilinear", align_corners=False)
            cam_np = cam.squeeze().cpu().numpy()
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
            heatmap = _heatmap_rgba(cam_np)
            overlay = Image.alpha_composite(image_rgb.convert("RGBA"), heatmap)
            path = Path(output_path) if output_path else Path(image_path).with_name(Path(image_path).stem + "_imagemol_gradcam.png")
            path.parent.mkdir(parents=True, exist_ok=True)
            overlay.save(path)
            return str(path)
        except Exception:
            return None
        finally:
            handle_f.remove()
            handle_b.remove()

    def _load_model(self, checkpoint_path: Path):
        import torch
        import torchvision

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        backbone = torch.nn.Sequential(*list(torchvision.models.resnet18(weights=None).children())[:-1])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        embedding_state = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("embedding_layer."):
                embedding_state[key[len("embedding_layer.") :]] = value
        missing, unexpected = backbone.load_state_dict(embedding_state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected ImageMol checkpoint keys: {unexpected[:5]}")
        critical_missing = [key for key in missing if not key.endswith("num_batches_tracked")]
        if critical_missing:
            raise RuntimeError(f"Missing ImageMol backbone keys: {critical_missing[:5]}")
        backbone.to(device)
        return device, backbone

    def _load_image_tensor(self, image_path: str):
        from PIL import Image
        from torchvision import transforms

        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transform(image).unsqueeze(0), image.resize((224, 224))


def load_imagemol_encoder(checkpoint_path: Optional[str] = None, embedding_dim: int = 128) -> ImageMolEncoder:
    cache_key = (str(checkpoint_path or "mock"), int(embedding_dim))
    if cache_key in _IMAGEMOL_ENCODER_CACHE:
        return _IMAGEMOL_ENCODER_CACHE[cache_key]
    if checkpoint_path:
        resolved = _resolve_checkpoint_path(checkpoint_path)
        if resolved.exists():
            encoder = TorchImageMolEncoder(checkpoint_path=str(resolved), embedding_dim=512)
            _IMAGEMOL_ENCODER_CACHE[cache_key] = encoder
            return encoder
    encoder = MockImageMolEncoder(checkpoint_path=checkpoint_path, embedding_dim=embedding_dim)
    _IMAGEMOL_ENCODER_CACHE[cache_key] = encoder
    return encoder


def _resolve_checkpoint_path(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path).expanduser()
    if path.exists():
        return path.resolve()
    project_root = Path(__file__).resolve().parents[2]
    candidate = project_root / path
    if candidate.exists():
        return candidate.resolve()
    return path


def _heatmap_rgba(cam: np.ndarray):
    from PIL import Image

    cam = np.clip(cam, 0.0, 1.0)
    red = (255 * cam).astype(np.uint8)
    green = (80 * (1.0 - np.abs(cam - 0.5) * 2.0)).astype(np.uint8)
    blue = (255 * (1.0 - cam)).astype(np.uint8)
    alpha = (120 * cam).astype(np.uint8)
    rgba = np.stack([red, green, blue, alpha], axis=-1)
    return Image.fromarray(rgba, mode="RGBA")
