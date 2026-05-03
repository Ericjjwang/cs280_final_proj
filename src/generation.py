"""
Zero123++ v1.2 novel-view synthesis wrapper.

Loads the diffusers pipeline lazily; use as a context manager to guarantee
GPU memory is freed after inference.
"""

from __future__ import annotations
import gc
from typing import Optional
import torch
from PIL import Image

_TILE = 320    # pixels per output tile (square)
_COLS = 2
_ROWS = 3


class Zero123PlusPipeline:
    """
    Wrapper around sudo-ai/zero123plus-v1.2 diffusers pipeline.

    Context manager (preferred):
        with Zero123PlusPipeline() as gen:
            views = gen.generate(cond_pil)

    Manual:
        gen = Zero123PlusPipeline().load()
        views = gen.generate(cond_pil)
        gen.unload()
    """

    def __init__(
        self,
        model_id: str = "sudo-ai/zero123plus-v1.2",
        device: str = "cuda",
        num_inference_steps: int = 36,
    ) -> None:
        self.model_id            = model_id
        self.device              = device
        self.num_inference_steps = num_inference_steps
        self._pipe               = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> "Zero123PlusPipeline":
        """Load pipeline onto device. Returns self for chaining."""
        if self._pipe is not None:
            return self
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
        pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
        pipe.to(self.device)
        pipe.enable_attention_slicing()
        self._pipe = pipe
        return self

    def unload(self) -> None:
        """Delete pipeline and free GPU memory."""
        self._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "Zero123PlusPipeline":
        return self.load()

    def __exit__(self, *_) -> bool:
        self.unload()
        return False   # do not suppress exceptions

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate(
        self,
        cond_image: Image.Image,
        num_inference_steps: Optional[int] = None,
    ) -> list[Image.Image]:
        """
        Generate 6 novel views from a condition image.

        Returns:
            List of 6 PIL Images (320×320, RGB) in grid reading order
            (top-left → bottom-right, same as camera.AZIMUTHS_DEG indices 0–5).
        """
        if self._pipe is None:
            raise RuntimeError("Pipeline not loaded — call .load() first.")
        steps = num_inference_steps or self.num_inference_steps
        grid  = self._pipe(cond_image, num_inference_steps=steps).images[0]
        return self._split_grid(grid)

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _split_grid(grid: Image.Image) -> list[Image.Image]:
        """Split 640×960 output grid into 6 tiles of 320×320 (row-major)."""
        tiles = []
        for row in range(_ROWS):
            for col in range(_COLS):
                x0, y0 = col * _TILE, row * _TILE
                tiles.append(grid.crop((x0, y0, x0 + _TILE, y0 + _TILE)))
        return tiles
