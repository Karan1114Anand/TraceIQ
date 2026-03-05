"""
app/ingestion/embedder.py

Local sentence-transformers embedder (no HuggingFace Inference API).
Supports local model path to avoid download failures on Windows/SSL issues.
"""

from __future__ import annotations

from typing import List, Optional
from loguru import logger

from app.config.settings import HF_EMBEDDING_MODEL, HF_LOCAL_MODEL_PATH


class HuggingFaceEmbedder:
    """
    Generates embeddings locally using sentence-transformers.
    No API calls — everything runs on-device.

    Windows SSL / offline fix:
        Set HF_LOCAL_MODEL_PATH env var to a folder where the model is cached,
        e.g. C:/models/all-MiniLM-L6-v2
    """

    def __init__(
        self,
        model_name: str = HF_EMBEDDING_MODEL,
        local_model_path: Optional[str] = HF_LOCAL_MODEL_PATH,
    ) -> None:
        self.model_name = model_name
        self.local_model_path = local_model_path
        self._model = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the sentence-transformers model, preferring local path."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed.\n"
                "Fix: pip install sentence-transformers"
            )

        # Try local path first (avoids SSL / download issues on Windows)
        load_target = self.local_model_path or self.model_name
        try:
            logger.info(f"Loading embedding model from: {load_target}")
            self._model = SentenceTransformer(load_target)
            logger.success(f"Embedding model loaded: {load_target}")
        except (OSError, ValueError) as exc:
            if self.local_model_path:
                # local path failed — try downloading as fallback
                logger.warning(
                    f"Local model path failed ({exc}). Attempting download of "
                    f"'{self.model_name}' — this requires internet access."
                )
                try:
                    self._model = SentenceTransformer(self.model_name)
                    logger.success(f"Downloaded model: {self.model_name}")
                except Exception as dl_exc:
                    raise RuntimeError(
                        f"Embedding model could not be loaded.\n"
                        f"  Tried local path : {self.local_model_path}\n"
                        f"  Tried download   : {self.model_name}\n\n"
                        f"Remediation:\n"
                        f"  1. Run on a machine with internet access:\n"
                        f"       python -c \"from sentence_transformers import "
                        f"SentenceTransformer; "
                        f"SentenceTransformer('{self.model_name}').save('./models/{self.model_name.split('/')[-1]}')\"\n"
                        f"  2. Copy the saved folder to this machine.\n"
                        f"  3. Set env var: HF_LOCAL_MODEL_PATH=./models/{self.model_name.split('/')[-1]}\n"
                        f"\n  Original error: {dl_exc}"
                    ) from dl_exc
            else:
                raise RuntimeError(
                    f"Embedding model '{self.model_name}' failed to load.\n\n"
                    f"Possible causes on Windows:\n"
                    f"  - SSL certificate error (corporate network / proxy)\n"
                    f"  - No internet access\n\n"
                    f"Remediation:\n"
                    f"  1. Download the model on another machine and set:\n"
                    f"     HF_LOCAL_MODEL_PATH=/path/to/model\n"
                    f"  2. Or fix SSL: pip install certifi; set REQUESTS_CA_BUNDLE\n"
                    f"\n  Original error: {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> Optional[List[float]]:
        """Embed a single text string. Returns None on failure."""
        if not text or not text.strip():
            return None
        try:
            vector = self._model.encode(text, convert_to_numpy=True)
            return vector.tolist()
        except Exception as exc:
            logger.error(f"embed_text failed: {exc}")
            return None

    def embed_batch(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> List[Optional[List[float]]]:
        """Embed a list of texts. Failed items return None."""
        if not texts:
            return []
        try:
            vectors = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
            )
            return [v.tolist() for v in vectors]
        except Exception as exc:
            logger.error(f"embed_batch failed: {exc}. Falling back to one-by-one.")
            return [self.embed_text(t) for t in texts]

    def get_embedding_dimension(self) -> Optional[int]:
        """Return the dimension of embeddings produced by this model."""
        sample = self.embed_text("dimension probe")
        return len(sample) if sample else None
