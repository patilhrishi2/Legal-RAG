# src/embedder.py  (new)
import torch

class QwenEmbedder:
    """Local Qwen3-Embedding-0.6B. Documents and queries are embedded differently."""
    name = "qwen3-0.6b"
    dim  = 1024

    def __init__(self, batch_size: int = 8, max_seq_length: int = 768):
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(
                "Qwen/Qwen3-Embedding-0.6B", device="cuda",
                model_kwargs={"torch_dtype": torch.float16})
            m.max_seq_length = self.max_seq_length
            self._model = m
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._load().encode(texts, batch_size=self.batch_size,
                                   normalize_embeddings=True)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self._load().encode([text], prompt_name="query",
                                  normalize_embeddings=True)
        return vec[0].tolist()


if __name__ == "__main__":
    e = QwenEmbedder()
    d = e.embed_documents(["Preventive detention under Article 22.", "Contract of sale of jute."])
    q = e.embed_query("What are the limits on preventive detention?")
    print(len(d), len(d[0]), len(q))
    print("sim to doc0:", sum(a*b for a, b in zip(q, d[0])),
          " doc1:", sum(a*b for a, b in zip(q, d[1])))