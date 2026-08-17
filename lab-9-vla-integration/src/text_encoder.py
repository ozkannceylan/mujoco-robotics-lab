"""Lab 9 — M2: frozen CLIP text tower and the instruction bank.

The design is `ozkannceylan/humanoid_vla`'s and is kept because it is right: a
frozen text encoder is used at *training* time to embed the instruction
vocabulary, and the resulting embeddings are stored **inside the checkpoint**.
Evaluation, the closed-loop runner and the capstone then need neither
`transformers` nor the network — they look the instruction up in the bank.

Why frozen, and why not something bigger
----------------------------------------
The policy has to map an instruction to a behaviour, not to understand English.
Four instructions in the vocabulary, three paraphrases each: a 512-d frozen
sentence embedding separates them with room to spare, and fine-tuning a language
model on twelve sentences would fit noise. The brief says so too — "Language
conditioning can be simple. Don't build a full LLM backbone; that's scope
creep."

Encoding a *novel* free-form instruction at inference time — which the capstone
does, to show the path does not depend on a task index — needs this module and
therefore `transformers`.
"""

from __future__ import annotations

import numpy as np

from lab9_common import all_instructions

__all__ = [
    "DEFAULT_TEXT_MODEL",
    "TEXT_EMBED_DIM",
    "TextEncoder",
    "build_instruction_bank",
    "InstructionBank",
]

DEFAULT_TEXT_MODEL = "openai/clip-vit-base-patch32"
TEXT_EMBED_DIM = 512


class TextEncoder:
    """Lazy wrapper around a frozen CLIP text tower (projected, L2-normalised)."""

    def __init__(self, model_name: str = DEFAULT_TEXT_MODEL, device: str = "cpu"):
        """Load the tower.

        Args:
            model_name: HuggingFace model id.
            device: Torch device.

        Raises:
            ImportError: If `transformers` is not installed.
        """
        try:
            from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError(
                "text conditioning needs `pip install transformers`. "
                "Evaluation does not: the instruction bank is baked into the "
                "checkpoint at training time."
            ) from error

        import torch

        self.device = device
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        self.model = CLIPTextModelWithProjection.from_pretrained(model_name).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self._torch = torch
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed instructions.

        Args:
            texts: Instruction strings.

        Returns:
            ``(N, 512)`` float32, L2-normalised.
        """
        missing = [text for text in texts if text not in self._cache]
        if missing:
            tokens = self.tokenizer(
                missing, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            with self._torch.no_grad():
                embeddings = self.model(**tokens).text_embeds
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            for text, embedding in zip(
                missing, embeddings.cpu().numpy().astype(np.float32), strict=True
            ):
                self._cache[text] = embedding
        return np.stack([self._cache[text] for text in texts])

    def encode_one(self, text: str) -> np.ndarray:
        """Embed a single instruction.

        Args:
            text: Instruction string.

        Returns:
            ``(512,)`` float32.
        """
        return self.encode([text])[0]


class InstructionBank:
    """Instruction embeddings carried inside a checkpoint.

    Args:
        embeddings: ``{instruction: (512,) float32}``.
    """

    def __init__(self, embeddings: dict[str, np.ndarray]):
        self.embeddings = {k: np.asarray(v, dtype=np.float32) for k, v in embeddings.items()}

    def __contains__(self, text: str) -> bool:
        return text in self.embeddings

    def __len__(self) -> int:
        return len(self.embeddings)

    def get(self, text: str) -> np.ndarray:
        """Look an instruction up, encoding it live if it is unknown.

        Args:
            text: Instruction string.

        Returns:
            ``(512,)`` float32.
        """
        if text in self.embeddings:
            return self.embeddings[text]
        encoder = TextEncoder()
        embedding = encoder.encode_one(text)
        self.embeddings[text] = embedding
        return embedding

    def batch(self, texts: list[str]) -> np.ndarray:
        """Look up several instructions.

        Args:
            texts: Instruction strings.

        Returns:
            ``(N, 512)`` float32.
        """
        return np.stack([self.get(text) for text in texts])

    def to_dict(self) -> dict:
        return {k: v.tolist() for k, v in self.embeddings.items()}

    @classmethod
    def from_dict(cls, payload: dict) -> "InstructionBank":
        return cls({k: np.asarray(v, dtype=np.float32) for k, v in payload.items()})

    def separation(self) -> dict:
        """How well the bank separates instructions that mean different things.

        A policy cannot follow instructions its conditioning cannot tell apart,
        so this is checked before training rather than inferred from a bad
        success rate afterwards.

        Returns:
            Cosine similarity summary: the closest pair, and the mean.
        """
        names = sorted(self.embeddings)
        matrix = np.stack([self.embeddings[n] for n in names])
        similarity = matrix @ matrix.T
        np.fill_diagonal(similarity, -np.inf)
        index = int(np.argmax(similarity))
        row, column = divmod(index, len(names))
        finite = similarity[np.isfinite(similarity)]
        return {
            "closest_pair": (names[row], names[column]),
            "closest_similarity": float(similarity[row, column]),
            "mean_similarity": float(finite.mean()),
            "count": len(names),
        }


def build_instruction_bank(
    texts: list[str] | None = None, device: str = "cpu"
) -> InstructionBank:
    """Encode the whole instruction vocabulary once.

    Args:
        texts: Instructions to encode; the full vocabulary (including held-out
            paraphrases) if omitted, so evaluation can test generalisation
            without needing the network.
        device: Torch device.

    Returns:
        A populated :class:`InstructionBank`.
    """
    texts = texts or all_instructions(train_only=False)
    encoder = TextEncoder(device=device)
    return InstructionBank(dict(zip(texts, encoder.encode(texts), strict=True)))


def _main() -> None:
    bank = build_instruction_bank()
    summary = bank.separation()
    print(f"instruction bank: {len(bank)} entries, dim {TEXT_EMBED_DIM}")
    print(f"mean cosine similarity   {summary['mean_similarity']:.3f}")
    print(f"closest pair             {summary['closest_pair']} "
          f"at {summary['closest_similarity']:.3f}")
    for text in sorted(bank.embeddings):
        print(f"  {text}")


if __name__ == "__main__":
    _main()
