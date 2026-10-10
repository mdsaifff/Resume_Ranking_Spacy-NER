import numpy as np


class FakeEmbed:
    """Identical strings -> cosine 1, different -> 0; strings in the same synonym group -> 1."""

    def __init__(self, *synonyms):
        self.ids = {s: i for i, group in enumerate(synonyms) for s in group}
        self.next = len(synonyms)

    def __call__(self, texts):
        vecs = np.zeros((len(texts), 512))
        for row, t in enumerate(texts):
            if t not in self.ids:
                self.ids[t], self.next = self.next, self.next + 1
            vecs[row, self.ids[t]] = 1.0
        return vecs
