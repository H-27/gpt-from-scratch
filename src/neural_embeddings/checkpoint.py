import heapq
import os
from dataclasses import dataclass, field
from typing import List

# Simple top-k checkpoint manager (logic placeholder)


@dataclass(order=True)
class _Entry:
    score: float
    path: str = field(compare=False)
    epoch: int = field(compare=False)


class TopKCheckpoints:
    def __init__(self, k: int, directory: str):
        self.k = k
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self._heap: List[_Entry] = []  # min-heap by score (e.g., perplexity)

    def consider(self, score: float, epoch: int) -> bool:
        """Return True if should save.
        Lower score assumed better (e.g., ppl)."""
        if len(self._heap) < self.k:
            return True
        if self._heap and score < self._heap[0].score:
            return True
        return False

    def add(self, score: float, epoch: int, filename: str):
        path = os.path.join(self.directory, filename)
        heapq.heappush(self._heap, _Entry(score=score, path=path, epoch=epoch))
        if len(self._heap) > self.k:
            removed = heapq.heappop(self._heap)
            try:
                if os.path.exists(removed.path):
                    os.remove(removed.path)
            except OSError:
                pass
        return path

    def current(self):
        return sorted(self._heap)
