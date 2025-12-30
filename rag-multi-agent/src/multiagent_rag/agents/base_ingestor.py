from abc import ABC, abstractmethod


class BaseIngestor(ABC):
    @abstractmethod
    def process(self, file_path: str) -> list:
        pass
