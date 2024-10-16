from abc import ABC, abstractmethod


class IndexingStrategy(ABC):
    @abstractmethod
    def build_index():
        pass

    @abstractmethod
    def query_index():
        pass


class HNSWIndexing(IndexingStrategy):
    def build_index():
        pass

    def query_index():
        pass


class IVFFlatIndexing(IndexingStrategy):
    def build_index():
        pass

    def query_index():
        pass
