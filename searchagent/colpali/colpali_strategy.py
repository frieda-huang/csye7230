from abc import ABC, abstractmethod
from typing import List


class RankingStrategy(ABC):
    @abstractmethod
    def execute_ranking_function(self, data: List):
        pass


class MaxSimStrategy(RankingStrategy):
    def execute_ranking_function(self, data: List):
        pass


class HammingDistanceStrategy(RankingStrategy):
    def execute_ranking_function(self, data: List):
        pass


class ColPaliContext:
    def __init__(self, strategy: RankingStrategy):
        self._strategy = strategy

    @property
    def strategy(self) -> RankingStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: RankingStrategy):
        self._strategy = strategy

    def execute_ranking_function(self, data: List):
        return self.strategy.execute_ranking_function(data)
