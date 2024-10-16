from searchagent.colpali.search_engine.search_strategies import (
    ANNHNSWHammingSearchStrategy,
    ANNIVFFlatEuclideanSearchStrategy,
    ExactMaxSimSearchStrategy,
)
from searchagent.colpali.search_engine.strategy_factory import SearchStrategyFactory

SearchStrategyFactory.register_strategy(
    "ExactMaxSim",
    ExactMaxSimSearchStrategy,
)

SearchStrategyFactory.register_strategy(
    "ANNHNSWHamming",
    ANNHNSWHammingSearchStrategy,
)

SearchStrategyFactory.register_strategy(
    "ANNIVFFlatEuclidean",
    ANNIVFFlatEuclideanSearchStrategy,
)
