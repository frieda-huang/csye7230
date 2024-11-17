from searchagent.colpali.search_engine.indexing_strategies import (
    HNSWIndexingBinaryQuantizationHammingDistance,
)
from searchagent.colpali.search_engine.search_strategies import (
    ANNHNSWHammingSearchStrategy,
    ANNIVFFlatEuclideanSearchStrategy,
    ExactMaxSimSearchStrategy,
)
from searchagent.colpali.search_engine.strategy_factory import (
    SearchStrategyFactory,
    IndexingStrategyFactory,
)

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

IndexingStrategyFactory.register_strategy(
    "HNSW", HNSWIndexingBinaryQuantizationHammingDistance
)
