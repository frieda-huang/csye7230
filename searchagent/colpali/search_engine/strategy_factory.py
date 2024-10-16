from searchagent.colpali.search_engine.search_strategies import SearchStrategy


class SearchStrategyFactory:
    _strategies = {}

    @classmethod
    def register_strategy(cls, name: str, strategy_cls: type):
        cls._strategies[name] = strategy_cls

    @classmethod
    def create_search_strategy(cls, name: str) -> SearchStrategy:
        strategy_cls = cls._strategies.get(name)
        if not strategy_cls:
            raise ValueError(f"Unsupported search strategy: {name}")
        return strategy_cls()
