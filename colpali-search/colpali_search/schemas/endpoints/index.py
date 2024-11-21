import datetime

from colpali_search.types import CustomBaseModel


class ConfigureIndexResponse(CustomBaseModel):
    id: int
    strategy_name: str
    created_at: datetime.datetime


class ResetIndexResponse(CustomBaseModel):
    status: str
    message: str
    reset_time: datetime.datetime
