from app.config.logger import fed_logger
from app.entity.aggregators.base_aggregator import BaseAggregator
from app.dto.base_model import BaseModel


class FedAvg(BaseAggregator):
    def __init__(self, total_data_size: int):
        self.total_data_size = total_data_size

    def aggregate(self, base_model: BaseModel, gathered_models: list[BaseModel]) -> BaseModel:
        keys = gathered_models[0][0].keys()
        for k in keys:
            for w in gathered_models:
                beta = float(w[1])
                if 'num_batches_tracked' in k:
                    base_model[k] = w[0][k]
                else:
                    base_model[k] += (w[0][k] * beta)

        return base_model
