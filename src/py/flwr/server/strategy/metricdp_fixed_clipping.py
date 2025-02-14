"""Metric differential privacy with fixed clipping.

Preprint: https://arxiv.org/abs/2502.01352
"""

import numpy as np
from logging import INFO, WARNING
from typing import Optional, Union
from scipy.spatial import distance

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.differential_privacy import (
    add_gaussian_noise_to_params,
    compute_clip_model_update,
    compute_stdv,
)
from flwr.common.differential_privacy_constants import (
    CLIENTS_DISCREPANCY_WARNING,
    KEY_CLIPPING_NORM,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy


class MetricDifferentialPrivacyServerSideFixedClipping(Strategy):
    """Strategy wrapper for metric privacy with server-side fixed clipping.

    Parameters
    ----------
    strategy : Strategy
        The strategy to which the metric privacy method will be added by this wrapper.
    noise_multiplier : float
        The noise multiplier for the Gaussian mechanism for model updates (divided by the distance).
        A value of 1.0 or higher is recommended for strong privacy.
    clipping_norm : float
        The value of the clipping norm.
    num_sampled_clients : int
        The number of clients that are sampled on each round.

    Examples
    --------
    Create a strategy:

    >>> strategy = fl.server.strategy.FedAvg( ... )

    Wrap the strategy with the MetricDifferentialPrivacyServerSideFixedClipping wrapper

    >>> dp_strategy = MetricDifferentialPrivacyServerSideFixedClipping(
    >>>     strategy, cfg.noise_multiplier, cfg.clipping_norm, cfg.num_sampled_clients
    >>> )
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        strategy: Strategy,
        noise_multiplier: float,
        clipping_norm: float,
        num_sampled_clients: int,
    ) -> None:
        super().__init__()

        self.strategy = strategy

        if noise_multiplier < 0:
            raise ValueError("The noise multiplier should be a non-negative value.")

        if clipping_norm <= 0:
            raise ValueError("The clipping norm should be a positive value.")

        if num_sampled_clients <= 0:
            raise ValueError(
                "The number of sampled clients should be a positive value."
            )

        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm
        self.num_sampled_clients = num_sampled_clients

        self.current_round_params: NDArrays = []

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = "Metric Privacy Strategy Wrapper (Server-Side Fixed Clipping)"
        return rep

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters using given strategy."""
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        self.current_round_params = parameters_to_ndarrays(parameters)
        return self.strategy.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        return self.strategy.configure_evaluate(
            server_round, parameters, client_manager
        )

    def distance_metric(
        self,
        results: list[tuple[ClientProxy, FitRes]],
        ) -> float:

        parameters = []
        for _, res in results:
            parameters.append(parameters_to_ndarrays(res.parameters))

        norm_params = []
        for i, param in enumerate(parameters):
            param_i = parameters[i]
            for j in range(i+1, len(parameters)):
                param_j = parameters[j]
                norm_wij = np.mean([np.linalg.norm(param_i[k]-param_j[k]) for k in range(len(param_j))])
                norm_params.append(norm_wij)
        
        print(f'Distance metric: {max(norm_params)}')
        return max(norm_params)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Compute the updates, clip, and pass them for aggregation.

        Afterward, add noise to the aggregated parameters.
        """
        if failures:
            return None, {}

        if len(results) != self.num_sampled_clients:
            log(
                WARNING,
                CLIENTS_DISCREPANCY_WARNING,
                len(results),
                self.num_sampled_clients,
            )

        d = self.distance_metric(results)
        print(f'Aggregate fit. d={d}')

        for _, res in results:
            param = parameters_to_ndarrays(res.parameters)
            # Compute and clip update
            compute_clip_model_update(
                param, self.current_round_params, self.clipping_norm
            )
            log(
                INFO,
                "aggregate_fit: parameters are clipped by value: %.4f.",
                self.clipping_norm,
            )
            # Convert back to parameters
            res.parameters = ndarrays_to_parameters(param)


        # Pass the new parameters for aggregation
        aggregated_params, metrics = self.strategy.aggregate_fit(
            server_round, results, failures
        )

        # Add Gaussian noise to the aggregated parameters
        if aggregated_params:
            aggregated_params = add_gaussian_noise_to_params(
                aggregated_params,
                self.noise_multiplier/d,
                self.clipping_norm,
                self.num_sampled_clients,
            )

            log(
                INFO,
                "aggregate_fit: metric privacy noise with %.4f stdev added",
                compute_stdv(
                    self.noise_multiplier/d, self.clipping_norm, self.num_sampled_clients
                ),
            )

        return aggregated_params, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using the given strategy."""
        return self.strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function from the strategy."""
        return self.strategy.evaluate(server_round, parameters)

