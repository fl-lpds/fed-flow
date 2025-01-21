import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from app.config import config
from app.config.logger import fed_logger
from app.dto.message import GlobalWeightMessage, NetworkTestMessage, SplitLayerConfigMessage, IterationFlagMessage
from app.dto.received_message import ReceivedMessage
from app.entity.aggregators.base_aggregator import BaseAggregator
from app.entity.fed_base_node_interface import FedBaseNodeInterface
from app.entity.http_communicator import HTTPCommunicator
from app.entity.mobility_manager import MobilityManager
from app.entity.node_identifier import NodeIdentifier
from app.entity.node_type import NodeType
from app.model.utils import get_available_torch_device
from app.util import model_utils


# noinspection PyTypeChecker
class FedClient(FedBaseNodeInterface):

    def __init__(self, ip: str, port: int, model_name, dataset, train_loader, LR, cluster,
                 aggregator: BaseAggregator, neighbors: list[NodeIdentifier] = None):
        super().__init__(ip, port, NodeType.CLIENT, cluster, neighbors)
        self._edge_based = None
        self.device = get_available_torch_device()
        self.model_name = model_name
        self.dataset = dataset
        self.train_loader = train_loader
        self.split_layers = None
        self.criterion = nn.CrossEntropyLoss()
        self.mobility_manager = MobilityManager(self)
        self.aggregator = aggregator

        # Standard model setup
        self.uninet = model_utils.get_model('Unit', None, self.device, self.is_edge_based)
        self.net = self.uninet
        self.optimizer = optim.SGD(self.net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, config.lr_step_size, config.lr_gamma)

        self.use_moon = False        # Toggle MOON
        self.moon_lambda = 1.0
        self.old_net = None          # For MOON contrastive

        self.use_fedprox = True     # Toggle FedProx
        self.mu_fedprox = 0.01
        self.global_net = None       # For FedProx reference

    def initialize(self, learning_rate):
        self.net = model_utils.get_model('Client', self.split_layers, self.device, self.is_edge_based)
        self.optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    def gather_global_weights(self, node_type: NodeType):
        msgs: list[ReceivedMessage] = self.gather_msgs(GlobalWeightMessage.MESSAGE_TYPE, [node_type])
        msg: GlobalWeightMessage = msgs[0].message
        pweights = model_utils.split_weights_client(msg.weights[0], self.net.state_dict())

        # For MOON: store old_net
        if self.use_moon:
            if self.old_net is None:
                self.old_net = model_utils.get_model('Client', self.split_layers, self.device, self.is_edge_based)
            self.old_net.load_state_dict(self.net.state_dict())

        # For FedProx: store global_net
        if self.use_fedprox:
            if self.global_net is None:
                self.global_net = model_utils.get_model('Client', self.split_layers, self.device, self.is_edge_based)
            self.global_net.load_state_dict(pweights)

        # Load newly received global weights
        self.net.load_state_dict(pweights)

    def scatter_network_speed_to_edges(self):
        msg = NetworkTestMessage([self.net.to(self.device).state_dict()])
        self.scatter_msg(msg, [NodeType.EDGE])
        fed_logger.info("test network sent")

        _ = self.gather_msgs(NetworkTestMessage.MESSAGE_TYPE, [NodeType.EDGE])
        fed_logger.info("test network received")

    def gather_split_config(self):
        msgs = self.gather_msgs(SplitLayerConfigMessage.MESSAGE_TYPE, [NodeType.EDGE])
        msg: SplitLayerConfigMessage = msgs[0].message
        self.split_layers = msg.data

    def start_offloading_train(self):
        self.net.to(self.device)
        self.net.train()

        i = 0

        split_point = self.split_layers
        if isinstance(self.split_layers, list):
            split_point = self.split_layers[0]

        # 1) No Offloading
        if split_point == model_utils.get_unit_model_len() - 1:
            fed_logger.info("no offloding training start----------------------------")
            self.scatter_msg(IterationFlagMessage(False), [NodeType.EDGE])

            i += 1

            for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                ce_loss = self.criterion(outputs, targets)

                # (A) MOON
                moon_loss = 0.0
                if self.use_moon and self.old_net is not None:
                    with torch.no_grad():
                        old_rep = self.old_net.get_representation(inputs)
                    new_rep = self.net.get_representation(inputs)
                    moon_loss = self._compute_moon_loss(new_rep, old_rep)

                # (B) FedProx
                fedprox_loss = 0.0
                if self.use_fedprox and self.global_net is not None:
                    fedprox_loss = self._compute_fedprox_loss()

                total_loss = ce_loss + (self.moon_lambda * moon_loss) + fedprox_loss
                total_loss.backward()
                self.optimizer.step()

            self.scheduler.step()

        # 2) Partial Offloading
        elif split_point < model_utils.get_unit_model_len() - 1:
            fed_logger.info(f"offloding training start {self.split_layers}----------------------------")
            self.scatter_msg(IterationFlagMessage(True), [NodeType.EDGE])

            i += 1

            for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if self.optimizer is not None:
                    self.optimizer.zero_grad()

                outputs = self.net(inputs)

                # (A) MOON partial
                if self.use_moon and self.old_net is not None:
                    with torch.no_grad():
                        old_rep = self.old_net.get_representation(inputs)
                    new_rep = self.net.get_representation(inputs)
                    moon_loss = self._compute_moon_loss(new_rep, old_rep)
                    (moon_loss * self.moon_lambda).backward(retain_graph=True)

                # (B) FedProx partial
                if self.use_fedprox and self.global_net is not None:
                    fedprox_loss = self._compute_fedprox_loss()
                    fedprox_loss.backward(retain_graph=True)

                # Send partial output & target to the Edge
                self.scatter_msg(IterationFlagMessage(True), [NodeType.EDGE])
                msg = GlobalWeightMessage([outputs.to(self.device), targets.to(self.device)])
                self.scatter_msg(msg, [NodeType.EDGE])

                # Receive gradient
                fed_logger.info("receiving gradients")
                msgs: list[ReceivedMessage] = self.gather_msgs(GlobalWeightMessage.MESSAGE_TYPE, [NodeType.EDGE])
                msg: GlobalWeightMessage = msgs[0].message
                gradients = msg.weights[0].to(self.device)
                fed_logger.info("received gradients")

                # Backprop partial gradient
                outputs.backward(gradients)

                if self.optimizer is not None:
                    self.optimizer.step()

                i += 1

            self.scheduler.step()
            self.scatter_msg(IterationFlagMessage(False), [NodeType.EDGE])

    def scatter_local_weights(self):
        self.scatter_msg(GlobalWeightMessage([self.net.to(self.device).state_dict()]), [NodeType.EDGE])

    def scatter_random_local_weights(self):
        is_leader = HTTPCommunicator.get_is_leader(self)
        if is_leader:
            self.scatter_msg(GlobalWeightMessage([self.net.to(self.device).state_dict()]), [NodeType.SERVER])

    def no_offloading_train(self):
        self.net.to(self.device)
        self.net.train()
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(self.train_loader)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.net(inputs)
            ce_loss = self.criterion(outputs, targets)

            # (A) MOON
            moon_loss = 0.0
            if self.use_moon and self.old_net is not None:
                with torch.no_grad():
                    old_rep = self.old_net.get_representation(inputs)
                new_rep = self.net.get_representation(inputs)
                moon_loss = self._compute_moon_loss(new_rep, old_rep)

            # (B) FedProx
            fedprox_loss = 0.0
            if self.use_fedprox and self.global_net is not None:
                fedprox_loss = self._compute_fedprox_loss()

            total_loss = ce_loss + (self.moon_lambda * moon_loss) + fedprox_loss
            total_loss.backward()
            self.optimizer.step()

    def gossip_with_neighbors(self):
        edge_neighbors = self.get_neighbors([NodeType.CLIENT])
        msg = GlobalWeightMessage([self.uninet.to(self.device).state_dict()])
        self.scatter_msg(msg, [NodeType.CLIENT])
        gathered_msgs = self.gather_msgs(GlobalWeightMessage.MESSAGE_TYPE, [NodeType.CLIENT])
        gathered_models = [(msg.message.weights[0], config.N / len(edge_neighbors)) for msg in gathered_msgs]
        zero_model = model_utils.zero_init(self.uninet).state_dict()
        aggregated_model = self.aggregator.aggregate(zero_model, gathered_models)
        self.uninet.load_state_dict(aggregated_model)

    # -------------------------------
    # Helpers for MOON / FedProx
    # -------------------------------
    def _compute_moon_loss(self, new_rep: torch.Tensor, old_rep: torch.Tensor, temperature=0.5) -> torch.Tensor:
        """
        Minimal negative-only MOON contrastive example.
        new_rep vs. old_rep => negative pair
        new_rep vs. new_rep => positive pair
        """
        cos = nn.CosineSimilarity(dim=1)
        pos_sim = cos(new_rep, new_rep)
        neg_sim = cos(new_rep, old_rep)
        logits = torch.stack([pos_sim, neg_sim], dim=1) / temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = nn.CrossEntropyLoss()(logits, labels)
        fed_logger.info(f"MOON loss computation complete. Loss value: {loss.item()}.")
        return loss

    def _compute_fedprox_loss(self) -> torch.Tensor:
        """
        FedProx penalty:
        mu_fedprox/2 * sum(||W_local - W_global||^2).
        """
        loss = torch.tensor(0.0, device=self.device)
        for param_local, param_global in zip(self.net.parameters(), self.global_net.parameters()):
            loss += torch.sum((param_local - param_global.detach()) ** 2)
        loss = 0.5 * self.mu_fedprox * loss
        fed_logger.info(f"FedProx loss computation complete. Loss value: {loss.item()}.")
        return loss
