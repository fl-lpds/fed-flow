import logging

init_energy = 200
tx_power = 0.15849
power = 15
counter = 0
process = None
simulate_network = True
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("requests").setLevel(logging.WARNING)
energy_logger = logging.getLogger(__name__)
