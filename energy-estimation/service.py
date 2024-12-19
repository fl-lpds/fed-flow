import logging
import threading
import warnings

import uvicorn
from colorama import Fore
from fastapi import FastAPI

import config
import system_utils
from config import energy_logger
from log_filters import EndpointFilter
from process import Process

warnings.filterwarnings('ignore')

app = FastAPI()
config.process = Process(pid=0, init_energy=config.init_energy, comp_power_usage=config.power,
                         trans_power_usage=config.tx_power)

excluded_endpoints = ["/init/", "/", "/computation-start/", "/computation-end/", "/start-transmission/",
                      "/end-transmission/", "/get-cpu-utilization/"]

# Add filter to the logger
logging.getLogger("uvicorn.access").addFilter(EndpointFilter(excluded_endpoints))


@app.get("/init/{pid}/{simnet}")
async def init(pid, simnet):
    system_utils.init_p(config.process, pid, simnet)


@app.get("/")
async def root():
    return system_utils.pcpuc(config.process, config)


@app.get("/computation-start/")
async def computation_start():
    x = threading.Thread(target=system_utils.computation_start, args=(config.process,))
    x.start()
    # energy_logger.info("computation started")
    # asyncio.create_task(system_utils.computation_start(config.process))


@app.get("/computation-end/")
async def computation_end():
    system_utils.computation_end(config.process)


@app.get("/start-transmission/")
async def start_transmission():
    system_utils.start_transmission(config.process)


@app.get("/end-transmission/{bits}")
async def end_transmission(bits):
    # energy_logger.info(Fore.RED + f"{int(bits)}")
    system_utils.end_transmission(config.process, int(bits))


@app.get("/get-transmission-time/")
async def get_transmission_time():
    return system_utils.get_transmission_time(config.process)


@app.get("/reset-transmission-time/")
async def reset_transmission_time():
    return system_utils.reset_transmission_time(config.process)


@app.get("/get-computation-time/")
async def get_computation_time():
    return system_utils.get_computation_time(config.process)


@app.get("/set-simnet/{simnetbw}")
async def set_simnet(simnetbw):
    system_utils.set_simnet(config.process, simnetbw)


@app.get("/energy/{computation_time}")
async def energy(computation_time):
    return system_utils.estimate_total_energy(config, config.process, float(computation_time))


@app.get("/remaining-energy/")
async def remaining_energy():
    return system_utils.remaining_energy(config.process)


@app.get("/energy/time/comp_tr")
async def energy_and_time_comp_tr():
    comp_time = config.process.comp_time
    tr_time = config.process.transmission_time
    energy_logger.info(Fore.GREEN + f"computation: {comp_time}, transmission: {tr_time}")

    comp_e = system_utils.estimate_computation_energy(config.process)
    tr_e = system_utils.estimate_communication_energy(config, config.process)
    energy_logger.info(Fore.MAGENTA + f"energy-computation: {comp_e}, energy-transmission: {tr_e}")

    comp_tr = str(comp_e) + "," + str(tr_e) + "," + str(comp_time) + "," + str(tr_time)

    config.process.comp_time = 0
    config.process.cpu_u_count = 0
    config.process.end_comp = False
    config.process.cpu_utilization = 0
    config.process.transmission_time = 0
    return comp_tr


@app.get("/get-cpu-utilization/{pid}")
async def get_cpu_utilization(pid):
    return system_utils.get_cpu_u(pid)


@app.get("/get-power")
async def get_power_usage():
    return system_utils.get_power_usage(config.process)


@app.get("/get-utilization")
async def get_utilization():
    return system_utils.get_utilization(config.process)


uvicorn.run(app, host="0.0.0.0", port=8023)
