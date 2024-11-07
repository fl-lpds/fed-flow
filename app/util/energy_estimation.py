import warnings

import requests

from app.config import config

URL = "http://127.0.0.1:8023"
warnings.filterwarnings('ignore')
is_init = False


def init(pid):
    session = requests.session()
    session.trust_env = False
    session.get(url=URL + "/init/" + str(pid) + "/" + str(config.simnet))
    is_init = True


def set_simnet(simnet):
    """set simulation network bandwidth"""
    if is_init:
        session = requests.session()
        session.trust_env = False
        session.get(url=URL + "/set-simnet/")


def computation_start():
    if is_init:
        session = requests.session()
        session.trust_env = False
        session.get(url=URL + "/computation-start/")


def computation_end():
    if is_init:
        session = requests.session()
        session.trust_env = False
        session.get(url=URL + "/computation-end/")


def start_transmission():
    if is_init:
        session = requests.session()
        session.trust_env = False
        session.get(url=URL + "/start-transmission/")


def end_transmission(bits):
    if is_init:
        session = requests.session()
        session.trust_env = False
        session.get(url=URL + "/end-transmission/" + str(bits))


def energy():
    if is_init:
        session = requests.session()
        session.trust_env = False
        result = session.get(url=URL + "/energy/")
        return result.text
    return 0


def remaining_energy():
    if is_init:
        session = requests.session()
        session.trust_env = False
        result = session.get(url=URL + "/remaining-energy/")
        return result.text
    return 10


def energy_and_time_comp_tr():
    if is_init:
        session = requests.session()
        session.trust_env = False
        result = session.get(url=URL + "/energy/time/comp_tr/").text.split(",")
        comp_e = float(result[0][1:])
        tr_e = float(result[1][:])
        comp_time = float(result[2][:])
        tr_time = float(result[3][:len(result[3]) - 1])
        return comp_e, tr_e, comp_time, tr_time
