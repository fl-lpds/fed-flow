class Process:
    def __init__(self, pid, init_energy, comp_power_usage, trans_power_usage):
        self.remaining_energy = init_energy
        self.pid = pid
        self.start_comp_time = 0
        self.end_comp_time = 0
        self.start_tr_time = 0
        self.end_tr_time = 0
        self.end_comp = False
        self.cpu_utilization = 0
        self.system_energy = 0
        self.cpu_u_count = 0
        self.transmission_bits = 0
        self.transmission_time = 0
        self.comp_time = 0
        self.bandwidth = 0
        self.comp_power_usage = comp_power_usage
        self.trans_power_usage = trans_power_usage
