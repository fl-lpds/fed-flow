from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class NodeIdentifier:
    ip: str
    port: int

    def __str__(self):
        return f"{self.ip}:{self.port}"

    def get_exchange_name(self, extra_target: str = '') -> str:
        if extra_target:
            return f"{self.__str__()}_{extra_target}"
        return self.__str__()

    def __eq__(self, other):
        return self.ip == other.ip and self.port == other.port

    def __hash__(self):
        return hash(self.ip) + hash(self.port)
