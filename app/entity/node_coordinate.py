import re
class NodeCoordinate:
    latitude: float
    longitude: float
    altitude: float
    seconds_since_start: float

    def __init__(self, latitude: float, longitude: float, altitude: float, seconds_since_start: float):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.seconds_since_start = seconds_since_start

    def __str__(self):
        return (f"Latitude: {self.latitude}, Longitude: {self.longitude}, "
                f"Altitude: {self.altitude}, Seconds Since Start: {self.seconds_since_start}")

    @classmethod
    def parse(cls, text: str, seconds_since_start: float = 0.0):
        """Parses a string like '39.99,116.33,45' or '39.99 و 116.33 و 45'."""
        nums = [float(x) for x in re.findall(r'[-+]?\d+(?:\.\d+)?', text or "")]
        if len(nums) < 2:
            raise ValueError("Not enough numeric values in coordinate string")
        lat, lon = nums[0], nums[1]
        alt = nums[2] if len(nums) >= 3 else 0.0
        return cls(lat, lon, alt, seconds_since_start)