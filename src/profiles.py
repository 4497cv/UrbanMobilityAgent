


class UserProfile:
    network_type ="walk"

    def __init__(self, network_type="walk"):
        self.network_type = network_type

    def get_network_type(self) -> str:
        return self.network_type


