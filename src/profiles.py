from dataclasses import dataclass

network_types = {
    "drive",
    "walk"
}

@dataclass
class Coordinates:
    x: float
    y: float

class UserProfile:
    network_type ="drive"
    place=""
    elevation_active = False
    vegetation_active = False
    insecurity_active = False
    start_coordinates = Coordinates(x=-103.376624, y=20.630163)
    end_coordinates   = Coordinates(x=-103.384384, y=20.697814)
    w_time = 0
    w_elev = 0
    w_veg = 0

    def __init__(self, network_type="drive", place="Guadalajara, Mexico",
                 start_coordinates_x=-103.376624, start_coordinates_y=20.630163,
                 end_coordinates_x=-103.384384, end_coordinates_y=20.697814,
                 w_time=0.5, w_elev=0.3, w_veg=0.2):
        self.network_type = network_type
        self.place = place
        self.w_time = w_time
        self.w_elev = w_elev
        self.w_veg  = w_veg

        if self.network_type == "drive":
            self.elevation_active = False
            self.vegetation_active = True
            self.insecurity_active = True
        elif self.network_type == "walk":
            self.elevation_active = True
            self.vegetation_active = True
            self.insecurity_active = True

        print("inicialización del perfil de usuario:")
        print("tipo de ruta: %s" % self.network_type)
        print("lugar: %s" % self.place)
        print("coordenadas de inicio: x=%.6f, y=%.6f" % (start_coordinates_x, start_coordinates_y))
        print("coordenadas de destino: x=%.6f, y=%.6f" % (end_coordinates_x, end_coordinates_y))
        print("pesos: w_time=%.2f  w_elev=%.2f  w_veg=%.2f" % (w_time, w_elev, w_veg))

        self.start_coordinates = Coordinates(x=start_coordinates_x, y=start_coordinates_y)
        self.end_coordinates = Coordinates(x=end_coordinates_x, y=end_coordinates_y)

    def get_network_type(self) -> str:
        return self.network_type


