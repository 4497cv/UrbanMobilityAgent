from dataclasses import dataclass
import sys

ZMG_PLACES = [
    "Guadalajara, Jalisco, Mexico",
    "Zapopan, Jalisco, Mexico",
    "San Pedro Tlaquepaque, Jalisco, Mexico",
    "Tonalá, Jalisco, Mexico",
    "Tlajomulco de Zúñiga, Jalisco, Mexico",
    "El Salto, Jalisco, Mexico",
]

network_types = {
    "drive",
    "walk"
}

@dataclass
class Coordinates:
    x: float
    y: float

ITESO_COORDINATES = Coordinates(20.608592, -103.414607)
PROVI_COORDINATES = Coordinates(20.700157, -103.383641)

MODE_NORMAL = "normal"
MODE_CPU    = "cpu"
MODE_GPU    = "gpu"
CPU_THREADS_LIM = 15
VEG_RADIO_INFLUENCIA_M = 50  # metros

class UserProfile:
    network_type ="drive"
    place=""
    elevation_active = False
    vegetation_active = False
    insecurity_active = False
    processing_mode = MODE_NORMAL
    ncpu_threads = 12
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
            self.elevation_active = True
            self.vegetation_active = True
            self.insecurity_active = True
        elif self.network_type == "walk":
            self.elevation_active = True
            self.vegetation_active = True
            self.insecurity_active = True

        if(w_elev == 0):
            print("Elevation is set to OFF")
            self.elevation_active = False
        else:
            print("Elevation is set to ON")

        if(w_veg == 0):
            print("Vegetation is set to OFF")
            self.vegetation_active = False
        else:
            print("Elevation is set to ON")

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

    def set_start_coordinates(self, start_coordinates_x=-103.376624, start_coordinates_y=20.630163):
        self.start_coordinates = Coordinates(x=start_coordinates_x, y=start_coordinates_y)

    def set_end_coordinates(self, end_coordinates_x=-103.376624, end_coordinates_y=20.630163):
        self.start_coordinates = Coordinates(x=end_coordinates_x, y=end_coordinates_y)

    def get_start_coordinates(self):
        return self.start_coordinates
    
    def get_stop_coordinates(self):
        return self.start_coordinates
    
    def get_place(self):
        if(self.place == "ZMG"):
            return ZMG_PLACES
        else:
            return self.place
    
    def get_processing_mode(self):
        return self.processing_mode
    
    def set_processing_mode(self, processing_mode):
        if((MODE_NORMAL == processing_mode) or
           (MODE_CPU    == processing_mode) or
           (MODE_GPU    == processing_mode)):
            return self.processing_mode
        else:
            sys.exit("incorrect processing mode has been set %s" % processing_mode)


    def get_cpu_threads(self):
        return self.ncpu_threads
    
    def set_cpu_threads(self, n_threads):
        if((n_threads >= 0) and (n_threads < CPU_THREADS_LIM)):
            self.ncpu_threads = n_threads