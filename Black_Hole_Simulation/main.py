import numpy as np
from einsteinpy.geodesic import Timelike
from einsteinpy.plotting.geodesic import StaticGeodesicPlotter

#--------------------------------------------#
# PARAMETERS
#--------------------------------------------#
# Position and Momentum (Spherical Coordinates)
position = [40., np.pi / 2, 0.] # Position of the particle
momentum = [0., 0., 3.83405] # Momentum of the particle

a = 0 # Defining Spin
steps = 5500 # Steps in the calculation
delta = 1 # Step size

#--------------------------------------------#
# CALCULATING GEODESIC
#--------------------------------------------#
geod = Timelike(
    metric="Schwarzschild",
    metric_params=(a,),
    position=position,
    momentum=momentum,
    steps=steps,
    delta=delta,
    suppress_warnings=True,
    return_cartesian=True
)

# Use InteractiveGeodesicPlotter() to get interactive plots
sgpl = StaticGeodesicPlotter()
sgpl.plot2D(geod)
sgpl.show()
