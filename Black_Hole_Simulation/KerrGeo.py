import numpy as np

from einsteinpy.geodesic import Timelike #, Geodesic, Nulllike
from einsteinpy.plotting import GeodesicPlotter, StaticGeodesicPlotter #, InteractiveGeodesicPlotter


# Initial Conditions
position = [4., np.pi / 3, 0.]
momentum = [0., 0., -1.5]
a = 0. # Schwarzschild Black Hole

geod = Timelike(
    metric = "Schwarzschild",
    metric_params = (a,),
    position=position,
    momentum=momentum,
    steps=15543, # As close as we can get before the integration becomes highly unstable
    delta=0.0005,
    return_cartesian=True,
    omega=0.01, # Small omega values lead to more stable integration
    suppress_warnings=True, # Uncomment to view the tolerance warning
)

gpl = GeodesicPlotter()

gpl.show()
