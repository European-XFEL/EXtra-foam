# Gets the data and displays it

import matplotlib.pyplot as plt

from karabo_bridge import KaraboBridge
from lpd_tools import LPDConfiguration, offset_image

config = LPDConfiguration(hole_size=-26.28e-3, q_offset=3)

client = KaraboBridge("tcp://localhost:4545")

while True:
    data = client.next()
    d = data["FXE_DET_LPD1M-1/DET/combined"]["image.data"]

    for i in range(d.shape[0]):
        plt.imshow(d[i], vmin=-10, vmax=6000)
        plt.show()

        fixed = offset_image(config, d[i])
        plt.imshow(fixed, vmin=-10, vmax=6000)
        plt.show()
