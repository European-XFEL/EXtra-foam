import math
import numpy as np
import pyFAI
import matplotlib.pyplot as plt


from lpd_receiver import LPDConfiguration, LPDReceiver
from karabo_bridge import KaraboBridge

from constants import (
    center_x, center_y, distance, hole_pixel_size, ts, tSi, mus,
    muSi, pixel_size, qnorm_max, qnorm_min, q_offset, SM, wavelength_lambda
)

# matplotlib value range
vmin = -10
vmax = 6000
plt.ion()

bridge = KaraboBridge("tcp://localhost:4545")
config = LPDConfiguration(hole_size=-26.28e-3, q_offset=3)

lpd_stuff = LPDReceiver(config=config, bridge=bridge)


"""
Example Data Manipulation
=========================

Do an azimuthal integration on LPD data:
 - Retrieve data from the Karabo bridge
 - Assemble modules into a single full image
 - Perform azimuthal integration
"""


# hole_pixel_size is the size of the gap in the center of the detector
total_img = np.zeros([SM * 4 + hole_pixel_size + q_offset,
                      SM * 4 + hole_pixel_size + q_offset],
                     dtype='int32')

# setting the integrator
Sq = None
npt = 512
ai = pyFAI.AzimuthalIntegrator(dist=distance,
                               poni1=center_y*pixel_size,
                               poni2=center_x*pixel_size,
                               pixel1=pixel_size,
                               pixel2=pixel_size,
                               rot1=0, rot2=0, rot3=0,
                               wavelength=wavelength_lambda)

N_set = []
train_ids = []

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plot, = ax.plot([], [])

plt.ylabel("Scattering Signal, S(q) [arb. un.]", fontsize=20)
plt.xlabel("Momentum Transfer, q[1/A]", fontsize=20)
plt.figure(figsize=(12, 12))

while True:
    pulse_integ_result = []
    data = lpd_stuff.get_pulse_data()
    for pulse in range(30):
        pulse_data = next(data)
        combined_imgs, full_img = lpd_stuff.stitch_image(pulse_data)

        # Define mask data to substract to reduce noise
        total_img = total_img + combined_imgs
        mask_data = np.zeros(combined_imgs.shape)
        mask_data[np.where(combined_imgs == 0)] = 1

        Q, i_unc = ai.integrate1d(combined_imgs,
                                  npt,
                                  method="lut",
                                  mask=mask_data,
                                  radial_range=(0.1, 4),
                                  correctSolidAngle=True,
                                  polarization_factor=1,
                                  unit="q_A^-1")

        I_unc = i_unc[:, None]

        if Sq is None:
            # Get the momentum transfer, q, from the integration
            q = Q[:, None]
            scattering = q * wavelength_lambda * 1e10 / (4 * math.pi)
            # 2-theta scattering angle
            tth = np.rad2deg(scattering)

            # silicon sensor absorption correction
            T_Si = ((1 - np.exp(-muSi*tSi)) /
                    (1 - np.exp(-muSi * tSi /
                     np.cos(np.deg2rad(tth))))
                    )

            Ts = (1 / (mus * ts) *
                  np.cos(np.deg2rad(tth)) /
                  (1 - np.cos(np.deg2rad(tth))) *
                  (np.exp(-mus * ts) -
                  np.exp(-mus * ts / np.cos(np.deg2rad(tth))))
                  )
            # sample absorption correction in isotropic case
            # (not to do here if done on the image before integration)
            Ts = Ts/Ts[0]

            Qnorm = Q[np.where(np.logical_and(Q >= qnorm_min,
                                              Q <= qnorm_max))]

        # Normalize
        to_trapz = i_unc[np.where(np.logical_and(Q >= qnorm_min,
                                                 Q <= qnorm_max))]
        N = np.trapz(to_trapz, x=Qnorm)

        # Apply corrections
        I_cor = I_unc * T_Si / Ts

        # Define or update the scattering signal
        Sq = I_unc if Sq is None else np.concatenate((Sq, I_unc), axis=1)

        pulse_integ_result.append((q, I_unc))

    plot.clear()
    for data in pulse_integ_result:
        plot.set_xdata(np.append(plot.get_xdata(), data[0]))
        plot.set_ydata(np.append(plot.get_ydata(), data[1]))

    plt.title("Azimuthal Integration over {n} pulses"
              "".format(n=len(pulse_integ_result)), fontsize=20)
    plt.draw()
