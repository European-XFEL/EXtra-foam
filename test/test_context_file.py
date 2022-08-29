import numpy as np
from scipy.signal import gausspulse

from extra_foam.utils import rich_output, Series as S


# This view won't work unless the suite is reading from EXtra-foam
@View.Image
def train_image(masked_train: "foam#image.masked_mean"):
    return masked_train

@View.Scalar
def scalar(tid: "internal#train_id"):
    return rich_output(tid,
                       title="Scalar",
                       xlabel="Index",
                       ylabel="Train ID")

# We don't actually use the train id in this view, but we add it as an
# argument so that the view gets executed anyway. Note that View's with
# names beginning with an underscore are hidden by default, check the 'Show
# hidden views' checkbox in a viewer frame to see them.
@View.Vector
def _vector(_: "internal#train_id"):
    # Example copied from: 
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.gausspulse.html
    t = np.linspace(-1, 1, 2 * 100, endpoint=False)
    i, q, e = gausspulse(t, fc=5, retquad=True, retenv=True)

    # Add some noise to make it more interesting
    noise_gen = lambda scale: scale * (np.random.rand(len(t)) - 0.5)
    i += noise_gen(0.2)
    q += noise_gen(0.1)
    e += noise_gen(0.05)

    return rich_output(t, 
                       y1=S(i, name="Real component"),
                       y2=S(q, name="Imaginary component"),
                       y3=S(e, name="Envelope"),
                       title="Gaussian modulated sinusoid",
                       xlabel="Amplitude",
                       ylabel="Time")
                       
@View.Matrix_Histogram
def corr_jf_diode(jf_frame: 'FXE_DET_JF1M/CAL/CORR_GAIN:output[data.image.pixels]', diode_signal: 'FXE_ADC_SOMEWHERE/ADC/SOMETHING:output[channel_1_A.signal]'):
    # Do cutting-edge science
    return x, y   # May also yield multiple for several pulses
