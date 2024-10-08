import scipy

import numpy as np
import numpy.typing as npt

import pyqtgraph as pg


def create_lut() -> npt.NDArray[np.float64]:
    """
    Return a greyscale LUT where darker is more intense.
    """
    lut = np.zeros((256, 4), dtype=np.ubyte)
    for i in range(256):
        lut[i] = [255 - i, 255 - i, 255 - i, 255]

    return lut


defaut_spectrogram_lut: npt.NDArray[np.float64] = create_lut()


class Spectrogram(pg.ImageItem):
    """Spectrogram displayed using a `pyqtgraph.ImageItem`."""

    def __init__(
        self,
        frequency_samples: npt.NDArray[np.float64] | None = None,
        time_segments: npt.NDArray[np.float64] | None = None,
        spect_data: npt.NDArray[np.float64] | None = None,
        lut: npt.NDArray[np.float64] = defaut_spectrogram_lut,
        zoom_blur: bool = True,
        axisOrder: str = "row-major",
        **kargs
    ) -> None:
        """
        Parameters
        ----------

        frequency_samples: ndarray
            Array of sample frequency_samples.
        time_segments: ndarray
            Array of segment times.
        spect_data: ndarray
            Spectrogram of x. By default, the last axis of Sxx corresponds
            to the segment times.
        lut: ndarray, optional
            pyqtgraph lut, passed to ImageItem. (Defaults to greyscale, where
            more intense is darker.)
        zoom_blur: bool
            blur the image to look more like praat when zoomed.
        axisOrder
            Uses 'row-major' by default unlike pyqtgraph.
        """

        # if frequency_samples.shape[0] != spect_data.shape[0]:
        #     raise ValueError(
        #             f"The dimensions of frequency_samples {frequency_samples.shape} "+
        #             f" and spect_data {spect_data.shape} are not compatible. " + 
        #             f"{frequency_samples.shape[0]} != {spect_data.shape[0]}"
        #     )
        #
        # if time_segments.shape[0] != spect_data.shape[1]:
        #     raise ValueError(
        #             f"The dimensions of time_segments {time_segments.shape} "+
        #             f"and spect_data {spect_data.shape} are not compatible. " +
        #             f"{time_segments.shape[0]} != {spect_data.shape[1]}"
            # )

        if zoom_blur:
            spect_data = scipy.ndimage.zoom(spect_data, 6, order=4)

        super().__init__(
            axisOrder=axisOrder, lut=lut, **kargs
        )

        if not (
            frequency_samples is None or
            time_segments is None or
            spect_data is None
        ):
            self.set_data(frequency_samples, time_segments, spect_data)

    def set_data(
        self,
        frequency_samples: npt.NDArray[np.float64],
        time_segments: npt.NDArray[np.float64],
        spect_data: npt.NDArray[np.float64],
    ) -> None:
        self.setImage(spect_data)

        # Scale the X and Y Axis to time and frequency
        rect = pg.QtCore.QRectF(0, 0, max(time_segments), max(frequency_samples))
        self.setRect(rect)


def _example_scipy_spectrogram():
    rng = np.random.default_rng()
    fs = 10e3

    N = 1e5
    amp = 2 * np.sqrt(2)
    noise_power = 0.01 * fs / 2
    time = np.arange(N) / float(fs)
    mod = 500 * np.cos(2 * np.pi * 0.25 * time)
    carrier = amp * np.sin(2 * np.pi * 3e3 * time + mod)

    noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    noise *= np.exp(-time / 5)

    x = carrier + noise
    return scipy.signal.spectrogram(x, fs)

