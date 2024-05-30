from dataclasses import dataclass, field

import numpy
import parselmouth

@dataclass
class Sound:
    timestamps: list[int] = field(default_factory=list)
    # Shape (1, len(amplitudes))
    amplitudes: list[list[int]] = field(default_factory=list)
    sample_rate : int = 44100


@dataclass
class Spectrogram:
    timestamps: list[int] = field(default_factory=list)
    frequencies: list[int] = field(default_factory=list)
    data_matrix: list[list[int]] = field(default_factory=list)


class Parselmouth:
    __sound_data : parselmouth.Sound

    def __init__(self, filepath: str):
        self.__sound_data = parselmouth.Sound(filepath)

    def get_sound(self) -> Sound:
       return Sound(self.__sound_data.xs(), self.__sound_data.values)
       #return Sound(self.__sound_data.xs(), self.__sound_data.values[0])

    def get_spectrogram(self):
        spectrogram = self.__sound_data.to_spectrogram()
        linear_values = 10 * numpy.log10(spectrogram.values)

        spect = Spectrogram(
            spectrogram.x_grid(), spectrogram.y_grid(), linear_values
        )

        return spect
