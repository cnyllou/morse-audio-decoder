> Gvido Bērziņš

---------------


This script has been based on some Python 2 code I found searching for this decoder, it was found on stackoverflow post mentioning [this link](https://code.google.com/archive/p/morse-to-text/).

Also based on [this online decoder](https://morsecode.world/international/decoder/audio-decoder-adaptive.html) I improved the Python 2 script and made my own spinoff.

## Fourier Transform

Calculating

Multiplying a signal by an analysing function.

Where it is similar, it is multiplied largely and dissimilar is lower.


## Discrete Fourier transform

Slightly different front the continuous Fourier transform

## Denoising Data with FFT (Python)


## The process

1. Extract data
2. Remove noise
3. Get pulses from audio
4. Translate the pulses to morse code

## Extracting data


## Removing noise

For removing noise we can use an FFT function from numpy

```python
from numpy.fft import fft
```


## Plotter class

- Making spectrogram plots throughout the process.



## How it works?

1. The script processes the given audio sample.

If the volume in the chosen frequency is louder than the "Volume threshold" then it is treated as being part of a dit or dah.
- Otherwise it records a gap

2. From these timings it determines if something is a dit, dah, or a sort of space and then converts it into a letter

In fully automatic mode, the decoder selects the loudest frequency and adjusts the Morse code speed to fit the data

* The script also has to calculate the WPM







