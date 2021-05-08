from numpy.fft import fft
from matplotlib.pyplot import *
from numpy import *

from pydub import AudioSegment
from pydub.playback import play

import scipy.io.wavfile as wavfile
import csv
import time
import os

SCRIPT_DIR = os.path.dirname(__file__)
REPORT_DIR = os.path.join(SCRIPT_DIR, 'reports')
SAMPLE_DIR = os.path.join(SCRIPT_DIR, 'samples')

plot_file = os.path.join(REPORT_DIR, 'plot.png')
file = os.path.join(SAMPLE_DIR, 'az.wav')
file = os.path.join(SAMPLE_DIR, 'wikipedia.wav')  # Give these as args
original_file = os.path.join(REPORT_DIR, 'original')


class Plotter:

    def __init__(self, format="pdf"):
        self.format = format

    def saveplot(self, name, data, length=-1, height=-1, dpi=None):
        name = os.path.join(REPORT_DIR, name)
        plot(data)
        if length != -1:
            axis(xmax=length)
        if height != -1:
            axis(ymax=height)
        savefig(name + "." + self.format, format=self.format, dpi=dpi)
        cla()

    def specgram(self, name, signal):
        name = os.path.join(REPORT_DIR, name)
        spectrogram = specgram(signal, Fs = 2)  # Fs needs to be supplied
        savefig(name + "." + self.format, format=self.format)
        cla()
        return spectrogram


class SoundFile:

    def __init__(self, path):
        the_file = wavfile.read(path)
        self.rate = the_file[0]  # Optimise the 'the_file' variable
        self.length = len(the_file[1])
        self.data = the_file[1]

        power = 10

        while pow(2,power) < self.length:
            power += 1
        self.data = append(self.data, zeros(pow(2,power) - self.length))

    def setdata(self, data):
        self.data = data

    def getdata(self):
        return self.data

    def getlength(self):
        return self.length

    def saveas(self, path):
        wavfile.write(path, self.rate, self.data)

    def saveplot(self, fileName):
        plotter.saveplot(fileName, self.data, length=self.length)


class SignalFilter:

    def filter(self, soundfile):
        trans = fft.rfft(soundfile.getdata())  # Transposing data?
        trans_real = abs(trans)

        #2b - Save the plot
        plotter.saveplot("transformed", trans_real)  # Saving transposed data

        #3 - busco la frecuencia
        band = 2000

        # ignore the first 200Hz
        hzignored = 200
        frec = hzignored + argmax(trans_real[hzignored:])
        min = int((frec - band / 2)) if (frec > band / 2) else 0

        filter_array = append(zeros(min), ones(band))
        filter_array = append(filter_array, zeros(len(trans_real) - len(filter_array)))
        filtered_array = multiply(trans, filter_array)
        plotter.saveplot("filtered_trans",abs(filtered_array))

        #4 - Untransform?
        filtered_signal = array(fft.irfft(filtered_array)[:soundfile.getlength()], dtype="int16")
        plotter.saveplot("filtered_signal",filtered_signal)
        soundfile.setdata(filtered_signal)


class SpectreAnalyzer:

	def spectrogram(self, signal):
		spectrogram = plotter.specgram("spectrogram", signal)  # Audio file data
		return spectrogram

	def sumarizecolumns(self, mat):
		vec_ones = ones(len(mat))
		vec_sum = (matrix(vec_ones) * matrix(mat)).transpose()
		plotter.saveplot("frecuency_volume",vec_sum)
		return vec_sum

	def findpresence(self, vec_sum):
		presence = zeros(len(vec_sum))
		threshold = max(vec_sum) / 2.0
		for i in range(len(presence)):
			if vec_sum[i] > threshold:
				presence[i] = 1
		plotter.saveplot("presence", presence, dpi=300, height=5)
		return presence

	def findpulses(self, soundfile):  # Audio file is passed in
		spec = self.spectrogram(soundfile.getdata())   # Audio file data is passed to the spectrogram
		# spec[0] es la matriz del rojo
		red_matrix = spec[0]
		vec_sum = self.sumarizecolumns(red_matrix)
		presence = self.findpresence(vec_sum)
		return presence


class ShortLong:
	def __init__(self, shorts, longs):
		self.shortmean = mean(shorts)
		self.shortstd = std(shorts)
		self.longmean = mean(longs)
		self.longstd = std(longs)

	def tostring(self):
		return "short: (" + repr(self.shortmean) + ", " + repr(self.shortstd) + ")\n\
long: (" + repr(self.longmean) + ", " + repr(self.longstd) + ")"


class SymbolDecoder:
	def __init__(self, onessl, zerossl, zeroextra=None):
		self.onessl = onessl
		self.zerossl = zerossl
		self.zeroextra = zeroextra

	def get(self, sl, n, ifshort, iflong, ifnone="?"):
		d = 4
		if (n > sl.shortmean - d * sl.shortstd) and (n < sl.shortmean + d * sl.shortstd):
			return ifshort
		if (n > sl.longmean - d * sl.longstd) and (n < sl.longmean + d * sl.longstd):
			return iflong
		return ifnone

	def getonesymbol(self, n):
		return self.get(self.onessl, n, ".", "-")

	def getzerosymbol(self, n):
		sym = self.get(self.zerossl, n, "", " ")
		if sym == "":
			return sym
		return self.get(self.zeroextra, n, " ", " | ", ifnone=" ")



class PulsesAnalyzer:

	def compress(self, pulses):
		vec = []
		i = 0

		if pulses[0] == 1:
			vec += [0]
			i = 1

		last = pulses[0]

		while i < len(pulses):
			c = 0
			last = pulses[i]
			while i < len(pulses) and pulses[i] == last:
				i += 1
				c += 1
			vec += [c]
			i += 1

		vec = vec[1:-1]
		return vec

	def split(self, vec):
		onesl = zeros(1+len(vec)//2)  # Operator to do integer division
		zerosl = zeros(len(vec)//2)
		for i in range(len(vec)//2):
			onesl[i] = vec[2*i]
			zerosl[i] = vec[2*i+1]
		onesl[-1] = vec[-1]
		return (onesl, zerosl)

	def findshortlongdup(self, vec):
		sor = sort(vec)
		last = sor[0]
		for i in range(len(sor))[1:]:
			if sor[i] > 2*last:
				shorts = sor[:i-1]
				longs = sor[i:]
				return (shorts, longs)
		return (vec, [])

	def createshortlong(self, shorts, longs):
		return ShortLong(shorts, longs)

	def findshortlong(self, vec):
		dup = self.findshortlongdup(vec)
		return self.createshortlong(dup[0], dup[1])


class PulsesTranslator:
	def tostring(self, pulses):
		pa = PulsesAnalyzer()
		comp_vec = pa.compress(pulses)
		comp_tup = pa.split(comp_vec)

		onessl = pa.findshortlong(comp_tup[0])
		# zeros are subdivided
		dup = pa.findshortlongdup(comp_tup[1])
		zerossl = pa.createshortlong(dup[0], dup[1])
		dup2 = pa.findshortlongdup(dup[1])
		zeroextra = pa.createshortlong(dup2[0], dup2[1])

		symdec = SymbolDecoder(onessl, zerossl, zeroextra)

		s = ""
		for i in range(len(comp_vec)//2):
			s += symdec.getonesymbol(comp_vec[2*i])
			s += symdec.getzerosymbol(comp_vec[2*i+1])
		s += symdec.getonesymbol(comp_vec[-1])
		return s


class Codes:
	def __init__(self, path):
		data = csv.DictReader(open(path), delimiter=',', fieldnames=["char", "code"])
		self.dic = {}
		for entry in data:
			self.dic[entry["code"]] = entry["char"]

	def tochar(self, code):
		if code in self.dic:
			return self.dic[code]
		return "?"


class StringTranslator:
	def __init__(self):
		self.codes = Codes("codes.csv")

	def totext(self, s):
		text = ""
		for code in s.split():
			if code == "|":
				char = " "
			else:
				char = self.codes.tochar(code)
			text += char
		return text


class Segmenter:
    def __init__(self, file):
        self.sound = AudioSegment.from_wav(file)

    def play(self):
        play(self.sound)




# Move all this into the main() later

# Plot for matplotlib figure
plotter = Plotter("png")

# Create a SoundFile object to get the data from the file
the_file = SoundFile(file)
the_file.saveplot(original_file)


# Filter the signal (maybe filtering the noise)
the_filter = SignalFilter()
the_filter.filter(the_file)


# Find the spikes/peaks for the morse code (or dees)
analyzer = SpectreAnalyzer()
pulses = analyzer.findpulses(the_file)


# Translate the highs/lows to appropriate string characters
pul_translator = PulsesTranslator()
code_string = pul_translator.tostring(pulses)

# Translate the morse code string to ascii text
str_translator = StringTranslator()
s = str_translator.totext(code_string)

# Print out the results
print(code_string)
print(s)
