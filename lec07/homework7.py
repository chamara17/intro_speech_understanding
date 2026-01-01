import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord, based at frequency f, with sampling frequency Fs.

    @param:
    f (scalar): frequency of the root tone, in Hertz
    Fs (scalar): sampling frequency, in samples/second

    @return:
    x (array): a one-half-second waveform containing the chord
     
    A major chord is three notes, played at the same time:
    (1) The root tone (f)
    (2) A major third, i.e., four semitones above f
    (3) A major fifth, i.e., seven semitones above f
    '''
    # 1. Calculate the three frequencies
    # Root frequency is given as f
    f_root = f
    # Major third is 4 semitones up: f * 2^(4/12)
    f_third = f * (2 ** (4/12))
    # Major fifth is 7 semitones up: f * 2^(7/12)
    f_fifth = f * (2 ** (7/12))
    
    # 2. Create the time array
    # Duration is 0.5 seconds, so N = 0.5 * Fs
    N = int(0.5 * Fs)
    n = np.arange(N)
    
    # 3. Generate the three cosine waves and sum them up
    # Wave formula: cos(2 * pi * frequency * n / Fs)
    wave1 = np.cos(2 * np.pi * f_root * n / Fs)
    wave2 = np.cos(2 * np.pi * f_third * n / Fs)
    wave3 = np.cos(2 * np.pi * f_fifth * n / Fs)
    
    x = wave1 + wave2 + wave3
    return x

def dft_matrix(N):
    '''
    Create a DFT transform matrix, W, of size N.
     
    @param:
    N (scalar): number of columns in the transform matrix
     
    @result:
    W (NxN array): a matrix of dtype='complex' whose (k,n)^th element is:
           W[k,n] = cos(2*np.pi*k*n/N) - j*sin(2*np.pi*k*n/N)
    '''
    # Create grid indices k (rows) and n (columns)
    k, n = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    
    # Compute the matrix using the complex exponential formula:
    # exp(-j * 2 * pi * k * n / N)
    # The '1j' ensures the result is a complex number
    W = np.exp(-1j * 2 * np.pi * k * n / N)
    
    return W

def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.

    @param:
    x (array): the waveform
    Fs (scalar): sampling frequency (samples/second)

    @return:
    f1, f2, f3: The three loudest frequencies (in Hertz)
      These should be sorted so f1 < f2 < f3.
    '''
    # 1. Compute the FFT magnitude
    X = np.fft.fft(x)
    magnitude = np.abs(X)
    
    # 2. Use only the first half of the spectrum (0 to N/2)
    # because the second half is just a mirror image for real signals.
    N = len(x)
    magnitude = magnitude[:int(N/2)]
    
    # 3. Find the indices of the largest peaks
    # argsort sorts low-to-high, so [-3:] gives the 3 largest indices
    peak_indices = np.argsort(magnitude)[-3:]
    
    # 4. Convert indices to frequencies in Hertz
    # Frequency = index * Fs / N
    frequencies = peak_indices * Fs / N
    
    # 5. Sort the frequencies from low to high
    frequencies = np.sort(frequencies)
    
    return frequencies[0], frequencies[1], frequencies[2]