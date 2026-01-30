import numpy as np

def voiced_excitation(duration, f0, fs):
    '''
    Create a voiced excitation signal with the given duration, pitch, and sampling rate.
    
    @param:
    duration (scalar): length of the signal, in samples
    f0 (scalar): fundamental frequency, in Hertz
    fs (scalar): sampling frequency, in samples/second
    
    @returns:
    excitation (np.ndarray): the excitation signal
             The signal should be all zeros, except for a value of -1
             every T0 samples, where T0 = int(round(fs/f0)).
    '''
    # Calculate the pitch period in samples (T0)
    T0 = int(np.round(fs / f0))
    
    # Create an array of zeros with the requested duration
    excitation = np.zeros(duration)
    
    # Set the value to -1 at every T0-th position (0, T0, 2*T0, etc.)
    # We use slicing [::T0] to grab every T0-th element
    excitation[::T0] = -1
    
    return excitation

def resonator(x, f, bw, fs):
    '''
    Generate the output of a resonator filter.
    
    @param:
    x (np.ndarray): the input signal
    f (scalar): the resonant frequency, in Hertz
    bw (scalar): the bandwidth of the resonance, in Hertz
    fs (scalar): the sampling frequency, in samples/second
    
    @returns:
    y (np.ndarray): the output signal
    '''
    # Calculate coefficients A, B, C based on the lecture formulas
    C = -np.exp(-2 * np.pi * bw / fs)
    B = 2 * np.exp(-np.pi * bw / fs) * np.cos(2 * np.pi * f / fs)
    A = 1 - B - C
    
    # Initialize the output array y with the same size as input x
    y = np.zeros(len(x))
    
    # Implement the difference equation: y[n] = A*x[n] + B*y[n-1] + C*y[n-2]
    
    # Handle the first index (n=0)
    y[0] = A * x[0]
    
    # Handle the second index (n=1) if the signal is long enough
    if len(x) > 1:
        y[1] = A * x[1] + B * y[0]
        
    # Loop for the rest of the signal (from index 2 to end)
    for n in range(2, len(x)):
        y[n] = A * x[n] + B * y[n-1] + C * y[n-2]
        
    return y

def synthesize_vowel(duration, f0, F1, F2, F3, F4, BW1, BW2, BW3, BW4, fs):
    '''
    Synthesize a vowel.
    
    @param:
    duration (scalar): duration in samples
    f0 (scalar): fundamental frequency (pitch)
    F1, F2, F3, F4 (scalars): formant frequencies
    BW1, BW2, BW3, BW4 (scalars): formant bandwidths
    fs (scalar): sampling frequency
    
    @returns:
    speech (np.ndarray): the synthesized vowel
    '''
    # 1. Generate the source (buzz)
    exc = voiced_excitation(duration, f0, fs)
    
    # 2. Filter it through 4 resonators in series (cascade)
    # Output of resonator 1 goes into resonator 2, and so on.
    y1 = resonator(exc, F1, BW1, fs)
    y2 = resonator(y1,  F2, BW2, fs)
    y3 = resonator(y2,  F3, BW3, fs)
    y4 = resonator(y3,  F4, BW4, fs)
    
    return y4