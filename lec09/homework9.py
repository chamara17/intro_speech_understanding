import numpy as np

def VAD(waveform, Fs):
    '''
    Extract the segments that have energy greater than 10% of maximum.
    Calculate the energy in frames that have 25ms frame length and 10ms frame step.
     
    @params:
    waveform (np.ndarray(N)) - the waveform
    Fs (scalar) - sampling rate
     
    @returns:
    segments (list of arrays) - list of the waveform segments where energy is 
        greater than 10% of maximum energy
    '''
    # Frame parameters
    frame_len = int(0.025 * Fs)
    step = int(0.01 * Fs)
    
    # 1. Calculate Short-Time Energy
    num_frames = int((len(waveform) - frame_len) / step) + 1
    energies = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * step
        stop = start + frame_len
        # Energy = sum of squares
        energies[i] = np.sum(waveform[start:stop]**2)
        
    # 2. Thresholding (10% of max energy)
    threshold = 0.1 * np.amax(energies)
    is_speech_frame = (energies > threshold).astype(int)
    
    # 3. Find boundaries (where it switches from 0 to 1 or 1 to 0)
    # We pad with 0 so we can detect starts at index 0 and ends at the last index
    padded = np.pad(is_speech_frame, (1, 1), 'constant')
    diff = np.diff(padded)
    
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    # 4. Extract segments
    segments = []
    for s, e in zip(starts, ends):
        # Convert frame indices to sample indices
        start_sample = s * step
        # Simple approximation for end sample:
        end_sample = (e * step) + (frame_len - step)
        
        # Ensure we don't go out of bounds
        end_sample = min(end_sample, len(waveform))
        
        segments.append(waveform[start_sample:end_sample])
        
    return segments

def segments_to_models(segments, Fs):
    '''
    Create a model spectrum from each segment:
    Pre-emphasize each segment, then calculate its spectrogram with 4ms frame length and 2ms step,
    then keep only the low-frequency half of each spectrum, then average the low-frequency spectra
    to make the model.
     
    @params:
    segments (list of arrays) - waveform segments that contain speech
    Fs (scalar) - sampling rate
     
    @returns:
    models (list of arrays) - average log spectra of pre-emphasized waveform segments
    '''
    models = []
    
    # Frame parameters for model creation (smaller windows)
    frame_len = int(0.004 * Fs)
    step = int(0.002 * Fs)
    
    for seg in segments:
        # 1. Pre-emphasis: x[n] = x[n] - x[n-1]
        pre_emph = np.diff(seg)
        
        # 2. Chop into frames
        num_frames = int((len(pre_emph) - frame_len) / step) + 1
        # If segment is too short, skip or handle (assuming segments are long enough here)
        if num_frames < 1:
            continue
            
        frames = np.zeros((num_frames, frame_len))
        for i in range(num_frames):
            start = i * step
            frames[i, :] = pre_emph[start : start + frame_len]
            
        # 3. Compute Magnitude Spectrum
        fft_res = np.fft.fft(frames, axis=1)
        mag_spec = np.abs(fft_res)
        
        # 4. Keep low-frequency half
        half_len = int(frame_len / 2)
        half_spec = mag_spec[:, :half_len]
        
        # 5. Log and Average
        # Add tiny epsilon to avoid log(0)
        log_spec = np.log(half_spec + 1e-10)
        avg_model = np.mean(log_spec, axis=0)
        
        models.append(avg_model)
        
    return models

def recognize_speech(testspeech, Fs, models, labels):
    '''
    Chop the testspeech into segments using VAD, convert it to models using segments_to_models,
    then compare each test segment to each model using cosine similarity,
    and output the label of the most similar model to each test segment.
     
    @params:
    testspeech (array) - test waveform
    Fs (scalar) - sampling rate
    models (list of Y arrays) - list of model spectra
    labels (list of Y strings) - one label for each model
     
    @returns:
    sims (Y-by-K array) - cosine similarity of each model to each test segment
    test_outputs (list of strings) - recognized label of each test segment
    '''
    # 1. Process Test Speech
    test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs)
    
    num_train = len(models)
    num_test = len(test_models)
    
    sims = np.zeros((num_train, num_test))
    test_outputs = []
    
    # 2. Compare Vectors
    for j in range(num_test):
        best_score = -np.inf
        best_label = None
        
        vec_test = test_models[j]
        norm_test = np.linalg.norm(vec_test)
        
        for i in range(num_train):
            vec_train = models[i]
            norm_train = np.linalg.norm(vec_train)
            
            # Cosine Similarity
            score = np.dot(vec_train, vec_test) / (norm_train * norm_test)
            sims[i, j] = score
            
            if score > best_score:
                best_score = score
                best_label = labels[i]
        
        test_outputs.append(best_label)
        
    return sims, test_outputs