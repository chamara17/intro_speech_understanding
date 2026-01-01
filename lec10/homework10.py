import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def get_features(waveform, Fs):
    '''
    Get features from a waveform.
    @params:
    waveform (numpy array) - the waveform
    Fs (scalar) - sampling frequency.

    @return:
    features (NFRAMES,NFEATS) - numpy array of feature vectors:
        Pre-emphasize the signal, then compute the spectrogram with a 4ms frame length and 2ms step,
        then keep only the low-frequency half (the non-aliased half).
    labels (NFRAMES) - numpy array of labels (integers):
        Calculate VAD with a 25ms window and 10ms skip. Find start time and end time of each segment.
        Then give every non-silent segment a different label.  Repeat each label five times.
    '''
    # --- 1. FEATURE EXTRACTION ---
    # Parameters
    frame_len = int(0.004 * Fs)
    step = int(0.002 * Fs)
    
    # Pre-emphasis
    pre_emph = np.diff(waveform)
    
    # Create frames
    num_frames = int((len(pre_emph) - frame_len) / step) + 1
    frames = np.zeros((num_frames, frame_len))
    for i in range(num_frames):
        start = i * step
        frames[i, :] = pre_emph[start : start + frame_len]
        
    # Spectrogram -> Magnitude -> Low freq half
    spec = np.abs(np.fft.fft(frames, axis=1))
    nfeats = int(frame_len / 2)
    features = spec[:, :nfeats]
    
    # Log magnitude (standard for ASR)
    features = np.log(features + 1e-10)

    # --- 2. LABEL GENERATION (VAD) ---
    # VAD Parameters
    vad_len = int(0.025 * Fs)
    vad_step = int(0.01 * Fs)
    
    # Calculate Energy
    num_vad = int((len(waveform) - vad_len) / vad_step) + 1
    energies = np.zeros(num_vad)
    for i in range(num_vad):
        start = i * vad_step
        seg = waveform[start : start + vad_len]
        energies[i] = np.sum(seg**2)
        
    # Thresholding (10% of max)
    threshold = 0.1 * np.amax(energies)
    vad_mask = (energies > threshold).astype(int)
    
    # Assign Segment IDs (1, 2, 3...)
    # We create a label array where each separate speech segment gets a unique integer
    vad_labels = np.zeros(num_vad, dtype=int)
    current_label = 0
    in_segment = False
    
    for i in range(num_vad):
        if vad_mask[i] == 1:
            if not in_segment:
                current_label += 1
                in_segment = True
            vad_labels[i] = current_label
        else:
            in_segment = False
            
    # Upsample labels (Repeat 5 times)
    # Because VAD step (10ms) is 5x slower than Feature step (2ms)
    labels = np.repeat(vad_labels, 5)
    
    # --- 3. ALIGN LENGTHS ---
    # Truncate to the shorter length to ensure they match exactly
    min_len = min(len(features), len(labels))
    features = features[:min_len]
    labels = labels[:min_len]
    
    return features, labels

def train_neuralnet(features, labels, iterations):
    '''
    @param:
    features (NFRAMES,NFEATS) - numpy array
    labels (NFRAMES) - numpy array
    iterations (scalar) - number of iterations of training

    @return:
    model - a neural net model created in pytorch, and trained using the provided data
    lossvalues (numpy array, length=iterations) - the loss value achieved on each iteration
    '''
    # Convert to PyTorch tensors
    # Features must be Float, Labels must be Long (integers)
    X = torch.tensor(features, dtype=torch.float32)
    Y = torch.tensor(labels, dtype=torch.long)
    
    input_dim = features.shape[1]
    # Output dim is max label + 1 (to account for 0 index)
    output_dim = int(np.max(labels)) + 1
    
    # Define Model: LayerNorm -> Linear
    model = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, output_dim)
    )
    
    # Loss and Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    lossvalues = []
    
    # Training Loop
    for _ in range(iterations):
        # Forward
        preds = model(X)
        loss = loss_fn(preds, Y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        lossvalues.append(loss.item())
        
    return model, np.array(lossvalues)

def test_neuralnet(model, features):
    '''
    @param:
    model - a neural net model created in pytorch, and trained
    features (NFRAMES, NFEATS) - numpy array
    @return:
    probabilities (NFRAMES, NLABELS) - model output, transformed by softmax, detach().numpy().
    '''
    # Convert to Tensor
    X = torch.tensor(features, dtype=torch.float32)
    
    # Forward Pass
    logits = model(X)
    
    # Softmax to get probabilities
    # dim=1 is the class dimension
    probs = nn.functional.softmax(logits, dim=1)
    
    return probs.detach().numpy()