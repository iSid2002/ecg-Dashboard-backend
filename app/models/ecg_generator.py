# Copy the ECGDataGenerator class from the provided code
# (The full class implementation will be copied here)

import numpy as np
from scipy import signal
import pywt
from tqdm import tqdm

class ECGDataGenerator:
    """Class to generate synthetic ECG data with both normal and abnormal patterns"""
    
    def __init__(self, sampling_rate=1000, duration=10):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.time = np.linspace(0, duration, sampling_rate * duration)
        
    def generate_normal_ecg(self):
        """Generate a normal ECG signal"""
        # Create basic ECG components
        # P wave
        p_wave = 0.15 * np.sin(2 * np.pi * 5 * self.time)
        
        # QRS complex using combination of sine waves
        qrs = np.zeros_like(self.time)
        for i in range(len(self.time)):
            t = self.time[i]
            if 0 <= t % 1 < 0.1:  # QRS duration
                qrs[i] = 1.0 * np.sin(2 * np.pi * 20 * t) * np.exp(-50 * (t % 1))
        
        # T wave
        t_wave = 0.3 * np.sin(2 * np.pi * 3 * self.time)
        
        # Combine components
        ecg = p_wave + qrs + t_wave
        
        # Add baseline wander
        baseline = 0.1 * np.sin(2 * np.pi * 0.5 * self.time)
        
        # Add noise
        noise = 0.05 * np.random.normal(0, 1, len(self.time))
        
        # Combine everything
        ecg = ecg + baseline + noise
        
        return ecg
    
    def generate_abnormal_ecg(self, abnormality_type=None):
        """Generate an abnormal ECG signal with specified abnormality type"""
        # Get normal ECG as base
        ecg = self.generate_normal_ecg()
        
        if abnormality_type is None:
            abnormality_type = np.random.choice([
                'st_elevation', 'st_depression', 'tachycardia',
                'bradycardia', 'afib', 'pvc', 'long_qt'
            ])
        
        if abnormality_type == 'st_elevation':
            # Add ST elevation
            st_segment = np.where((self.time % 1 >= 0.1) & (self.time % 1 < 0.3))[0]
            ecg[st_segment] += 0.5
            
        elif abnormality_type == 'st_depression':
            # Add ST depression
            st_segment = np.where((self.time % 1 >= 0.1) & (self.time % 1 < 0.3))[0]
            ecg[st_segment] -= 0.5
            
        elif abnormality_type == 'tachycardia':
            # Increase heart rate
            ecg = signal.resample(ecg, int(len(ecg) * 0.7))
            ecg = signal.resample(ecg, len(self.time))
            
        elif abnormality_type == 'bradycardia':
            # Decrease heart rate
            ecg = signal.resample(ecg, int(len(ecg) * 1.3))
            ecg = signal.resample(ecg, len(self.time))
            
        elif abnormality_type == 'afib':
            # Add atrial fibrillation
            afib = 0.2 * np.random.normal(0, 1, len(self.time))
            afib = signal.filtfilt(*signal.butter(4, 0.1), afib)
            ecg += afib
            
        elif abnormality_type == 'pvc':
            # Add premature ventricular contraction
            pvc_indices = np.random.choice(len(self.time), size=3, replace=False)
            for idx in pvc_indices:
                ecg[idx:idx+100] += 1.5
            
        elif abnormality_type == 'long_qt':
            # Prolong QT interval
            qt_segment = np.where((self.time % 1 >= 0.1) & (self.time % 1 < 0.4))[0]
            ecg[qt_segment] *= 1.2
        
        return ecg
    
    def generate_dataset(self, n_samples, abnormality_ratio=0.3):
        """Generate a synthetic dataset with both normal and abnormal ECGs"""
        X = []
        y = []
        
        # Ensure we have at least one sample of each class
        n_abnormal = max(1, int(n_samples * abnormality_ratio))
        n_normal = max(1, n_samples - n_abnormal)
        
        # Generate normal ECGs
        for _ in range(n_normal):
            ecg = self.generate_normal_ecg()
            X.append(ecg)
            y.append(0)  # Normal
        
        # Generate abnormal ECGs
        abnormality_types = ['st_elevation', 'st_depression', 'tachycardia', 
                           'bradycardia', 'afib', 'pvc', 'long_qt']
        for _ in range(n_abnormal):
            abnormality_type = np.random.choice(abnormality_types)
            ecg = self.generate_abnormal_ecg(abnormality_type)
            X.append(ecg)
            y.append(1)  # Abnormal
        
        # Shuffle the dataset
        indices = np.random.permutation(len(X))
        X = np.array(X)[indices]
        y = np.array(y)[indices]
        
        return X, y
