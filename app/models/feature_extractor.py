import numpy as np
from scipy import signal
import pywt

class ECGFeatureExtractor:
    """Class to extract features from ECG signals"""
    
    def __init__(self, sampling_rate=250):
        self.sampling_rate = sampling_rate
    
    def extract_features(self, ecg):
        """Extract time domain, frequency domain, and wavelet features from an ECG signal"""
        features = {}
        
        # Time domain features
        features.update(self._extract_time_domain_features(ecg))
        
        # Frequency domain features
        features.update(self._extract_frequency_domain_features(ecg))
        
        # Wavelet features
        features.update(self._extract_wavelet_features(ecg))
        
        return features
    
    def _extract_time_domain_features(self, ecg):
        """Extract time domain features from an ECG signal"""
        features = {}
        
        # Basic statistical features
        features['mean'] = np.mean(ecg)
        features['std'] = np.std(ecg)
        features['var'] = np.var(ecg)
        features['skew'] = self._skewness(ecg)
        features['kurtosis'] = self._kurtosis(ecg)
        features['rms'] = np.sqrt(np.mean(np.square(ecg)))
        
        # Peak detection
        peaks, _ = signal.find_peaks(ecg, height=0.5, distance=self.sampling_rate*0.5)
        
        if len(peaks) > 1:
            # Heart rate
            rr_intervals = np.diff(peaks) / self.sampling_rate  # in seconds
            features['heart_rate'] = 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
            
            # Heart rate variability
            features['hrv_std'] = np.std(rr_intervals) if len(rr_intervals) > 0 else 0
            features['hrv_rmssd'] = np.sqrt(np.mean(np.square(np.diff(rr_intervals)))) if len(rr_intervals) > 1 else 0
        else:
            features['heart_rate'] = 0
            features['hrv_std'] = 0
            features['hrv_rmssd'] = 0
            
        # QRS detection and features
        qrs_features = self._detect_qrs_features(ecg)
        features.update(qrs_features)
        
        return features
    
    def _extract_frequency_domain_features(self, ecg):
        """Extract frequency domain features from an ECG signal"""
        features = {}
        
        # Compute FFT
        fft_vals = np.abs(np.fft.rfft(ecg))
        fft_freq = np.fft.rfftfreq(len(ecg), 1/self.sampling_rate)
        
        # Power in different frequency bands
        vlf_power = np.sum(fft_vals[(fft_freq >= 0.0033) & (fft_freq < 0.04)]**2)
        lf_power = np.sum(fft_vals[(fft_freq >= 0.04) & (fft_freq < 0.15)]**2)
        hf_power = np.sum(fft_vals[(fft_freq >= 0.15) & (fft_freq < 0.4)]**2)
        total_power = vlf_power + lf_power + hf_power
        
        features['vlf_power'] = vlf_power
        features['lf_power'] = lf_power
        features['hf_power'] = hf_power
        features['lf_hf_ratio'] = lf_power / hf_power if hf_power > 0 else 0
        features['vlf_total_ratio'] = vlf_power / total_power if total_power > 0 else 0
        features['lf_total_ratio'] = lf_power / total_power if total_power > 0 else 0
        features['hf_total_ratio'] = hf_power / total_power if total_power > 0 else 0
        
        # Spectral entropy
        if np.sum(fft_vals) > 0:
            p = fft_vals / np.sum(fft_vals)
            features['spectral_entropy'] = -np.sum(p * np.log2(p + 1e-10))
        else:
            features['spectral_entropy'] = 0
        
        # Dominant frequency
        if len(fft_vals) > 0:
            features['dominant_frequency'] = fft_freq[np.argmax(fft_vals)]
        else:
            features['dominant_frequency'] = 0
            
        return features
    
    def _extract_wavelet_features(self, ecg):
        """Extract wavelet transform features from an ECG signal"""
        features = {}
        
        # Perform wavelet decomposition (5 levels)
        coeffs = pywt.wavedec(ecg, 'db4', level=5)
        
        # Extract features from each decomposition level
        for i, coef in enumerate(coeffs):
            if i == 0:
                level_name = 'approx'
            else:
                level_name = f'detail_{i}'
                
            features[f'{level_name}_mean'] = np.mean(coef)
            features[f'{level_name}_std'] = np.std(coef)
            features[f'{level_name}_energy'] = np.sum(coef**2)
            features[f'{level_name}_entropy'] = self._shannon_entropy(coef)
            
        return features
    
    def _detect_qrs_features(self, ecg):
        """Detect QRS complexes and extract related features"""
        features = {}
        
        # Find R peaks
        r_peaks, _ = signal.find_peaks(ecg, height=0.5, distance=self.sampling_rate*0.5)
        
        if len(r_peaks) < 2:
            features['qrs_duration'] = 0
            features['qt_interval'] = 0
            features['st_segment'] = 0
            features['t_wave_amplitude'] = 0
            features['p_wave_amplitude'] = 0
            return features
        
        # Estimate QRS duration (simplified)
        qrs_durations = []
        qt_intervals = []
        st_segments = []
        t_wave_amplitudes = []
        p_wave_amplitudes = []
        
        for peak_idx in r_peaks:
            if peak_idx < 50 or peak_idx > len(ecg) - 100:
                continue
                
            # QRS detection (simplified)
            # Find Q point (minimum before R)
            q_idx = peak_idx - np.argmin(ecg[peak_idx-50:peak_idx][::-1])
            
            # Find S point (minimum after R)
            s_idx = peak_idx + np.argmin(ecg[peak_idx:peak_idx+50])
            
            qrs_duration = (s_idx - q_idx) / self.sampling_rate * 1000  # in ms
            qrs_durations.append(qrs_duration)
            
            # T wave detection (simplified)
            if s_idx + 100 < len(ecg):
                t_idx = s_idx + np.argmax(ecg[s_idx:s_idx+100])
                t_wave_amplitude = ecg[t_idx]
                t_wave_amplitudes.append(t_wave_amplitude)
                
                # ST segment
                st_segment = np.mean(ecg[s_idx:t_idx])
                st_segments.append(st_segment)
                
                # QT interval
                qt_interval = (t_idx - q_idx) / self.sampling_rate * 1000  # in ms
                qt_intervals.append(qt_interval)
            
            # P wave detection (simplified)
            if q_idx > 50:
                p_idx = q_idx - np.argmax(ecg[q_idx-50:q_idx][::-1])
                p_wave_amplitude = ecg[p_idx]
                p_wave_amplitudes.append(p_wave_amplitude)
        
        # Calculate mean values
        features['qrs_duration'] = np.mean(qrs_durations) if qrs_durations else 0
        features['qt_interval'] = np.mean(qt_intervals) if qt_intervals else 0
        features['st_segment'] = np.mean(st_segments) if st_segments else 0
        features['t_wave_amplitude'] = np.mean(t_wave_amplitudes) if t_wave_amplitudes else 0
        features['p_wave_amplitude'] = np.mean(p_wave_amplitudes) if p_wave_amplitudes else 0
        
        return features
    
    def _skewness(self, data):
        """Calculate skewness of a signal"""
        n = len(data)
        if n == 0:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.sum(((data - mean)/std)**3) / n
    
    def _kurtosis(self, data):
        """Calculate kurtosis of a signal"""
        n = len(data)
        if n == 0:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.sum(((data - mean)/std)**4) / n - 3
    
    def _shannon_entropy(self, data):
        """Calculate Shannon entropy of a signal"""
        # Create histogram
        hist, _ = np.histogram(data, bins=20)
        # Normalize
        hist = hist / np.sum(hist)
        # Calculate entropy
        return -np.sum(hist * np.log2(hist + 1e-10))
