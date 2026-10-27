import os
import io
import wave
import tempfile
import numpy as np

LIBROSA_AVAILABLE = False
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None

SCIPY_AVAILABLE = False
try:
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    wavfile = None

def get_audio_backend_status() -> dict:
    backend = "None (Demo / Waveform Simulator)"
    if LIBROSA_AVAILABLE:
        backend = "Librosa (Full Audio Processing)"
    elif SCIPY_AVAILABLE:
        backend = "SciPy wavfile (Local WAV Processing)"
    else:
        backend = "Python Standard Library (Wave)"
        
    return {
        "librosa_available": LIBROSA_AVAILABLE,
        "scipy_available": SCIPY_AVAILABLE,
        "active_backend": backend
    }

def compute_rms_curve(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes RMS (Root Mean Square) energy curve across frames using pure NumPy.
    Returns: (times, rms_values)
    """
    if len(y) == 0:
        return np.array([]), np.array([])
        
    # Ensure 1D mono float32 normalized between -1.0 and 1.0
    if y.ndim > 1:
        y = np.mean(y, axis=1)
        
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    num_frames = max(1, (len(y) - frame_length) // hop_length + 1)
    rms_values = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = y[start:end]
        rms_values[i] = np.sqrt(np.mean(frame ** 2))
        
    times = np.arange(num_frames) * (hop_length / 22050.0)
    return times, rms_values

def process_audio_file(file_bytes: bytes, filename: str) -> dict:
    """
    Processes uploaded audio file using Librosa or SciPy fallback.
    Returns signal statistics, waveform data, RMS energy curve, and acoustic activity summary.
    """
    temp_path = None
    suffix = os.path.splitext(filename)[1].lower() if filename else ".wav"
    if not suffix:
        suffix = ".wav"
        
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        y = None
        sr = 22050
        backend_used = ""

        # 1. Try Librosa if available
        if LIBROSA_AVAILABLE:
            try:
                y, sr = librosa.load(temp_path, sr=None, mono=True)
                backend_used = "Librosa"
            except Exception:
                y = None

        # 2. Try SciPy wavfile for WAV files
        if y is None and SCIPY_AVAILABLE and suffix == ".wav":
            try:
                rate, data = wavfile.read(temp_path)
                sr = rate
                if data.dtype == np.int16:
                    y = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    y = data.astype(np.float32) / 2147483648.0
                elif data.dtype == np.uint8:
                    y = (data.astype(np.float32) - 128.0) / 128.0
                else:
                    y = data.astype(np.float32)
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                backend_used = "SciPy wavfile"
            except Exception:
                y = None

        # 3. Fallback to Python wave module for uncompressed WAV
        if y is None and suffix == ".wav":
            try:
                with wave.open(temp_path, 'rb') as wf:
                    sr = wf.getframerate()
                    n_frames = wf.getnframes()
                    channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    raw = wf.readframes(n_frames)
                    if sampwidth == 2:
                        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    else:
                        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                    if channels > 1:
                        data = data.reshape(-1, channels).mean(axis=1)
                    y = data
                    backend_used = "Python Standard Wave Module"
            except Exception:
                y = None

        if y is None:
            return {
                "success": False,
                "error": f"Could not decode audio file '{filename}'. Note: If Librosa is not installed, only standard WAV format files can be processed without external codecs. Please upload a .wav file or use Demo Mode.",
                "stats": {}
            }

        # Normalize y
        peak = np.max(np.abs(y)) if len(y) > 0 else 0
        if peak > 0:
            y_norm = y / peak
        else:
            y_norm = y

        duration_sec = round(len(y) / sr, 2)
        
        # Downsample waveform for plotting if too long (max 2000 points)
        step = max(1, len(y_norm) // 2000)
        downsampled_wave = y_norm[::step]
        wave_times = np.linspace(0, duration_sec, len(downsampled_wave))
        
        # Calculate RMS curve
        frame_len = int(sr * 0.05) # 50ms window
        hop_len = int(sr * 0.025)  # 25ms step
        times_rms, rms_vals = compute_rms_curve(y_norm, frame_length=frame_len, hop_length=hop_len)
        if len(times_rms) > 0:
            times_rms = np.linspace(0, duration_sec, len(rms_vals))

        # Acoustic activity classification
        mean_rms = float(np.mean(rms_vals)) if len(rms_vals) > 0 else 0.0
        max_rms = float(np.max(rms_vals)) if len(rms_vals) > 0 else 0.0
        
        high_thresh = 0.35
        mod_thresh = 0.12
        
        high_pct = float(np.mean(rms_vals >= high_thresh) * 100) if len(rms_vals) > 0 else 0.0
        mod_pct = float(np.mean((rms_vals >= mod_thresh) & (rms_vals < high_thresh)) * 100) if len(rms_vals) > 0 else 0.0
        quiet_pct = float(np.mean(rms_vals < mod_thresh) * 100) if len(rms_vals) > 0 else 0.0

        # Overall activity classification
        if high_pct > 30.0 or mean_rms >= 0.25:
            overall_activity = "High Acoustic Activity (Interactive / Discussion)"
            activity_score = 85.0
        elif quiet_pct > 65.0 or mean_rms < 0.10:
            overall_activity = "Low Acoustic Activity (Quiet / Independent Study)"
            activity_score = 45.0
        else:
            overall_activity = "Moderate Acoustic Activity (Standard Lecture / Speech)"
            activity_score = 70.0

        return {
            "success": True,
            "filename": filename,
            "backend": backend_used,
            "metadata": {
                "sample_rate": sr,
                "duration_seconds": duration_sec,
                "total_samples": len(y),
            },
            "stats": {
                "mean_rms": round(mean_rms, 3),
                "peak_rms": round(max_rms, 3),
                "dynamic_range_db": round(float(20 * np.log10(max_rms / (mean_rms + 1e-6))), 1) if mean_rms > 0 else 0.0,
                "high_activity_pct": round(high_pct, 1),
                "moderate_activity_pct": round(mod_pct, 1),
                "quiet_period_pct": round(quiet_pct, 1),
                "overall_activity": overall_activity,
                "activity_proxy_score": activity_score
            },
            "waveform": {
                "times": wave_times.tolist(),
                "amplitudes": downsampled_wave.tolist()
            },
            "rms_curve": {
                "times": times_rms.tolist() if len(times_rms) > 0 else [],
                "values": rms_vals.tolist() if len(rms_vals) > 0 else []
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Audio processing error: {str(e)}",
            "stats": {}
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

def generate_synthetic_demo_audio() -> dict:
    """Provides a realistic classroom lecture audio profile for immediate demonstration."""
    duration_sec = 60.0
    sr = 22050
    t = np.linspace(0, duration_sec, 2000)
    
    # Simulate a dynamic classroom:
    # 0-20s: Teacher talking (moderate volume)
    # 20-40s: Student Q&A discussion (lively bursts)
    # 40-50s: Quiet thinking / problem solving (low volume)
    # 50-60s: Teacher summary
    wave_amp = np.zeros_like(t)
    wave_amp[:666] = 0.35 * np.sin(2 * np.pi * 3.0 * t[:666]) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.5 * t[:666]))
    wave_amp[666:1333] = 0.70 * np.sin(2 * np.pi * 4.5 * t[666:1333]) * (0.7 + 0.3 * np.cos(2 * np.pi * 1.2 * t[666:1333]))
    wave_amp[1333:1666] = 0.10 * np.sin(2 * np.pi * 1.5 * t[1333:1666])
    wave_amp[1666:] = 0.40 * np.sin(2 * np.pi * 2.8 * t[1666:])
    
    # Add subtle ambient classroom noise
    np.random.seed(42)
    wave_amp += np.random.normal(0, 0.05, len(wave_amp))
    wave_amp = np.clip(wave_amp, -1.0, 1.0)
    
    # RMS curve
    times_rms = np.linspace(0, duration_sec, 120)
    rms_vals = np.zeros_like(times_rms)
    rms_vals[:40] = 0.25 + 0.05 * np.sin(times_rms[:40])
    rms_vals[40:80] = 0.55 + 0.15 * np.sin(times_rms[40:80] * 2)
    rms_vals[80:100] = 0.08 + 0.02 * np.random.rand(20)
    rms_vals[100:] = 0.32 + 0.06 * np.cos(times_rms[100:])
    rms_vals = np.clip(rms_vals, 0.02, 0.85)
    
    return {
        "success": True,
        "is_demo": True,
        "filename": "classroom_lecture_session_demo.wav",
        "backend": "Demo Audio Simulator",
        "metadata": {
            "sample_rate": sr,
            "duration_seconds": duration_sec,
            "total_samples": int(sr * duration_sec),
        },
        "stats": {
            "mean_rms": 0.32,
            "peak_rms": 0.78,
            "dynamic_range_db": 18.5,
            "high_activity_pct": 33.3,
            "moderate_activity_pct": 50.0,
            "quiet_period_pct": 16.7,
            "overall_activity": "Moderate-to-High Activity (Interactive Lecture & Discussion)",
            "activity_proxy_score": 78.0
        },
        "waveform": {
            "times": t.tolist(),
            "amplitudes": wave_amp.tolist()
        },
        "rms_curve": {
            "times": times_rms.tolist(),
            "values": rms_vals.tolist()
        }
    }
