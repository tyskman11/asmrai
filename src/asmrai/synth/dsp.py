"""Prozedurale Trigger-Klaenge (Tapping, Brushing, Scratching, Atem, Knistern) und binaurale Verraeumlichung.

Alles ist reine Signalsynthese mit numpy/scipy - es werden keine Aufnahmen eines Creators wiederverwendet.
Die Parameter (Tempo, Helligkeit, Pegel, Breite) stammen aus dem Stil-Profil.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

SR = 44100

# Material -> (Grundfrequenzen Hz, Abklingzeit s, Klickanteil)
MATERIALS = {
    "wood": ([420, 910, 1650], 0.045, 0.5), "plastic": ([1100, 2300, 3900], 0.020, 0.7),
    "glass": ([2600, 5200, 7400], 0.090, 0.4), "nails": ([3400, 6100, 9000], 0.012, 0.9),
    "metal": ([1800, 3900, 6700], 0.140, 0.3), "skin": ([220, 520], 0.030, 0.3),
}


def _rng(seed: int) -> np.random.Generator: return np.random.default_rng(seed)


def _fade(x: np.ndarray, ms: float = 8) -> np.ndarray:
    n = min(int(SR * ms / 1000), len(x) // 2)
    if n > 0:
        ramp = np.linspace(0, 1, n); x[:n] *= ramp; x[-n:] *= ramp[::-1]
    return x


def _bandpass(x: np.ndarray, low: float, high: float, order: int = 2) -> np.ndarray:
    high = min(high, SR / 2 - 100); low = max(low, 20)
    sos = signal.butter(order, [low, high], btype="band", fs=SR, output="sos"); return signal.sosfilt(sos, x)


def _lowpass(x: np.ndarray, freq: float, order: int = 2) -> np.ndarray:
    return signal.sosfilt(signal.butter(order, min(freq, SR / 2 - 100), btype="low", fs=SR, output="sos"), x)


def _highpass(x: np.ndarray, freq: float, order: int = 2) -> np.ndarray:
    return signal.sosfilt(signal.butter(order, freq, btype="high", fs=SR, output="sos"), x)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white); freqs = np.fft.rfftfreq(n); freqs[0] = freqs[1]
    out = np.fft.irfft(spectrum / np.sqrt(freqs), n)
    return out / (np.max(np.abs(out)) + 1e-9)


def room_tone(duration: float, level_db: float, seed: int) -> np.ndarray:
    n = int(duration * SR); noise = _lowpass(pink_noise(n, _rng(seed)), 3000, 1)
    return noise / (np.sqrt(np.mean(noise ** 2)) + 1e-9) * 10 ** (level_db / 20)


def tap(rng: np.random.Generator, material: str, strength: float = 1.0, pitch: float = 1.0, decay_scale: float = 1.0) -> np.ndarray:
    freqs, decay, click = MATERIALS[material]; decay *= decay_scale
    n = int(SR * decay * 6); t = np.arange(n) / SR
    body = sum(rng.uniform(0.5, 1.0) / (i + 1) * np.sin(2 * np.pi * f * pitch * rng.uniform(0.98, 1.02) * t) * np.exp(-t / (decay / (1 + i * 0.5)))
               for i, f in enumerate(freqs))
    burst = _highpass(rng.standard_normal(n), 1500) * np.exp(-t / 0.003) * click
    hit = (body + burst)
    return _fade(hit / (np.max(np.abs(hit)) + 1e-9) * strength, 2)


def tapping(duration: float, rate: float, material: str, brightness: float, seed: int, jitter: float = 0.25, decay_scale: float = 1.0) -> np.ndarray:
    """Taps mit leichter Humanisierung, Akzenten und gelegentlichen Doppel-/Dreifachschlaegen."""
    rng = _rng(seed); out = np.zeros(int(duration * SR)); t = rng.uniform(0.05, 0.4); interval = 1 / max(rate, 0.4)
    while t < duration - 0.1:
        pattern = rng.choice([1, 1, 1, 2, 3], p=[0.4, 0.2, 0.1, 0.2, 0.1])
        for k in range(pattern):
            start = int((t + k * interval * 0.22) * SR); hit = tap(rng, material, strength=rng.uniform(0.5, 1.0) * (1 if k == 0 else 0.7), pitch=brightness * rng.uniform(0.95, 1.05), decay_scale=decay_scale)
            end = min(len(out), start + len(hit))
            if start < len(out): out[start:end] += hit[:end - start]
        t += interval * rng.uniform(1 - jitter, 1 + jitter * 1.4) * (1 + 0.3 * pattern)
    return out


def stroke_envelope(n: int, rng: np.random.Generator, min_s: float, max_s: float, gap: float = 0.15) -> np.ndarray:
    env = np.zeros(n); pos = 0
    while pos < n:
        length = int(rng.uniform(min_s, max_s) * SR); end = min(n, pos + length)
        env[pos:end] = np.hanning(end - pos) * rng.uniform(0.5, 1.0)
        pos = end + int(rng.uniform(0, gap) * SR)
    return env


def brushing(duration: float, centroid: float, seed: int) -> np.ndarray:
    """Mikro-Buerstengeraeusch: schmalbandiges Rauschen mit Strich-Huellkurve und feiner Borsten-Textur."""
    rng = _rng(seed); n = int(duration * SR)
    base = _bandpass(rng.standard_normal(n), max(300, centroid * 0.45), centroid * 2.2)
    grain = (rng.random(n) < 0.015).astype(float) * rng.standard_normal(n)
    grain = _bandpass(grain, 1500, 9000) * 2.0
    return (base + grain * 0.6) * stroke_envelope(n, rng, 0.6, 1.8)


def scratching(duration: float, centroid: float, seed: int) -> np.ndarray:
    rng = _rng(seed); n = int(duration * SR)
    body = _bandpass(rng.standard_normal(n), max(900, centroid * 0.8), min(12000, centroid * 3.0))
    modulation = 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(18, 40) * np.arange(n) / SR)   # feine Kratzer-Textur
    return body * modulation * stroke_envelope(n, rng, 0.25, 0.8, 0.1)


def breathing(duration: float, seed: int) -> np.ndarray:
    rng = _rng(seed); n = int(duration * SR); noise = _bandpass(rng.standard_normal(n), 250, 2600, 2)
    env = np.zeros(n); pos = 0
    while pos < n:
        length = int(rng.uniform(1.4, 2.8) * SR); end = min(n, pos + length)
        env[pos:end] = np.sin(np.linspace(0, np.pi, end - pos)) ** 1.5 * rng.uniform(0.5, 1.0); pos = end + int(rng.uniform(0.2, 0.9) * SR)
    return noise * env


def crinkle(duration: float, density: float, seed: int) -> np.ndarray:
    rng = _rng(seed); n = int(duration * SR); out = np.zeros(n)
    cluster = 0
    for i in np.sort(rng.integers(0, n, int(duration * density * 18))):
        if rng.random() < 0.25: cluster = int(rng.integers(3, 12))
        length = int(rng.uniform(0.002, 0.012) * SR); end = min(n, i + length)
        out[i:end] += _highpass(rng.standard_normal(end - i), 2500) * np.exp(-np.arange(end - i) / (length / 3)) * rng.uniform(0.2, 1.0)
        if cluster: cluster -= 1
    return out


def make_trigger(label: str, duration: float, params: dict, seed: int) -> np.ndarray:
    """Erzeugt ein Mono-Signal fuer ein Segment. params kommt aus dem Stil-Profil (+ Session-Vorgaben)."""
    centroid = float(params.get("centroid", 2200)); rate = float(params.get("onset_rate", 2.0))
    if label == "tapping":
        material = params.get("material") or "wood"; recorded = float(params.get("decay_ms") or 0)
        decay_scale = float(np.clip(recorded / (MATERIALS[material][1] * 1000 * 1.15), 0.5, 2.5)) if recorded > 0 else 1.0     # 10-dB-Abklingzeit = 1,15 x Zeitkonstante                  # Abklingzeit wie im Original
        jitter = float(np.clip(float(params.get("rate_cv", 0.6)) * 0.4, 0.08, 0.5))                                            # Regelmaessigkeit wie im Original
        return tapping(duration, max(rate, 1.0), material, max(0.7, min(1.6, centroid / 2200)), seed, jitter, decay_scale)
    if label == "brushing": return brushing(duration, centroid, seed)
    if label == "scratching": return scratching(duration, centroid, seed)
    if label == "breath": return breathing(duration, seed)
    if label == "crinkle": return crinkle(duration, max(rate, 1.0), seed)
    return np.zeros(int(duration * SR))


# ----------------------------------------------------------------------------------------------
# Binaurale Verraeumlichung (vereinfachtes Kugelkopf-Modell: Woodward-ITD + Kopfschatten-ILD)

HEAD_RADIUS, SOUND_SPEED = 0.0875, 343.0


def spatialize(mono: np.ndarray, azimuth: np.ndarray | float, distance: float = 0.3) -> np.ndarray:
    """azimuth in Grad (-90 links ... +90 rechts), entweder konstant oder pro Sample. -> (n, 2)"""
    n = len(mono)
    az = np.full(n, float(azimuth)) if np.isscalar(azimuth) else np.interp(np.arange(n), np.linspace(0, n - 1, len(azimuth)), azimuth)
    theta = np.deg2rad(np.clip(az, -90, 90))
    itd = HEAD_RADIUS / SOUND_SPEED * (np.abs(theta) + np.sin(np.abs(theta)))            # Sekunden Vorlauf fuers nahe Ohr
    delay_far = itd * SR
    index = np.arange(n)
    far = np.interp(index - delay_far, index, mono, left=0.0)                           # fraktionale Verzoegerung
    shadow = np.abs(np.sin(theta))                                                       # 0 = frontal, 1 = seitlich
    far_dull = _lowpass(far, 2200, 1)
    far = far * (1 - shadow * 0.8) + far_dull * shadow * 0.8 * 10 ** (-3 * shadow / 20)
    near = mono * (1 + 0.25 * shadow)
    proximity = 1.0 / max(distance / 0.3, 0.25) ** 0.6
    right_is_near = theta >= 0
    left = np.where(right_is_near, far, near); right = np.where(right_is_near, near, far)
    return np.stack([left, right], axis=1) * proximity


def azimuth_path(duration: float, mode: str, rng: np.random.Generator, start: float = 0.0) -> np.ndarray:
    """Bewegungsbahnen fuer die Schallquelle. Rueckgabe: Azimut-Kontrollpunkte (10 pro Sekunde)."""
    n = max(2, int(duration * 10)); t = np.linspace(0, duration, n)
    if mode == "left": return np.full(n, -70.0)
    if mode == "right": return np.full(n, 70.0)
    if mode == "center": return np.full(n, 0.0)
    if mode == "sweep": return np.linspace(-75, 75, n) * (1 if rng.random() < 0.5 else -1)
    if mode == "orbit": return 75 * np.sin(2 * np.pi * t / max(duration / rng.uniform(1, 3), 4) + rng.uniform(0, 6))
    if mode == "wander":
        walk = np.cumsum(rng.standard_normal(n)) * 6 + start
        return np.clip(walk, -80, 80)
    return np.full(n, start)


def whisperize(voice: np.ndarray, sr: int, amount: float = 1.0, seed: int = 0) -> np.ndarray:
    """Wandelt Sprache in Fluestern um: Spektralhuellkurve bleibt, harmonische Anregung wird durch Rauschen ersetzt."""
    from scipy.ndimage import uniform_filter1d
    rng = _rng(seed); n_fft, hop = 1024, 256
    f, _, Z = signal.stft(voice, sr, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.abs(Z)
    envelope = uniform_filter1d(mag, size=14, axis=0)            # ~350 Hz Glaettung entfernt die Obertonstruktur
    phase = np.exp(1j * rng.uniform(0, 2 * np.pi, Z.shape))
    _, whisper = signal.istft(envelope * phase, sr, nperseg=n_fft, noverlap=n_fft - hop)
    whisper = whisper[:len(voice)]
    whisper *= np.sqrt(np.mean(voice ** 2)) / (np.sqrt(np.mean(whisper ** 2)) + 1e-9)
    return amount * whisper + (1 - amount) * voice[:len(whisper)]


def relax(voice: np.ndarray, sr: int, amount: float = 0.5) -> np.ndarray:
    """Macht eine Stimme entspannter: leicht tiefer (bis -1,6 Halbtoene), weichere Hoehen, sanftere Spitzen. amount 0..1."""
    amount = float(max(0.0, min(1.0, amount)))
    if amount < 0.02 or len(voice) < sr // 5: return voice
    import librosa
    y = voice.astype(np.float32)
    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=-1.6 * amount)
    soft = signal.sosfilt(signal.butter(1, 6500, btype="low", fs=sr, output="sos"), y)
    y = (1 - 0.4 * amount) * y + 0.4 * amount * soft
    # Sanfte Dynamik: langsamer Kompressor (Huellkurve ~20 ms) statt Verzerrung - Spitzen werden leiser, die Wellenform bleibt unverzerrt
    from scipy.ndimage import uniform_filter1d
    original_peak = float(np.max(np.abs(voice))) + 1e-9
    envelope = uniform_filter1d(np.abs(y), size=max(1, int(0.02 * sr)))
    threshold, ratio = 0.45 * original_peak, 1.5 + 2.5 * amount
    gain = np.where(envelope > threshold, (np.maximum(envelope, 1e-9) / threshold) ** (1 / ratio - 1), 1.0)
    y = y * uniform_filter1d(gain, size=max(1, int(0.01 * sr)))
    peak = float(np.max(np.abs(y)))
    if peak > original_peak: y = y * (original_peak / peak)
    return y.astype(np.float32)


def resample(x: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr: return x
    return signal.resample_poly(x, to_sr, from_sr)


def rms_db(x: np.ndarray) -> float: return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9))


def normalize_rms(x: np.ndarray, target_db: float, peak_db: float = -1.0) -> np.ndarray:
    gain = 10 ** ((target_db - rms_db(x)) / 20)
    out = x * gain; peak = np.max(np.abs(out)) + 1e-9
    limit = 10 ** (peak_db / 20)
    return out * (limit / peak) if peak > limit else out
