"""Synthetic narration fixture: no recorded voices."""
import numpy as np
from scipy import signal

RATE = 48000


def narration(seconds=40, seed=7, speech_peak=.3, noise_dbfs=-62, lead=2.):
    """Speech-like phrases (harmonics, formants, fricatives) in a noisy room."""
    rng = np.random.default_rng(seed)
    n = seconds*RATE
    t = np.arange(n)/RATE
    gate = np.zeros(n)
    fric = np.zeros(n)
    pos = int(lead*RATE)  # Lead-in room tone, as in typical narration takes.
    while pos < n-RATE:
        length = int(rng.uniform(1.5, 3)*RATE)
        end = min(pos+length, n-RATE//2)
        u = np.arange(end-pos)/RATE
        syllables = np.clip(np.sin(2*np.pi*rng.uniform(3.5, 5)*u)**2*1.4, 0, 1)
        ramp = np.minimum(1, np.minimum(u, u[-1]-u)/.05)
        gate[pos:end] = syllables*ramp
        for _ in range(int(u[-1]*2)):
            a = pos+int(rng.uniform(0, u[-1]-.12)*RATE)
            fric[a:a+int(.09*RATE)] = np.hanning(int(.09*RATE))
        pos = end+int(rng.uniform(.5, 1.2)*RATE)
    f0 = 150+25*np.sin(2*np.pi*.3*t)+10*np.sin(2*np.pi*2.1*t)
    phase = 2*np.pi*np.cumsum(f0)/RATE
    voiced = np.zeros(n)
    for k in range(1, 30):
        hz = k*150
        envelope = sum(g/(1+((hz-c)/w)**2) for c, w, g in
                       ((500, 150, 1), (1500, 250, .5), (2600, 300, .25)))
        voiced += envelope*np.sin(k*phase)/k**.5
    hiss = signal.sosfilt(signal.butter(4, [3500, 8000], 'bandpass', fs=RATE,
                                        output='sos'), rng.normal(0, 1, n))
    dry = voiced*gate+.6*hiss*fric*(gate > 0)
    ir_t = np.arange(int(.5*RATE))/RATE
    ir = rng.normal(0, 1, len(ir_t))*np.exp(-ir_t/.08)*.08
    ir[0] = 1
    wet = signal.fftconvolve(dry, ir)[:n]
    wet *= speech_peak/np.max(abs(wet))
    noise = signal.lfilter([1], [1, -.95], rng.normal(0, 1, n))
    noise *= 10**(noise_dbfs/20)/np.sqrt(np.mean(noise**2))
    return wet+noise+10**((noise_dbfs-6)/20)*np.sin(2*np.pi*60*t), gate
