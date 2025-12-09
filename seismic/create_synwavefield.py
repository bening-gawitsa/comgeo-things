import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ricker
from numpy.fft import rfft, irfft, rfftfreq
from obspy import Trace, Stream, UTCDateTime
import numpy as np

def make_synthetic_shotgather_optionA(
    nx=60, dx=1.0, S=1.0,
    nt=2048, dt=0.0005,        # time samples and sampling interval
    Vs_layer=200.0, Vp_layer=800.0, 
    # rho_layer=2000.0, 
    h=10.0,
    Vs_half=400.0, 
    # Vp_half=1200.0, rho_half=2000.0,
    f0=25.0,                   # central frequency for surface-wave spectrum (Hz)
    fmax_plot=60.0,
    # noise_level=0.02,         # relative noise amplitude
    plot=True
):
    """
    Create synthetic u(x,t) for Option A (2-layer model):
      - dispersive surface-like arrival made by frequency-dependent phase shift
      - direct P and S arrivals (non-dispersive) using Vp and Vs of near-surface
      - one reflection from the layer bottom (two-way travel)
    Returns:
        u: (nt, nx) array of time-domain traces
        t: time vector (nt,)
        x: receiver positions (nx,)
    """

    # ===== geometry =====
    x = np.arange(nx)*dx + S       # receiver positions from first sensor (source at x=0 offset S)
    t = np.arange(nt)*dt

    # ===== frequency domain setup =====
    nfft = 2**int(np.ceil(np.log2(nt)))    # next power of 2
    freqs = rfftfreq(nfft, dt)            # positive frequencies
    nf = freqs.size

    # Surface-wave (dispersive) frequency content (Gaussian centered at f0)
    spec_amp = np.exp(-0.5 * ((freqs - f0)/(f0/2))**2)  # amplitude spectrum
    spec_amp[freqs<1e-6] = 0.0   # remove DC

    # ===== Define a simple dispersion law v(f) for Rayleigh-like wave =====
    # Make v(f) increase with frequency: low freq -> ~Vs_layer*1.1 (slow), high freq -> Vs_half*0.95 (fast)
    fmin = 1.0
    fmax = min(fmax_plot, freqs.max())
    # linear interpolation between two limits
    v_low = Vs_layer * 1.05
    v_high = Vs_half * 0.95
    # ensure monotonic mapping on available freq band
    v_of_f = np.interp(freqs, [fmin, fmax], [v_low, v_high])
    v_of_f[freqs < fmin] = v_low

    # ===== Build frequency-domain wavefield for each receiver (surface wave) =====
    # U(f,x) = spec_amp(f) * exp(-i*2π*f * (x / v(f)))
    # We'll compute for each receiver then inverse FFT to time
    U_f_x = np.zeros((nf, nx), dtype=np.complex64)
    for ix, xi in enumerate(x):
        tau_shift = xi / v_of_f                 # frequency-dependent travel time for this receiver
        phase = np.exp(-2j * np.pi * freqs * tau_shift)
        U_f_x[:, ix] = spec_amp * phase

    # Inverse FFT (real) to get time-domain traces (pad to nfft and take nt samples)
    u_surface = np.zeros((nfft, nx))
    for ix in range(nx):
        u_surface[:, ix] = irfft(U_f_x[:, ix], n=nfft)
    u_surface = u_surface[:nt, :]

    # Normalize surface arrival amplitude
    u_surface /= np.max(np.abs(u_surface)) + 1e-5
    u_surface *= 1.0   # control amplitude

    # ===== Add direct P and S arrivals (non-dispersive) =====
    # Use small Ricker wavelets inserted at appropriate travel times
    # Direct (near-surface) velocities used for travel times
    wavelet_len = int(0.05 / dt) + 1  # ~50 ms wavelet
    if wavelet_len % 2 == 0:
        wavelet_len += 1
    # Create simple Ricker-like wavelet using scipy.signal.ricker requires points and width param
    # We'll create a small gaussian-like pulse via ricker and scale it
    rick = ricker(wavelet_len, a=wavelet_len/6.0)
    rick = rick / np.max(np.abs(rick))

    u_body = np.zeros((nt, nx))
    amp_P = 4.6
    amp_S = 2.9
    amp_ref = 2.5

    for ix, xi in enumerate(x):
        # direct P
        tP = xi / Vp_layer + 0.06
        idxP = int(round(tP / dt))
        if 0 <= idxP < nt - wavelet_len:
            u_body[idxP:idxP+wavelet_len, ix] += amp_P * rick

        # direct S
        tS = xi / Vs_layer + 0.06
        idxS = int(round(tS / dt))
        if 0 <= idxS < nt - wavelet_len:
            u_body[idxS:idxS+wavelet_len, ix] += amp_S * rick * 3.7

        # reflected from layer bottom (two-way travel to interface depth h then to receiver approximated)
        # Simple approx: vertical two-way to depth h plus horizontal propagation at near-surface speed
        t_ref = (2.0 * h) / Vp_layer + xi / Vp_layer
        idxR = int(round(t_ref / dt))
        if 0 <= idxR < nt - wavelet_len:
            u_body[idxR:idxR+wavelet_len, ix] += amp_ref * rick * 5.8

    # Normalize body relative to surface
    max_total = max(np.max(np.abs(u_surface)), np.max(np.abs(u_body))) + 1e-5
    u_body = u_body / max_total
    u_surface = u_surface / max_total

    # ===== Combine surface + body, add noise =====
    u = 2.0 * u_surface + 3.9 * u_body
    # # Add Gaussian noise
    # u += noise_level * np.random.randn(*u.shape)

    # Simple gain vs offset (to mimic real data amplitude decay)
    # apply 1/sqrt(distance) taper
    for ix, xi in enumerate(x):
        u[:, ix] *= 1.0 / np.sqrt(xi + 1.0)

    if plot:
        # Wiggle plot (filled for negative lobes) — vertical time axis
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        gain = 1.5 * dx  # horizontal scaling of wiggles
        maxamp = np.max(np.abs(u))
        for ix in range(nx):
            trace = u[:, ix] / (maxamp + 1e-12) * gain
            ax.plot(x[ix] + trace, t, color='k', linewidth=0.6)
            ax.fill_betweenx(t, x[ix], x[ix] + trace, where=(trace < 0), color='k', alpha=0.6)
        ax.set_xlim(x.min()-dx, x.max()+dx)
        ax.set_ylim(t.max(), t.min())
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Time (s)')
        ax.set_title('Synthetic shot-gather (Option A)')
        plt.gca()
        plt.show()

    return u, t, x



def write_segy_from_synthetic(u_xt, t, x, dt, filename="synthetic.sgy"):
    nt, nx = u_xt.shape  # nt time samples, nx receivers

    stream = Stream()

    for i in range(nx):
        trace = Trace(data=u_xt[:, i].astype(np.float32))
        trace.stats.delta = dt
        trace.stats.starttime = UTCDateTime(0)  # arbitrary
        trace.stats.station = f"REC{i+1}"
        trace.stats.channel = "EHZ"
        trace.stats.segy = {}

        # SEG-Y trace header assignments
        trace.stats.segy.trace_header = {
            "trace_sequence_number_within_line": i + 1,
            "trace_sequence_number_within_segy_file": i + 1,
            "source_receiver_offset": int(x[i]),    # offset in meters
            "receiver_group_elevation": 0,
            "coordinate_scalar": -100,  # -100 = divide coordinates by 100
            "group_coordinate_x": int(x[i] * 100),
            "group_coordinate_y": 0,
        }

        stream.append(trace)

    # Write SEG-Y file
    stream.write(filename, format="SEGY", data_encoding=1)  # 1 = 4-byte IBM float

    print(f"SEG-Y written to: {filename}")


# === Example usage ===
u_xt, t, x = make_synthetic_shotgather_optionA(
    nx=60, dx=1.0, S=1.0,
    nt=1548, dt=0.0005,
    Vs_layer=200.0, Vp_layer=800.0, h=10.0,
    Vs_half=400.0, 
    # Vp_half=1200.0,
    f0=15.0, 
    # noise_level=0.03,
    plot=True
)

file_path = r"C:\Users\benin\orion\comgeo-things\data\synthetic_3.sgy"
write_segy_from_synthetic(u_xt, t, x, dt=0.0005, 
                    filename=os.path.join(file_path))