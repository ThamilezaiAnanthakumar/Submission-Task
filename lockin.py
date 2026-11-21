"""
I/Q lock-in extraction of electrode-tissue impedance (ETI) in the sub-100 Hz
region of ear-EEG. 

Signal model (Section 1 of the report):
    x(t) = I0 |Z(f0,t)| cos(2 pi f0 t + phi(t))     probe response (wanted)
         + s_EEG(t)                                  delta..gamma bands + 1/f
         + v_mains(t)                                50 Hz + harmonics
         + v_dc(t)                                   half-cell offset + drift
         + n(t)                                      white electrode/amplifier noise
Z is a Randles model  Z = Rs + Rct / (1 + j w Rct Cdl).
"""
import json
import os

import matplotlib
import numpy as np
import scipy.signal as sg
from scipy.integrate import trapezoid

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False,
                     "pgf.texsystem": "pdflatex", "pgf.rcfonts": False})


def save(fig_name, **kw):
    """Save as PDF."""
    plt.savefig(f"figures/{fig_name}.pdf", **kw)
    plt.close()

rng = np.random.default_rng(42)


fs = 2000.0                 # sampling rate (Hz)
T = 20.0                    # record length (s)
t = np.arange(0, T, 1 / fs)
N = t.size

f0 = 75.0                   # probe frequency (Hz): above EEG bands, away from 50/100 Hz
I0 = 2e-9                   # probe current amplitude (A)
Rs, Cdl = 5e3, 47e-9        # Randles parameters (Ohm, F)
Rct0 = 200e3                # baseline charge-transfer resistance (Ohm)

B_IN = (0.5, 100.0)         # recording band entering the lock-in -> B_in ~ 100 Hz

EEG_BANDS = {               # name: (low Hz, high Hz, rms V)
    "delta": (0.5, 4, 10e-6),
    "theta": (4, 8, 6e-6),
    "alpha": (8, 13, 10e-6),
    "beta": (13, 30, 4e-6),
    "gamma": (30, 45, 1.5e-6),
}
PINK_RMS = 3e-6             # broadband 1/f background (V rms)
WHITE_DENSITY = 0.3e-6      # electrode + amplifier white noise (V/sqrt(Hz))
MAINS = [(50, 200e-6), (100, 40e-6), (150, 15e-6)]   # (Hz, V amplitude)
V_HALF_CELL = 5e-3          # DC half-cell offset (V)


def randles(f, area=1.0):
    """Randles impedance for a relative wetted contact area `area` (1 = baseline).
    Smaller contact area: Rct = Rct0/area, Cdl = Cdl0*area, Rs = Rs0/sqrt(area)."""
    rs, rct, cdl = Rs / np.sqrt(area), Rct0 / area, Cdl * area
    return rs + rct / (1 + 1j * 2 * np.pi * f * rct * cdl)


def area_profile(tt):
    """Relative contact area: earpiece loosens at 8 s (area -> 0.67), 150 ms tap at 14 s (area dips to 0.5)."""
    step = -0.33 / (1 + np.exp(-(tt - 8.0) / 0.05))
    tap = -0.17 * np.exp(-0.5 * ((tt - 14.0) / 0.05) ** 2)
    return 1.0 + step + tap


#  Signal generation
def band_noise(lo, hi, rms, n=N):
    w = rng.standard_normal(n)
    sos = sg.butter(4, [lo, hi], "bandpass", fs=fs, output="sos")
    y = sg.sosfiltfilt(sos, w)
    return y / np.std(y) * rms


def pink_noise(rms, n=N):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    y = np.fft.irfft(spec, n)
    return y / np.std(y) * rms


def make_eeg(n=N):
    return sum(band_noise(lo, hi, r, n) for lo, hi, r in EEG_BANDS.values()) + pink_noise(PINK_RMS, n)


def make_interference(n=N, tt=t):
    mains = sum(a * np.cos(2 * np.pi * f * tt + rng.uniform(0, 2 * np.pi)) for f, a in MAINS)
    vdc = V_HALF_CELL + 0.5e-3 * np.sin(2 * np.pi * 0.03 * tt)       # slow half-cell drift
    white = WHITE_DENSITY * np.sqrt(fs / 2) * rng.standard_normal(n)
    return mains, vdc, white


def probe_voltage(freqs, area_t, tt=t, i0=I0):
    """Sum of probe responses I0|Z|cos(2 pi f t + phi) for one or more tones."""
    v = np.zeros_like(tt)
    for f in np.atleast_1d(freqs):
        z = randles(f, area_t)
        v += i0 * np.abs(z) * np.cos(2 * np.pi * f * tt + np.angle(z))
    return v


area_t = area_profile(t)
Z_true = randles(f0, area_t)
probe = probe_voltage(f0, area_t)
eeg = make_eeg()
mains, vdc, white = make_interference()
background = eeg + mains + vdc + white
x = probe + background


#  I/Q lock-in
def lockin(sig, f_ref, B, order=4, tt=t, i0=I0):
    """Mix with 2cos / 2sin, low-pass (Butterworth, cut-off B), recover |Z| and phase."""
    sos = sg.butter(order, B, "low", fs=fs, output="sos")
    I = sg.sosfilt(sos, sig * 2 * np.cos(2 * np.pi * f_ref * tt))
    Q = sg.sosfilt(sos, sig * 2 * np.sin(2 * np.pi * f_ref * tt))
    return np.hypot(I, Q) / i0, np.degrees(np.arctan2(-Q, I)), I, Q


ENBW_FACTOR = 1.026          # 4th-order Butterworth: ENBW = 1.026 * cut-off


# Figure : spectra before and after mixing (replaces the hand-drawn sketch)
def asd(sig, nper=int(4 * fs)):
    f, p = sg.welch(sig, fs=fs, nperseg=nper, detrend=False)   # keep DC visible
    return f, np.sqrt(p) * 1e6        # uV / sqrt(Hz)


# use a constant-Z record so the probe is a clean line.
probe_const = probe_voltage(f0, np.ones(N))
eeg_all = eeg + vdc + white
x_const = probe_const + eeg_all + mains
B_demo = 1.0
RED, GRN, BLU, YEL = "#c0392b", "#2e7d32", "#3d6fb6", "#e6b800"
mixer = 2 * np.cos(2 * np.pi * f0 * t)
comp = {"eeg": eeg_all, "probe": probe_const, "mains": mains}
# EEG drawn at 0.5 Hz resolution (smooth, light figure); spikes at 0.25 Hz resolution
specA = {k: asd(v, int(2 * fs) if k == "eeg" else int(4 * fs)) for k, v in comp.items()}
specB = {k: asd(v * mixer, int(2 * fs) if k == "eeg" else int(4 * fs)) for k, v in comp.items()}
FLOOR = 3e-2


def draw(ax, spec, fmax, label=True):
    f, a = spec["eeg"]; m = f <= fmax
    ax.set_yscale("log")
    ax.fill_between(f[m], FLOOR, a[m], facecolor=(0.24, 0.44, 0.71, 0.25), edgecolor=BLU, lw=0.8,
                    label="EEG (incl. half-cell drift + random noise)" if label else None)
    for key, col, lab in [("mains", GRN, "mains interference"), ("probe", RED, "probe response (wanted)")]:
        f, a = spec[key]; m = f <= fmax
        a_pad = np.r_[0, a[m], 0]                      # lets a peak at 0 Hz be found
        pk, _ = sg.find_peaks(a_pad, height=10 * FLOOR)
        ax.vlines(f[m][pk - 1], FLOOR, a[m][pk - 1], color=col, lw=1.6, label=lab if label else None)


def peak(spec, f_at):
    f, a = spec
    return a[np.argmin(abs(f - f_at))]


def tag(ax, x, y, text, color):
    ax.text(x, y * 2.5, text, color=color, fontsize=7, ha="center", va="bottom")


fH = np.linspace(0, 250, 1001)
_, H = sg.freqz(*sg.butter(4, B_demo, "low", fs=fs), worN=fH, fs=fs)
HdB = 20 * np.log10(np.abs(H) + 1e-12)

fig = plt.figure(figsize=(6.85, 4.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.6, wspace=0.6)

#  measured signal
a = fig.add_subplot(gs[0, :])
draw(a, specA, 200)
tag(a, f0, peak(specA["probe"], f0), r"probe at $f_0$ = 75 Hz", RED)
for fm in (50, 100, 150):
    tag(a, fm, peak(specA["mains"], fm), f"{fm} Hz", GRN)
a.text(22, 6, "EEG bands\n(0.5-45 Hz)", color=BLU, fontsize=7, ha="center")
a.set_xlim(-2, 200); a.set_ylim(FLOOR, 1e7)
a.set_xlabel("Frequency (Hz)"); a.set_ylabel(r"ASD ($\mu$V/$\sqrt{\mathrm{Hz}}$)")
a.legend(loc="upper right", fontsize=7, ncol=1, handlelength=1.6, borderaxespad=0.3)
a.set_title(r"(a) Measured signal $x(t)$, before the lock-in", loc="left", fontsize=8.5)

# after mixing
b = fig.add_subplot(gs[1, :3])
draw(b, specB, 250, label=False)
b.text(4, 8e3, r"probe $\to$ DC ($=I$)", color=RED, fontsize=7, ha="left")
tag(b, 2 * f0, peak(specB["probe"], 2 * f0), r"$2f_0$", RED)
tag(b, f0, peak(specB["eeg"], f0), r"EEG $\to$ around $f_0$", BLU)
b.text(27, 5e2, r"$|50{-}f_0|$, $100{-}f_0$", color=GRN, fontsize=7, ha="center")
b.text(80, 60, r"$150{-}f_0$", color=GRN, fontsize=7, ha="left")
for fm, lab in [(125, r"$50{+}f_0$"), (175, r"$100{+}f_0$"), (225, r"$150{+}f_0$")]:
    tag(b, fm, peak(specB["mains"], fm), lab, GRN)
b2 = b.twinx(); b2.plot(fH, HdB, color=YEL, lw=1.6); b2.set_ylim(-130, 5)
b2.set_yticks([]); b2.spines["right"].set_visible(False)
b.set_xlim(-2, 250); b.set_ylim(FLOOR, 1e6)
b.set_xlabel("Frequency (Hz)"); b.set_ylabel(r"ASD ($\mu$V/$\sqrt{\mathrm{Hz}}$)")
b.set_title(r"(b) After mixing with $2\cos(2\pi f_0 t)$; yellow = LPF", loc="left", fontsize=8.5)

# zoom on the LPF passband
c = fig.add_subplot(gs[1, 3])
fz_max = 4
draw(c, specB, fz_max, label=False)
fZ = np.linspace(0, fz_max, 201)
_, HZ = sg.freqz(*sg.butter(4, B_demo, "low", fs=fs), worN=fZ, fs=fs)
c.axvspan(0, B_demo, color=YEL, alpha=0.25, lw=0)
c2 = c.twinx(); c2.plot(fZ, 20 * np.log10(np.abs(HZ) + 1e-12), color=YEL, lw=1.6); c2.set_ylim(-60, 5)
c2.spines["right"].set_visible(True); c2.tick_params(labelsize=6.5)
c2.set_ylabel("LPF gain (dB)", color="#b38f00", fontsize=7)
c.text(0.35, 60, "probe kept", color=RED, fontsize=7)
c.text(0.9, 1.2, "EEG residual", color=BLU, fontsize=7)
c.set_xlim(-0.1, fz_max); c.set_ylim(FLOOR, 1e4)
c.set_xlabel("Frequency (Hz)")
c.set_title(rf"(c) Zoom, $B$ = {B_demo:g} Hz", loc="left", fontsize=8.5)
save("fig2_spectra", bbox_inches="tight")


# Figure :  tracking a changing Z with narrow vs wide LPF
t_show = (t >= 2) & (np.arange(N) % 40 == 0)   # plot every 40th sample (50 Hz is enough to show B <= 5 Hz)
fig, ax = plt.subplots(2, 1, figsize=(6.85, 2.9), sharex=True)
ax[0].plot(t[t_show], np.abs(Z_true[t_show]) / 1e3, "k", lw=1.6, label="true $|Z(f_0)|$")
ax[1].plot(t[t_show], np.degrees(np.angle(Z_true[t_show])), "k", lw=1.6)
track = {}
for B, c in [(5.0, "#e67e22"), (0.5, "#2c6fbb")]:
    zm, ph, _, _ = lockin(x, f0, B)
    track[B] = (zm, ph)
    ax[0].plot(t[t_show], zm[t_show] / 1e3, color=c, lw=0.9, label=f"lock-in, $B$ = {B:g} Hz")
    ax[1].plot(t[t_show], ph[t_show], color=c, lw=0.9)
ax[0].set_ylabel(r"$|Z|$ (k$\Omega$)"); ax[1].set_ylabel(r"$\angle Z$ (deg)")
ax[1].set_xlabel("Time (s)")
ax[0].legend(ncol=3, loc="upper left", fontsize=7)
ax[0].annotate("earpiece loosens", xy=(8.3, 60), xytext=(4.0, 78), arrowprops=dict(arrowstyle="->"), fontsize=7)
ax[0].annotate("150 ms tap", xy=(14.05, 85), xytext=(15.5, 92), arrowprops=dict(arrowstyle="->"), fontsize=7)
ax[0].set_ylim(35, 100)
plt.tight_layout()
save("fig3_tracking")


A = I0 * abs(randles(f0))                    # probe amplitude (V)
f_w, p_w = sg.welch(background, fs=fs, nperseg=int(8 * fs))
win = (f_w > 70) & (f_w < 80)
S_f0 = np.mean(p_w[win])                           # one-sided PSD of background near f0 (V^2/Hz)
in_band = (f_w >= B_IN[0]) & (f_w <= B_IN[1])
P_in = trapezoid(p_w[in_band], f_w[in_band])        # background power entering the lock-in
snr_in_db = 10 * np.log10((A ** 2 / 2) / P_in)
Bin_width = B_IN[1] - B_IN[0]
S_alpha = np.mean(p_w[(f_w > 9) & (f_w < 11)])

TL = 60.0                                          # longer records for error statistics
tL = np.arange(0, TL, 1 / fs)
NL = tL.size
seg = slice(int(15 * fs), NL)                      # skip filter start-up
probe_L = probe_voltage(f0, np.ones(NL), tt=tL)
bg_L = [make_eeg(NL) + sum(make_interference(NL, tL)) for _ in range(3)]

rows = []
for B in [0.2, 0.5, 1, 2, 5, 10]:
    rel = np.mean([np.std(lockin(probe_L + bg, f0, B, tt=tL)[0][seg]) for bg in bg_L]) / abs(randles(f0))
    rel_th = np.sqrt(2 * S_f0 * ENBW_FACTOR * B) / A
    snr_out_db = 20 * np.log10(1 / rel)
    rows.append(dict(B=B, T=1 / (2 * ENBW_FACTOR * B), tr=0.35 / B, rel=rel, rel_th=rel_th,
                     snr_out=snr_out_db, gain_white=10 * np.log10(Bin_width / (ENBW_FACTOR * B)),
                     gain_meas=snr_out_db - snr_in_db))



# Going beyond: multi-tone averaging

B_mt = 0.5
mt_res = []
for K in [1, 2, 4, 8]:
    errs = []
    for trial in range(4):
        freqs = np.array([f0]) if K == 1 else np.linspace(60, 90, K)
        bg = make_eeg(NL) + sum(make_interference(NL, tL))
        sig = probe_voltage(freqs, np.ones(NL), tt=tL) + bg
        est = np.mean([lockin(sig, fk, B_mt, tt=tL)[0] for fk in freqs], axis=0)
        truth = np.mean([abs(randles(fk)) for fk in freqs])
        errs.append(np.std(est[seg]) / truth)
    mt_res.append((K, float(np.mean(errs))))

# second panel: neighbouring-electrode subtraction, variance model check
rho_vals = np.array([0, 0.5, 0.8, 0.9, 0.95, 0.99])
sub_res = {}
for f_probe, label in [(75.0, "f0 = 75 Hz"), (11.0, "f0 = 11 Hz (alpha)")]:
    res = []
    pr = probe_voltage(f_probe, np.ones(NL), tt=tL)
    for rho in rho_vals:
        ratios = []
        for trial in range(3):
            e1 = make_eeg(NL)
            e2 = rho * e1 + np.sqrt(1 - rho ** 2) * make_eeg(NL)
            n1 = WHITE_DENSITY * np.sqrt(fs / 2) * rng.standard_normal(NL)
            n2 = WHITE_DENSITY * np.sqrt(fs / 2) * rng.standard_normal(NL)
            single = lockin(pr + e1 + n1, f_probe, 0.5, tt=tL)[0]
            diff = lockin(pr + e1 + n1 - (e2 + n2), f_probe, 0.5, tt=tL)[0]
            ratios.append(np.std(diff[seg]) / np.std(single[seg]))
        res.append(float(np.mean(ratios)))
    sub_res[label] = res

fig, ax = plt.subplots(1, 2, figsize=(6.85, 2.3))
Ks = [k for k, _ in mt_res]
ax[0].plot(Ks, [100 * e for _, e in mt_res], "o-", color="#2c6fbb", label="simulation")
ax[0].plot(Ks, [100 * mt_res[0][1] / np.sqrt(k) for k in Ks], "--", color="gray", label=r"$1/\sqrt{K}$")
ax[0].set_xscale("log", base=2); ax[0].set_xticks(Ks); ax[0].set_xticklabels(Ks)
ax[0].set_xlabel("number of probe tones $K$ (60-90 Hz)"); ax[0].set_ylabel(r"error in $|Z|$ $(\%)$")
ax[0].set_title(f"(a) Multi-tone averaging, $B$ = {B_mt:g} Hz", loc="left", fontsize=8.5)
ax[0].legend(fontsize=7)
fe, pe = sg.welch(make_eeg(NL), fs=fs, nperseg=int(8 * fs))
Sn = WHITE_DENSITY ** 2
rr = np.linspace(0, 0.995, 100)
for (label, res), c, fp in zip(sub_res.items(), ["#2c6fbb", "#c0392b"], [75.0, 11.0]):
    Se = np.mean(pe[(fe > fp - 1) & (fe < fp + 1)])
    ax[1].plot(rho_vals, res, "o", ms=3.5, color=c, label=label.replace("f0", "$f_0$") + " (sim.)")
    ax[1].plot(rr, np.sqrt((2 * (1 - rr) * Se + 2 * Sn) / (Se + Sn)), "-", lw=1, color=c, alpha=0.7)
ax[1].axhline(1, color="gray", lw=0.8, ls=":")
ax[1].set_xlabel(r"EEG correlation $\rho$ between neighbouring electrodes")
ax[1].set_ylabel("error ratio (subtracted / single)")
ax[1].set_title("(b) Neighbour subtraction (lines: theory)", loc="left", fontsize=8.5)
ax[1].legend(fontsize=7)
plt.tight_layout()
save("fig4_beyond")
