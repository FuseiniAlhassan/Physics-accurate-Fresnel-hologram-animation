import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from datetime import datetime

config = {
    "seed": 42,
    "N": 256,                    # grid size (NxN)
    "physical_size": 5e-3,       # physical extent (meters) of the window (sensor/object plane)
    "wavelength": 633e-9,        # meters
    "z": 0.05,                   # propagation distance (m) from object plane to sensor
    "object_type": "gaussian",   # 'point' or 'gaussian' or 'circle'
    "object_amp": 1.0,
    "object_sigma_frac": 0.02,   # gaussian width as fraction of physical_size
    "ref_type": "offaxis",       # 'onaxis' or 'offaxis'
    "ref_tilt_deg": 2.0,         # tilt angle (degrees) for off-axis reference
    "frames": 120,
    "path": "sine",              # 'sine' or 'line' or 'circle'
}

np.random.seed(config["seed"])

def grids(N, physical_size):
    x = np.linspace(-physical_size/2, physical_size/2, N)
    y = np.linspace(-physical_size/2, physical_size/2, N)
    X, Y = np.meshgrid(x, y)
    dx = x[1] - x[0]
    return X, Y, dx

def fresnel_transfer(N, dx, wavelength, z):
    # frequency coordinates
    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(np.fft.fftshift(fx), np.fft.fftshift(fy))
    # Fresnel transfer function (frequency domain)
    H = np.exp(1j * 2 * np.pi / wavelength * z) * np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    # use fftshifted H since we'll multiply with fftshifted spectra
    H = np.fft.ifftshift(H)
    return H

def fresnel_propagate(u0, H):
    # propagate: Uout = ifft2( fft2(u0) * H )
    U0 = np.fft.fft2(u0)
    U1 = U0 * H
    u1 = np.fft.ifft2(U1)
    return u1

def fresnel_backpropagate(u_sensor, H):
    # backpropagate by multiplying with conjugate of H
    U = np.fft.fft2(u_sensor)
    Ub = U * np.conj(H)
    ub = np.fft.ifft2(Ub)
    return ub


# Build coordinate grids and precompute H
N = config["N"]
X, Y, dx = grids(N, config["physical_size"])
H = fresnel_transfer(N, dx, config["wavelength"], config["z"])
k = 2 * np.pi / config["wavelength"]

def reference_field(X, Y, config):
    if config["ref_type"].lower() == "onaxis":
        return np.ones_like(X, dtype=complex) * 1.0
    else:
        # small tilt in x-direction
        theta = np.deg2rad(config["ref_tilt_deg"])
        kx = k * np.sin(theta)
        ky = 0.0
        return np.exp(1j * (kx * X + ky * Y))

def make_object_field(center_x, center_y, config):
    if config["object_type"] == "point":
        # delta-like: put energy in one pixel (approx)
        obj = np.zeros((N, N), dtype=complex)
        # find nearest indices
        ix = int(np.round((center_x + config["physical_size"]/2) / dx))
        iy = int(np.round((center_y + config["physical_size"]/2) / dx))
        ix = np.clip(ix, 0, N-1)
        iy = np.clip(iy, 0, N-1)
        obj[iy, ix] = config["object_amp"]
        return obj
    else:
        # gaussian amplitude
        sigma = config["object_sigma_frac"] * config["physical_size"]
        A = config["object_amp"] * np.exp(-((X-center_x)**2 + (Y-center_y)**2) / (2*sigma**2))
        return A.astype(complex)
      
def generate_path(config):
    frames = config["frames"]
    if config["path"] == "sine":
        t = np.linspace(0, 1, frames)
        # move in x across, y follows sine
        x_centers = (t - 0.5) * config["physical_size"] * 0.8
        y_centers = 0.2 * config["physical_size"] * np.sin(4 * np.pi * t)
    elif config["path"] == "circle":
        t = np.linspace(0, 2*np.pi, frames)
        r = 0.2 * config["physical_size"]
        x_centers = r * np.cos(t)
        y_centers = r * np.sin(t)
    else:  # line
        x_centers = np.linspace(-0.4*config["physical_size"], 0.4*config["physical_size"], frames)
        y_centers = np.zeros(frames)
    return x_centers, y_centers

x_centers, y_centers = generate_path(config)
U_ref_grid = reference_field(X, Y, config)

frames = config["frames"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ax_obj, ax_holo, ax_rec = axes
im_obj = ax_obj.imshow(np.zeros((N,N)), cmap='gray', vmin=0, vmax=1,
                       extent=[-config["physical_size"]/2, config["physical_size"]/2,
                               -config["physical_size"]/2, config["physical_size"]/2])
ax_obj.set_title("Object plane (amplitude)")
ax_obj.set_xlabel("x (m)")
ax_obj.set_ylabel("y (m)")

im_holo = ax_holo.imshow(np.zeros((N,N)), cmap='gray', vmin=0, vmax=1,
                         extent=[-1/(2*dx), 1/(2*dx), -1/(2*dx), 1/(2*dx)])
ax_holo.set_title("Hologram intensity (sensor)")
ax_holo.set_xlabel("spatial freq (1/m)")
ax_holo.set_ylabel("")

im_rec = ax_rec.imshow(np.zeros((N,N)), cmap='gray', vmin=0, vmax=1,
                       extent=[-config["physical_size"]/2, config["physical_size"]/2,
                               -config["physical_size"]/2, config["physical_size"]/2])
ax_rec.set_title("Reconstruction (amplitude)")
ax_rec.set_xlabel("x (m)")
ax_rec.set_ylabel("")

plt.tight_layout()

def update(frame_idx):
    cx = x_centers[frame_idx]
    cy = y_centers[frame_idx]
    # object field (amplitude-only or small complex)
    U_obj = make_object_field(cx, cy, config)

    # propagate object to sensor
    U_sensor_obj = fresnel_propagate(U_obj, H)

    # reference at sensor
    U_ref = U_ref_grid

    # hologram intensity: |U_obj_at_sensor + U_ref|^2
    U_total = U_sensor_obj + U_ref
    I_holo = np.abs(U_total)**2
    I_holo = I_holo / (I_holo.max() + 1e-12)

    # simple reconstruction: assume amplitude sqrt(I) with zero phase, then backpropagate
    U_recorded_amp = np.sqrt(I_holo)
    U_rec = fresnel_backpropagate(U_recorded_amp, H)
    A_rec = np.abs(U_rec)
    A_rec = A_rec / (A_rec.max() + 1e-12)

    # object amplitude for display
    A_obj = np.abs(U_obj)
    if A_obj.max() > 0:
        A_obj_disp = A_obj / A_obj.max()
    else:
        A_obj_disp = A_obj

    # update images
    im_obj.set_data(A_obj_disp)
    im_holo.set_data(I_holo)
    im_rec.set_data(A_rec)

    # update titles with frame info
    ax_obj.set_title(f"Object plane (frame {frame_idx+1}/{frames})")
    ax_holo.set_title("Hologram intensity (sensor)")
    ax_rec.set_title("Reconstruction (amplitude)")
    return im_obj, im_holo, im_rec

# Create animation and save
ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
ani.save("wavefront_propagation.gif", writer="pillow", fps=20)

# Display with play/pause buttons
HTML(ani.to_jshtml())
