## Introduction
Digital holography provides a versatile framework for capturing and reconstructing the full optical wavefront of an object, including both amplitude and phase. By recording the interference between an object field and a coherent reference beam on a sensor plane, one can recover the original object field through computational reconstruction. This principle underlies a broad range of applications in optical metrology, microscopy, wavefront sensing, and information storage.

Among the reconstruction strategies, Fresnel propagation offers a computationally efficient approach for modeling near field diffraction. It provides an approximation to the full Rayleigh Sommerfeld diffraction integral and is well suited for simulating wave propagation over moderate distances. In holographic imaging, Fresnel propagation is particularly effective in modeling both the forward propagation of an object field to the detector plane and the inverse process of back-propagation for reconstruction.

The present implementation combines synthetic hologram generation, Fresnel wave propagation, and numerical reconstruction within a controlled simulation environment. The model supports different object types (point, Gaussian, circular aperture) and reference field configurations (on-axis, off-axis). It also allows for dynamic object motion along prescribed paths (linear, sinusoidal, or circular trajectories), enabling time resolved hologram sequences. The resulting holographic data are animated and stored as image sequences, illustrating both the forward hologram formation and the reconstruction process.

This framework demonstrates the essential principles of digital holography while providing an extensible tool for exploring more advanced scenarios, such as noise robustness, phase-shifting methods, or deep learning-based reconstructions.
 
![](wavefront_propagation.gif)


This project implements a **physics-based simulation of digital holography** using Fresnel diffraction.  
It generates dynamic hologram sequences for moving objects, simulates interference with a reference beam,  
and reconstructs the object field through numerical backpropagation.

## Features

- **Fresnel propagation model** for forward and backward wavefront computation.  
- Configurable **object types**: point source, Gaussian distribution, or circular aperture.  
- Configurable **reference fields**: on-axis plane wave or tilted off-axis reference.  
- Object motion along **sinusoidal, linear, or circular trajectories**.  
- Animated visualization of:
  - Object amplitude in the object plane.  
  - Hologram intensity at the sensor plane.  
  - Reconstructed amplitude after numerical backpropagation.  
- Output saved as an animated `.gif` and playable inline in Jupyter notebooks.  

## Requirements

- Python ≥ 3.8  
- NumPy  
- Matplotlib  
- IPython / Jupyter environment  
- Pillow (for saving animations)

**Install dependencies with:** pip install numpy matplotlib pillow



## Extensions

This framework can be extended by:

Adding noise models to simulate realistic detectors.

Incorporating multiple objects or extended phase objects.

Comparing different reconstruction algorithms.

Integrating machine learning for denoising or phase retrieval.

## License

This project is released under the MIT License
