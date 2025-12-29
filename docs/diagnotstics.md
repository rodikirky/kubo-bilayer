# Diagnostics: FFT kz coverage and real-space wrap-around

This project computes real-space kernels \(G(\Delta z)\) via an FFT over \(k_z\), then uses those kernels
in further numerical integrations (e.g., Kubo-style expressions). Because later steps can sum/integrate
over many grid points, it is crucial that the FFT step is both correct and “safe” numerically.

Two different failure modes matter:

1. **Insufficient kz-range (coverage)**  
   The FFT kz window is too narrow and misses important peaks/weight of \(|G_{ij}(k_z)|\).

2. **Wrap-around / aliasing in real space**  
   \(G(\Delta z)\) does not decay sufficiently near the FFT box edges, so periodic wrap-around contaminates
   the result.

These checks are complementary: passing one does not guarantee passing the other.

---

## How to run the bulk diagnostics plotting script

Run the model-agnostic bulk plotting script for various choices of model and parameters

Defaults:
```bash
python -m scripts.plot_bulk_greens --preset toy_fft_near_shell_mid
python -m scripts.plot_bulk_greens --preset orbitronic_fft_mid
```
Custom:
```bash
python -m scripts.plot_bulk_greens --preset toy_fft_near_shell_mid --eta 0.05
python -m scripts.plot_bulk_greens --preset orbitronic_fft_mid --side right
python -m scripts.plot_bulk_greens --preset orbitronic_fft_mid --ij1 0 0 --ij2 2 2
python -m scripts.plot_bulk_greens --preset toy_fft_near_shell_mid --kz-diag-max 10 --nkz-diag 8001
python -m scripts.plot_bulk_greens --preset toy_fft_near_shell_mid --dz-zoom 50 --downsample 2
```