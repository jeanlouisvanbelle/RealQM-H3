"""
RealQM H₃⁺ Simulation - Multi-Core Flexible Model
Y-7: Exploring tri-core molecular topology
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
import json
from datetime import datetime

class RealQMMultiCore:
    """
    Flexible RealQM simulator for N cores and M electrons.
    Configured for H3+ (3 cores, 2 electrons) by default.
    """
    
    def __init__(self, core_positions, Z_eff=1.0, 
                 grid_size=41, box_half=3.0, 
                 steps=500, dt=0.0005, plot_every=50):
        
        self.core_positions = np.array(core_positions)
        self.n_cores = len(core_positions)
        
        if isinstance(Z_eff, (int, float)):
            self.Z_eff = np.full(self.n_cores, Z_eff)
        else:
            self.Z_eff = np.array(Z_eff)
        
        self.N = grid_size
        self.L = box_half
        self.steps = steps
        self.dt = dt
        self.plot_every = plot_every
        
        self._setup_grid()
        self._setup_cores()
        self._initialize_densities()
        
        self.history = {
            'energy': [], 'overlap': [], 'kinetic': [], 
            'potential': [], 'ee_repulsion': []
        }
    
    def _setup_grid(self):
        x = np.linspace(-self.L, self.L, self.N)
        y = np.linspace(-self.L, self.L, self.N)
        z = np.linspace(-self.L, self.L, self.N)
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')
        self.dx = x[1] - x[0]
        self.dv = self.dx**3
    
    def _setup_cores(self):
        eps = 0.05
        self.V_core = np.zeros_like(self.X)
        for i, (cx, cy, cz) in enumerate(self.core_positions):
            R = np.sqrt((self.X - cx)**2 + (self.Y - cy)**2 + (self.Z - cz)**2 + eps**2)
            self.V_core += -self.Z_eff[i] / R
        self.core_x = [p[0] for p in self.core_positions]
        self.core_y = [p[1] for p in self.core_positions]
    
    def _initialize_densities(self):
        center = np.mean(self.core_positions, axis=0)
        rho1 = np.exp(-((self.X - center[0] + 0.3)**2 + 
                       (self.Y - center[1] + 0.3)**2 + 
                       (self.Z - center[2] + 0.3)**2) / 2.0)
        rho2 = np.exp(-((self.X - center[0] - 0.3)**2 + 
                       (self.Y - center[1] - 0.3)**2 + 
                       (self.Z - center[2] - 0.3)**2) / 2.0)
        self.rho1 = rho1 / (np.sum(rho1) * self.dv)
        self.rho2 = rho2 / (np.sum(rho2) * self.dv)
    
    def _poisson_solve(self, rho):
        kx = 2 * np.pi * np.fft.fftfreq(self.N, self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.N, self.dx)
        kz = 2 * np.pi * np.fft.fftfreq(self.N, self.dx)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k2 = KX**2 + KY**2 + KZ**2
        k2[0,0,0] = 1.0
        rho_hat = np.fft.fftn(rho)
        V_hat = -4 * np.pi * rho_hat / k2
        V_hat[0,0,0] = 0
        return np.real(np.fft.ifftn(V_hat))
    
    def _advect(self, rho, Fx, Fy, Fz):
        x = np.arange(self.N)
        y = np.arange(self.N)
        z = np.arange(self.N)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        X_new = X - self.dt * Fx / self.dx
        Y_new = Y - self.dt * Fy / self.dx
        Z_new = Z - self.dt * Fz / self.dx
        X_new = np.clip(X_new, 0, self.N-1)
        Y_new = np.clip(Y_new, 0, self.N-1)
        Z_new = np.clip(Z_new, 0, self.N-1)
        X_int = np.round(X_new).astype(int)
        Y_int = np.round(Y_new).astype(int)
        Z_int = np.round(Z_new).astype(int)
        return rho[X_int, Y_int, Z_int]
    
    def _compute_energies(self):
        sqrt_rho1 = np.sqrt(np.maximum(self.rho1, 0))
        sqrt_rho2 = np.sqrt(np.maximum(self.rho2, 0))
        g1x, g1y, g1z = np.gradient(sqrt_rho1, self.dx, axis=(0,1,2))
        g2x, g2y, g2z = np.gradient(sqrt_rho2, self.dx, axis=(0,1,2))
        kin1 = 0.5 * np.sum(g1x**2 + g1y**2 + g1z**2) * self.dv
        kin2 = 0.5 * np.sum(g2x**2 + g2y**2 + g2z**2) * self.dv
        pot1 = np.sum(self.rho1 * self.V_core) * self.dv
        pot2 = np.sum(self.rho2 * self.V_core) * self.dv
        pot_ee = 0.5 * np.sum((self.rho1 + self.rho2) * self.V_ee) * self.dv
        return kin1 + kin2, pot1 + pot2, pot_ee
    
    def run(self):
        print("=" * 60)
        print("RealQM H3+ Simulation")
        print(f"Cores: {self.n_cores} at positions:")
        for i, pos in enumerate(self.core_positions):
            print(f"  Core {i}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) Z={self.Z_eff[i]:.1f}")
        print(f"Grid: {self.N}^3, box +/- {self.L:.1f}")
        print("=" * 60)
        
        for step in range(self.steps):
            rho_total = self.rho1 + self.rho2
            self.V_ee = self._poisson_solve(rho_total)
            self.V_total = self.V_core + self.V_ee
            
            Fx = -np.gradient(self.V_total, self.dx, axis=0)
            Fy = -np.gradient(self.V_total, self.dx, axis=1)
            Fz = -np.gradient(self.V_total, self.dx, axis=2)
            
            self.rho1 = self._advect(self.rho1, Fx, Fy, Fz)
            self.rho2 = self._advect(self.rho2, Fx, Fy, Fz)
            self.rho1 = np.maximum(self.rho1, 0)
            self.rho2 = np.maximum(self.rho2, 0)
            
            mask1 = self.rho1 >= self.rho2
            mask2 = ~mask1
            rho1_new = self.rho1 * mask1
            rho2_new = self.rho2 * mask2
            
            sum1 = np.sum(rho1_new) * self.dv
            sum2 = np.sum(rho2_new) * self.dv
            if sum1 > 0: rho1_new /= sum1
            if sum2 > 0: rho2_new /= sum2
            
            sigma = max(0.15, 0.4 * (1 - step / self.steps))
            self.rho1 = gaussian_filter(rho1_new, sigma=sigma)
            self.rho2 = gaussian_filter(rho2_new, sigma=sigma)
            self.rho1 = np.maximum(self.rho1, 0)
            self.rho2 = np.maximum(self.rho2, 0)
            self.rho1 /= np.sum(self.rho1) * self.dv
            self.rho2 /= np.sum(self.rho2) * self.dv
            
            kin, pot, ee = self._compute_energies()
            total = kin + pot + ee
            overlap = np.sum(self.rho1 * self.rho2) * self.dv
            
            self.history['energy'].append(total)
            self.history['kinetic'].append(kin)
            self.history['potential'].append(pot)
            self.history['ee_repulsion'].append(ee)
            self.history['overlap'].append(overlap)
            
            if step % self.plot_every == 0 or step == self.steps-1:
                print(f"Step {step:4d}: E = {total:10.4f}, T = {kin:8.4f}, V = {pot:8.4f}, Overlap = {overlap:.8f}")
                self._plot_snapshot(step)
        
        print("=" * 60)
        print(f"Final energy: {self.history['energy'][-1]:.6f}")
        print("=" * 60)
        self._plot_final()
        self._save_data()
    
    def _plot_snapshot(self, step):
        iz = self.N // 2
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        slice_rho1 = self.rho1[:, :, iz]
        slice_rho2 = self.rho2[:, :, iz]
        vmax_dens = max(0.15, np.max(slice_rho1) * 1.1, np.max(slice_rho2) * 1.1)
        
        axes[0,0].imshow(slice_rho1.T, origin='lower', extent=[-self.L, self.L, -self.L, self.L],
                         cmap='Blues', vmin=0, vmax=vmax_dens)
        axes[0,0].scatter(self.core_x, self.core_y, c='red', marker='x', s=100)
        axes[0,0].set_title(f'rho1 (step {step})')
        
        axes[0,1].imshow(slice_rho2.T, origin='lower', extent=[-self.L, self.L, -self.L, self.L],
                         cmap='Reds', vmin=0, vmax=vmax_dens)
        axes[0,1].scatter(self.core_x, self.core_y, c='red', marker='x', s=100)
        axes[0,1].set_title(f'rho2 (step {step})')
        
        combined = slice_rho1 + slice_rho2
        axes[0,2].imshow(combined.T, origin='lower', extent=[-self.L, self.L, -self.L, self.L],
                         cmap='hot', vmin=0, vmax=vmax_dens)
        axes[0,2].contour(self.X[:,:,iz], self.Y[:,:,iz], slice_rho1.T, levels=[0.01], colors='blue', linewidths=2)
        axes[0,2].contour(self.X[:,:,iz], self.Y[:,:,iz], slice_rho2.T, levels=[0.01], colors='red', linewidths=2)
        axes[0,2].scatter(self.core_x, self.core_y, c='white', marker='x', s=100)
        axes[0,2].set_title('Combined (Blue=rho1, Red=rho2)')
        
        axes[1,0].plot(self.history['energy'], 'b-')
        axes[1,0].set_xlabel('Iteration')
        axes[1,0].set_ylabel('Total Energy')
        axes[1,0].set_title('Energy Convergence')
        axes[1,0].grid(True)
        
        axes[1,1].plot(self.history['overlap'], 'r-')
        axes[1,1].set_xlabel('Iteration')
        axes[1,1].set_ylabel('Overlap')
        axes[1,1].set_title('Overlap')
        axes[1,1].grid(True)
        
        axes[1,2].plot(self.history['kinetic'], 'g-', label='Kinetic')
        axes[1,2].plot(self.history['potential'], 'b-', label='Core attraction')
        axes[1,2].plot(self.history['ee_repulsion'], 'r-', label='EE repulsion')
        axes[1,2].set_xlabel('Iteration')
        axes[1,2].set_ylabel('Energy Component')
        axes[1,2].set_title('Energy Components')
        axes[1,2].legend()
        axes[1,2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'snapshot_step_{step:04d}.png', dpi=150)
        plt.close()
    
    def _plot_final(self):
        iz = self.N // 2
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        slice_rho1 = self.rho1[:, :, iz]
        slice_rho2 = self.rho2[:, :, iz]
        vmax_dens = max(0.15, np.max(slice_rho1) * 1.1, np.max(slice_rho2) * 1.1)
        
        axes[0,0].imshow(slice_rho1.T, origin='lower', extent=[-self.L, self.L, -self.L, self.L],
                         cmap='Blues', vmin=0, vmax=vmax_dens)
        axes[0,0].scatter(self.core_x, self.core_y, c='red', marker='x', s=100)
        axes[0,0].set_title('Final rho1')
        
        axes[0,1].imshow(slice_rho2.T, origin='lower', extent=[-self.L, self.L, -self.L, self.L],
                         cmap='Reds', vmin=0, vmax=vmax_dens)
        axes[0,1].scatter(self.core_x, self.core_y, c='red', marker='x', s=100)
        axes[0,1].set_title('Final rho2')
        
        combined = slice_rho1 + slice_rho2
        axes[0,2].imshow(combined.T, origin='lower', extent=[-self.L, self.L, -self.L, self.L],
                         cmap='hot', vmin=0, vmax=vmax_dens)
        axes[0,2].contour(self.X[:,:,iz], self.Y[:,:,iz], slice_rho1.T, levels=[0.01], colors='blue', linewidths=2)
        axes[0,2].contour(self.X[:,:,iz], self.Y[:,:,iz], slice_rho2.T, levels=[0.01], colors='red', linewidths=2)
        axes[0,2].scatter(self.core_x, self.core_y, c='white', marker='x', s=100)
        axes[0,2].set_title('Boundary: Blue=rho1, Red=rho2')
        
        axes[1,0].plot(self.history['energy'], 'b-')
        axes[1,0].set_xlabel('Iteration')
        axes[1,0].set_ylabel('Total Energy')
        axes[1,0].set_title('Energy Convergence')
        axes[1,0].grid(True)
        
        axes[1,1].plot(self.history['overlap'], 'r-')
        axes[1,1].set_xlabel('Iteration')
        axes[1,1].set_ylabel('Overlap')
        axes[1,1].set_title('Overlap (should approach 0)')
        axes[1,1].grid(True)
        
        axes[1,2].plot(self.history['kinetic'], 'g-', label='Kinetic')
        axes[1,2].plot(self.history['potential'], 'b-', label='Core attraction')
        axes[1,2].plot(self.history['ee_repulsion'], 'r-', label='EE repulsion')
        axes[1,2].set_xlabel('Iteration')
        axes[1,2].set_ylabel('Energy Component')
        axes[1,2].set_title('Energy Components')
        axes[1,2].legend()
        axes[1,2].grid(True)
        
        plt.tight_layout()
        plt.savefig('final_results.png', dpi=200)
        plt.close()
        print("Final plot saved as 'final_results.png'")
    
    def _save_data(self):
        with open('convergence_data.txt', 'w', encoding='utf-8') as f:
            f.write("# H3+ RealQM Convergence Data\n")
            f.write("# Columns: Step, Energy, Overlap, Kinetic, Potential, EE_Repulsion\n\n")
            for i in range(len(self.history['energy'])):
                f.write(f"{i:6d}  {self.history['energy'][i]:12.6f}  "
                       f"{self.history['overlap'][i]:12.8f}  "
                       f"{self.history['kinetic'][i]:12.6f}  "
                       f"{self.history['potential'][i]:12.6f}  "
                       f"{self.history['ee_repulsion'][i]:12.6f}\n")
        
        with open('simulation_parameters.txt', 'w', encoding='utf-8') as f:
            f.write("# H3+ RealQM Simulation Parameters\n\n")
            f.write(f"Cores: {self.n_cores}\n")
            for i, pos in enumerate(self.core_positions):
                f.write(f"  Core {i}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) Z={self.Z_eff[i]:.1f}\n")
            f.write(f"Grid: {self.N}^3\n")
            f.write(f"Box: +/- {self.L:.1f}\n")
            f.write(f"Steps: {self.steps}\n")
            f.write(f"dt: {self.dt}\n\n")
            f.write(f"Final energy: {self.history['energy'][-1]:.6f}\n")
            f.write(f"Final kinetic: {self.history['kinetic'][-1]:.6f}\n")
            f.write(f"Final potential: {self.history['potential'][-1]:.6f}\n")
            f.write(f"Final EE repulsion: {self.history['ee_repulsion'][-1]:.6f}\n")
            f.write(f"Final overlap: {self.history['overlap'][-1]:.8f}\n")
            if self.history['kinetic'][-1] > 0:
                virial = -self.history['potential'][-1] / self.history['kinetic'][-1]
                f.write(f"\nVirial ratio (-V/T): {virial:.6f}\n")


# ================================================================
# CONVENIENCE FUNCTIONS FOR DIFFERENT GEOMETRIES
# ================================================================

def equilateral_H3(R=1.0, **kwargs):
    """Equilateral triangle H3+."""
    core_positions = [
        (0.0, R, 0.0),
        (-R * np.sqrt(3)/2, -R/2, 0.0),
        (R * np.sqrt(3)/2, -R/2, 0.0)
    ]
    sim = RealQMMultiCore(core_positions, **kwargs)
    sim.run()
    return sim

def linear_H3(d=1.0, **kwargs):
    """Linear H3+."""
    core_positions = [(-d, 0.0, 0.0), (0.0, 0.0, 0.0), (d, 0.0, 0.0)]
    sim = RealQMMultiCore(core_positions, **kwargs)
    sim.run()
    return sim

def isosceles_H3(R=1.0, height=0.8, **kwargs):
    """Isosceles triangle H3+."""
    core_positions = [(-R, 0.0, 0.0), (R, 0.0, 0.0), (0.0, height, 0.0)]
    sim = RealQMMultiCore(core_positions, **kwargs)
    sim.run()
    return sim


# ================================================================
# MAIN: RUN ALL THREE GEOMETRIES
# ================================================================

if __name__ == "__main__":
    
    # Create output directories
    for folder in ['equilateral', 'linear', 'isosceles']:
        os.makedirs(folder, exist_ok=True)
    
    # Run equilateral
    print("\n" + "="*80)
    print("EQUILATERAL H3+ (R=1.0)")
    print("="*80)
    os.chdir('equilateral')
    sim_eq = equilateral_H3(R=1.0, grid_size=41, steps=500, dt=0.0005, plot_every=50)
    os.chdir('..')
    
    # Run linear
    print("\n" + "="*80)
    print("LINEAR H3+ (d=1.0)")
    print("="*80)
    os.chdir('linear')
    sim_lin = linear_H3(d=1.0, grid_size=41, steps=500, dt=0.0005, plot_every=50)
    os.chdir('..')
    
    # Run isosceles
    print("\n" + "="*80)
    print("ISOSCELES H3+ (R=1.0, height=0.8)")
    print("="*80)
    os.chdir('isosceles')
    sim_iso = isosceles_H3(R=1.0, height=0.8, grid_size=41, steps=500, dt=0.0005, plot_every=50)
    os.chdir('..')
    
    print("\n" + "="*80)
    print("ALL SIMULATIONS COMPLETE!")
    print("Results saved in: equilateral/, linear/, isosceles/")
    print("="*80)