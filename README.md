# NeuroPhy3D: Neural Network-Enhanced 3D Physics Simulator

NeuroPhy3D is an advanced 3D physics simulation project that combines traditional particle physics with cutting-edge neural network techniques. This project showcases a GPU-accelerated particle system where some particles exhibit learned, intelligent behaviors.

## Features

- 3D visualization of particle physics using PyQt5 and PyQtGraph
- GPU-accelerated computations using PyTorch for high-performance simulation
- Neural network-based state prediction for improved physics calculations
- Smart particles with learned behaviors controlled by neural networks
- Real-time adjustable parameters (gravity, restitution) for dynamic experimentation
- Interactive particle selection and detailed information display
- FPS counter for performance monitoring
- Ability to add new particles by clicking in the 3D view

## Requirements

- Python 3.6+
- PyQt5
- PyQtGraph
- PyOpenGL
- NumPy
- PyTorch (with CUDA support for GPU acceleration)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/NeuroPhy3D.git
   cd NeuroPhy3D
   ```

2. Install the required packages:
   ```
   pip install PyQt5 pyqtgraph PyOpenGL numpy torch
   ```

   Note: For GPU support, make sure to install the CUDA-enabled version of PyTorch.

## Usage

Run the simulation:

```
python neurophy3d_simulator.py
```

### Controls

- Use the mouse to rotate the 3D view
- Click in the 3D view to add new particles
- Use the "Pause/Resume" button to control the simulation
- Adjust gravity and restitution using the sliders
- Select individual particles from the dropdown menu to view their properties

## How it Works

1. The `PhysicsEnvironment` class manages the particle system, including positions, velocities, and interactions.
2. Neural networks (`StatePredictor` and `SmartParticleController`) are used to predict future states and control "smart" particles.
3. The `SimulationWindow` class provides the GUI using PyQt5 and PyQtGraph for 3D rendering.
4. Physics calculations and neural network operations are performed on the GPU using PyTorch for improved performance.

## Customization

You can modify the following parameters in the script:

- `num_particles`: Initial number of particles
- `box_size`: Size of the simulation box
- Neural network architectures in `StatePredictor` and `SmartParticleController` classes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source.
