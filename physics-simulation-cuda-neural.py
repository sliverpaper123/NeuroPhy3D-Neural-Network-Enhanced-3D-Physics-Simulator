import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QSlider, QLabel, QComboBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QVector3D
import pyqtgraph.opengl as gl
import sys
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Particle:
    def __init__(self, id, position, velocity, mass, size, is_smart=False):
        self.id = id
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.size = size
        self.kinetic_energy = 0.5 * mass * torch.sum(velocity**2)
        self.is_smart = is_smart

class StatePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StatePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        ).to(device)

    def forward(self, x):
        return self.network(x)

class SmartParticleController(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SmartParticleController, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()  # Output in range [-1, 1]
        ).to(device)

    def forward(self, x):
        return self.network(x)

class PhysicsEnvironment:
    def __init__(self, num_particles, box_size, dt=0.001, gravity=9.8):
        self.num_particles = num_particles
        self.box_size = box_size
        self.dt = dt
        self.gravity_magnitude = gravity
        self.gravity = torch.tensor([0, -gravity, 0], device=device)
        self.restitution = 0.8

        # Initialize particle properties as tensors
        self.positions = torch.rand((num_particles, 3), device=device) * box_size
        self.velocities = (torch.rand((num_particles, 3), device=device) - 0.5) * 10
        self.masses = torch.rand(num_particles, device=device) * 0.5 + 0.5
        self.sizes = torch.rand(num_particles, device=device) * 0.3 + 0.2
        self.is_smart = torch.zeros(num_particles, dtype=torch.bool, device=device)
        self.is_smart[:num_particles//10] = True  # 10% of particles are smart

        # Initialize neural networks
        self.state_predictor = StatePredictor(6, 32, 6).to(device)
        self.smart_controller = SmartParticleController(6, 32, 3).to(device)
        
        self.optimizer_predictor = optim.Adam(self.state_predictor.parameters(), lr=0.001)
        self.optimizer_controller = optim.Adam(self.smart_controller.parameters(), lr=0.001)

    def add_particle(self, position=None, is_smart=False):
        if position is None:
            position = torch.rand(3, device=device) * self.box_size
        else:
            position = torch.tensor(position, device=device)

        self.positions = torch.cat([self.positions, position.unsqueeze(0)])
        self.velocities = torch.cat([self.velocities, ((torch.rand(3, device=device) - 0.5) * 10).unsqueeze(0)])
        self.masses = torch.cat([self.masses, torch.rand(1, device=device) * 0.5 + 0.5])
        self.sizes = torch.cat([self.sizes, torch.rand(1, device=device) * 0.3 + 0.2])
        self.is_smart = torch.cat([self.is_smart, torch.tensor([is_smart], device=device)])
        self.num_particles += 1  # Increment the particle count

    def update(self):
        # Apply gravity and smart controller
        self.velocities += torch.where(self.is_smart.unsqueeze(1),
                                       self.smart_controller(torch.cat([self.positions, self.velocities], dim=1)),
                                       self.gravity) * self.dt

        # Update positions
        self.positions += self.velocities * self.dt

        # Handle box collisions
        for dim in range(3):
            below_min = self.positions[:, dim] < self.sizes
            above_max = self.positions[:, dim] > self.box_size - self.sizes
            self.positions[below_min, dim] = self.sizes[below_min]
            self.positions[above_max, dim] = self.box_size - self.sizes[above_max]
            self.velocities[torch.logical_or(below_min, above_max), dim] *= -self.restitution

        # Handle particle collisions (simplified, non-optimized version)
        for i in range(self.num_particles):
            distances = torch.norm(self.positions - self.positions[i], dim=1)
            collisions = (distances < (self.sizes + self.sizes[i])) & (distances > 0)
            if collisions.any():
                normals = (self.positions[i] - self.positions[collisions]) / distances[collisions].unsqueeze(1)
                relative_velocities = self.velocities[i] - self.velocities[collisions]
                impulses = (-(1 + self.restitution) * torch.sum(relative_velocities * normals, dim=1) /
                            (1/self.masses[i] + 1/self.masses[collisions]))
                self.velocities[i] += torch.sum(impulses.unsqueeze(1) * normals / self.masses[i], dim=0)
                self.velocities[collisions] -= impulses.unsqueeze(1) * normals / self.masses[collisions].unsqueeze(1)

    def train_networks(self):
        # Train state predictor
        sample_size = min(100, self.num_particles)
        indices = torch.randperm(self.num_particles)[:sample_size]
        current_states = torch.cat([self.positions[indices], self.velocities[indices]], dim=1)
        next_states = current_states + torch.cat([self.velocities[indices], self.gravity.repeat(sample_size, 1)], dim=1) * self.dt
        predicted_next_states = self.state_predictor(current_states)
        
        predictor_loss = nn.MSELoss()(predicted_next_states, next_states)
        self.optimizer_predictor.zero_grad()
        predictor_loss.backward(retain_graph=True)
        self.optimizer_predictor.step()

        # Train smart controller
        smart_indices = torch.where(self.is_smart)[0]
        if len(smart_indices) > 0:
            sample_size = min(50, len(smart_indices))
            smart_sample = smart_indices[torch.randperm(len(smart_indices))[:sample_size]]
            states = torch.cat([self.positions[smart_sample], self.velocities[smart_sample]], dim=1)
            actions = self.smart_controller(states)
            
            # Define a simple reward function (e.g., stay in the center of the box)
            target_position = torch.tensor([self.box_size/2, self.box_size/2, self.box_size/2], device=device)
            rewards = -torch.norm(self.positions[smart_sample] - target_position, dim=1)
            
            controller_loss = -rewards.mean()  # We want to maximize reward, so we minimize negative reward
            self.optimizer_controller.zero_grad()
            controller_loss.backward()
            self.optimizer_controller.step()

        # Detach tensors to free the computation graph
        self.positions = self.positions.detach()
        self.velocities = self.velocities.detach()

    def get_state(self):
        kinetic_energies = 0.5 * self.masses * torch.sum(self.velocities**2, dim=1)
        return self.positions, self.velocities, self.sizes, kinetic_energies, self.is_smart
class SimulationWindow(QMainWindow):
    def __init__(self, physics_env):
        super().__init__()
        self.physics_env = physics_env
        self.initUI()
        self.last_update = time.time()
        self.frame_count = 0
        self.paused = False
        self.selected_particle = None

    # ... (rest of the SimulationWindow class remains the same)
    def initUI(self):
        self.setWindowTitle('Physics Simulation with Neural Networks')
        self.setGeometry(100, 100, 1200, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 3D view
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 20  # Adjust the initial camera distance
        self.view.mousePressEvent = self.mouse_press_event
        main_layout.addWidget(self.view, 4)

        positions, velocities, sizes, energies, is_smart = self.physics_env.get_state()
        colors = self.get_particle_colors(energies, is_smart)
        self.scatter = gl.GLScatterPlotItem(pos=positions.cpu().numpy(), size=sizes.cpu().numpy()*10, color=colors)
        self.view.addItem(self.scatter)

        cube = gl.GLBoxItem(size=QVector3D(self.physics_env.box_size, self.physics_env.box_size, self.physics_env.box_size))
        cube.setColor((0.2, 0.2, 0.2, 0.1))
        self.view.addItem(cube)

        # Control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel, 1)

        # Pause/Resume button
        self.pause_button = QPushButton('Pause')
        self.pause_button.clicked.connect(self.toggle_pause)
        control_layout.addWidget(self.pause_button)

        # Gravity slider
        gravity_layout = QHBoxLayout()
        gravity_layout.addWidget(QLabel('Gravity:'))
        self.gravity_slider = QSlider(Qt.Horizontal)
        self.gravity_slider.setRange(0, 200)
        self.gravity_slider.setValue(int(self.physics_env.gravity_magnitude * 10))
        self.gravity_slider.valueChanged.connect(self.update_gravity)
        gravity_layout.addWidget(self.gravity_slider)
        control_layout.addLayout(gravity_layout)

        # Restitution slider
        restitution_layout = QHBoxLayout()
        restitution_layout.addWidget(QLabel('Restitution:'))
        self.restitution_slider = QSlider(Qt.Horizontal)
        self.restitution_slider.setRange(0, 100)
        self.restitution_slider.setValue(int(self.physics_env.restitution * 100))
        self.restitution_slider.valueChanged.connect(self.update_restitution)
        restitution_layout.addWidget(self.restitution_slider)
        control_layout.addLayout(restitution_layout)

        # Particle selector
        self.particle_selector = QComboBox()
        self.particle_selector.addItems([f"Particle {i}" for i in range(self.physics_env.num_particles)])
        self.particle_selector.currentIndexChanged.connect(self.select_particle)
        control_layout.addWidget(self.particle_selector)

        # Particle info display
        self.particle_info = QLabel("Select a particle to view its info")
        control_layout.addWidget(self.particle_info)

        self.fps_label = QLabel('FPS: 0')
        control_layout.addWidget(self.fps_label)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1)

        self.view.mousePressEvent = self.mouse_press_event

    def get_particle_colors(self, energies, is_smart):
        norm_energies = (energies - energies.min()) / (energies.max() - energies.min() + 1e-6)
        colors = torch.zeros((len(energies), 4), device=energies.device)
        colors[:, 0] = norm_energies  # Red channel based on energy
        colors[:, 1] = is_smart.float()  # Green channel for smart particles
        colors[:, 2] = 1 - norm_energies  # Blue channel inverse of energy
        colors[:, 3] = 1  # Alpha channel
        return colors.cpu().numpy()

    def update(self):
        if not self.paused:
            self.physics_env.update()
            self.physics_env.train_networks()
        positions, velocities, sizes, energies, is_smart = self.physics_env.get_state()
        colors = self.get_particle_colors(energies, is_smart)
        self.scatter.setData(pos=positions.cpu().numpy(), size=sizes.cpu().numpy()*10, color=colors)

        if self.selected_particle is not None:
            position = positions[self.selected_particle]
            velocity = velocities[self.selected_particle]
            energy = energies[self.selected_particle]
            smart = is_smart[self.selected_particle]
            self.particle_info.setText(f"Particle {self.selected_particle}:\n"
                                       f"Position: {position.cpu().numpy().round(2)}\n"
                                       f"Velocity: {velocity.cpu().numpy().round(2)}\n"
                                       f"Energy: {energy.item():.2f}\n"
                                       f"Smart: {'Yes' if smart else 'No'}")

        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_update >= 1:
            fps = self.frame_count / (current_time - self.last_update)
            self.fps_label.setText(f'FPS: {fps:.2f}')
            self.frame_count = 0
            self.last_update = current_time
    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.setText('Resume' if self.paused else 'Pause')

    def update_gravity(self):
        new_gravity = self.gravity_slider.value() / 10
        self.physics_env.gravity_magnitude = new_gravity
        self.physics_env.gravity = torch.tensor([0, -new_gravity, 0], device=device)
    def update_restitution(self):
        self.physics_env.restitution = self.restitution_slider.value() / 100

    def mouse_press_event(self, event):
        if event.button() == Qt.LeftButton:
            # Get the mouse position in widget coordinates
            mouse_pos = event.pos()

            # Get the widget size
            width = self.view.width()
            height = self.view.height()

            # Normalize mouse coordinates
            x = (2.0 * mouse_pos.x() - width) / width
            y = (height - 2.0 * mouse_pos.y()) / height

            # Get camera position and orientation
            cpos = self.view.cameraPosition()
            center = self.view.opts['center']
            up = self.view.opts['up']

            # Calculate camera's right vector
            forward = (center - cpos).normalized()
            right = QVector3D.crossProduct(forward, up).normalized()

            # Calculate the ray direction
            ray_dir = (forward + right * x + up * y).normalized()

            # Set a fixed distance for new particles
            distance = 10.0  # Adjust this value as needed

            # Calculate new particle position
            new_particle_pos = cpos + ray_dir * distance

            # Add a new particle at the calculated position
            self.physics_env.add_particle(new_particle_pos.toTuple())
            self.particle_selector.addItem(f"Particle {self.physics_env.num_particles - 1}")
        
    def select_particle(self, index):
        self.selected_particle = index
        
if __name__ == '__main__':
    num_particles = 10  # Increased number of particles
    box_size = 5

    env = PhysicsEnvironment(num_particles=num_particles, box_size=box_size)

    app = QApplication(sys.argv)
    window = SimulationWindow(env)
    window.show()
    sys.exit(app.exec_())
