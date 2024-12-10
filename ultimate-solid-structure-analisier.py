
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')  # Suppress unnecessary warnings


def analyze_diffraction(wavelength: float, two_theta_angles: List[float]) -> Tuple[
    List[float], List[float], List[float], List[float], float, str
]:
    """
    Analyze diffraction data and determine the lattice parameter.
    Returns:
    - theta_angles: list of θ angles.
    - sin_theta: list of sin(θ) values.
    - sin2_theta: list of sin²(θ) values.
    - sin2_ratios: ratios sin²(θ)/sin²(θ_1).
    - a_average: average lattice parameter.
    - structure: identified structure (default 'FCC' for simplicity).
    """

    # Step 1: Convert 2θ to θ
    theta_angles = [angle / 2 for angle in two_theta_angles]

    # Step 2: Calculate sin(θ)
    sin_theta = [np.sin(np.radians(theta)) for theta in theta_angles]

    # Step 3: Calculate sin²(θ)
    sin2_theta = [sin_t**2 for sin_t in sin_theta]

    # Step 4: Calculate ratios of sin²(θ)
    sin2_ratios = [sin2 / sin2_theta[0] for sin2 in sin2_theta]

    # Step 5: Find best multiplication factor
    def find_multiplication_factor(ratios):
        best_factor = 1
        min_error = float('inf')
        best_rounded = None
        for factor in range(1, 21):
            rounded = [round(ratio * factor) for ratio in ratios]
            error = sum(abs(ratio * factor - round(ratio * factor)) for ratio in ratios)
            if error < min_error:
                min_error = error
                best_factor = factor
                best_rounded = rounded
        return best_factor, best_rounded

    factor, sigma_values = find_multiplication_factor(sin2_ratios)

    # Step 6: Calculate lattice parameter
    a_values = [wavelength * np.sqrt(sigma / (4 * sin2)) for sigma, sin2 in zip(sigma_values, sin2_theta)]
    a_average = np.mean(a_values)

    # Default to FCC for simplicity
    structure = 'FCC'

    return theta_angles, sin_theta, sin2_theta, sin2_ratios, a_average, structure


def generate_step_by_step_report(
    wavelength: float, two_theta_angles: List[float],
    analysis: Tuple[List[float], List[float], List[float], List[float], float, str]
):
    """
    Generate a LaTeX-style step-by-step report of the analysis.
    """
    theta_angles, sin_theta, sin2_theta, sin2_ratios, a_average, structure = analysis

    print("\n\\textbf{Pasos del Análisis:}")
    print("\\begin{enumerate}")
    print("\\item \\textbf{Conversión de $2\\theta$ a $\\theta$:}")
    print(f"$[{', '.join([f'{theta:.2f}^\\circ' for theta in theta_angles])}]$")
    print("\\item \\textbf{Valores $\\sin(\\theta)$:}")
    print(f"$[{', '.join([f'{sin_t:.4f}' for sin_t in sin_theta])}]$")
    print("\\item \\textbf{Valores $\\sin^2(\\theta)$:}")
    print(f"$[{', '.join([f'{sin2:.4f}' for sin2 in sin2_theta])}]$")
    print("\\item \\textbf{Relaciones $\\sin^2(\\theta)/\\sin^2(\\theta_1)$:}")
    print(f"$[{', '.join([f'{ratio:.4f}' for ratio in sin2_ratios])}]$")
    print("\\item \\textbf{Parámetro de Red Promedio:}")
    print(f"$a = {a_average:.4f} \\ \\mathrm{{\\AA}}$")
    print("\\item \\textbf{Estructura Identificada:}")
    print(f"{structure}")
    print("\\end{enumerate}")


def plot_structure_ratios(angles: List[float], ratios: List[float]) -> None:
    """
    Plot the sin²(theta) ratios for visualization.
    """
    plt.figure(figsize=(10, 6))
    plt.plot([a/2 for a in angles], ratios, marker='o', label='Relaciones $\\sin^2(\\theta)/\\sin^2(\\theta_1)$')
    plt.xlabel('$\\theta$ (Degrees)')
    plt.ylabel('Relaciones $\\sin^2(\\theta)$')
    plt.title('Análisis de Ratios $\\sin^2(\\theta)$')
    plt.grid(True)
    plt.legend()
    plt.show()


class CrystalVisualizer:
    def __init__(self):
        # Initialize the figure with custom styling
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.patch.set_facecolor('#F5F5F5')
        
        # Initialize variables
        self.atoms = []
        self.atom_size = 100
        self.rotation_speed = 2
        self.is_rotating = False
        self.last_x = 0
        self.last_y = 0
        
        # Color schemes
        self.colors = {
            'corner': '#1976D2',  # Blue
            'center': '#D32F2F',  # Red
            'face': '#388E3C',    # Green
            'edges': '#424242'    # Dark Gray
        }
        
        self.setup_layout()
        
    def setup_layout(self):
        """Set up the main layout including the 3D plot and controls"""
        # Create main 3D axis
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.subplots_adjust(bottom=0.25)
        
        # Configure main plot styling
        self.ax.set_facecolor('white')
        self.ax.grid(False)
        
        # Make panes transparent
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('w')
        self.ax.yaxis.pane.set_edgecolor('w')
        self.ax.zaxis.pane.set_edgecolor('w')
        
        # Add size control slider
        slider_ax = self.fig.add_axes([0.2, 0.1, 0.6, 0.03])
        self.size_slider = Slider(
            slider_ax, 'Atom Size',
            valmin=50, valmax=200,
            valinit=self.atom_size,
            color=self.colors['corner'],
            alpha=0.7
        )
        self.size_slider.on_changed(self.update_atom_size)
        
        # Add event listeners for rotation
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def add_atom(self, x: float, y: float, z: float, color: str = 'corner') -> None:
        """Add an atom to the structure"""
        atom = self.ax.scatter(
            x, y, z,
            c=self.colors.get(color, self.colors['corner']),
            s=self.atom_size,
            alpha=0.8,
            edgecolor='white',
            linewidth=1
        )
        self.atoms.append(atom)
        return atom

    def add_unit_cell_edges(self, a: float, c: float = None) -> None:
        """Draw the unit cell edges"""
        vertices = np.array([
            [0,0,0], [a,0,0], [a,a,0], [0,a,0],
            [0,0,a], [a,0,a], [a,a,a], [0,a,a]
        ])
        
        edges = [
            [0,1], [1,2], [2,3], [3,0],
            [4,5], [5,6], [6,7], [7,4],
            [0,4], [1,5], [2,6], [3,7]
        ]
        
        for edge in edges:
            self.ax.plot3D(
                xs=[vertices[edge[0]][0], vertices[edge[1]][0]],
                ys=[vertices[edge[0]][1], vertices[edge[1]][1]],
                zs=[vertices[edge[0]][2], vertices[edge[1]][2]],
                color=self.colors['edges'],
                alpha=0.3,
                linewidth=1
            )

    def plot_structure(self, structure_type: str, a: float = 1.0) -> None:
        """Plot the selected crystal structure"""
        self.ax.clear()
        self.atoms = []
        
        structure_methods = {
            'SC': self._plot_sc,
            'BCC': self._plot_bcc,
            'FCC': self._plot_fcc,
            'DIAMOND': self._plot_diamond,
            'HCP': self._plot_hcp
        }
        
        if structure_type in structure_methods:
            structure_methods[structure_type](a)
        else:
            raise ValueError(f"Unknown structure type: {structure_type}")
            
        self._style_plot(structure_type)
        
    def _plot_sc(self, a: float) -> None:
        """Plot Simple Cubic structure"""
        self.add_unit_cell_edges(a)
        for x in [0, a]:
            for y in [0, a]:
                for z in [0, a]:
                    self.add_atom(x, y, z, 'corner')

    def _plot_bcc(self, a: float) -> None:
        """Plot Body-Centered Cubic structure"""
        self.add_unit_cell_edges(a)
        # Corner atoms
        for x in [0, a]:
            for y in [0, a]:
                for z in [0, a]:
                    self.add_atom(x, y, z, 'corner')
        # Center atom
        self.add_atom(a/2, a/2, a/2, 'center')

    def _plot_fcc(self, a: float) -> None:
        """Plot Face-Centered Cubic structure"""
        self.add_unit_cell_edges(a)
        # Corner atoms
        for x in [0, a]:
            for y in [0, a]:
                for z in [0, a]:
                    self.add_atom(x, y, z, 'corner')
        # Face-centered atoms
        for x in [a/2]:
            for y in [a/2]:
                for z in [0,a]:
                    self.add_atom(x, y, z, 'face')
        for x in [0, a]:
            for y in [a/2]:
                for z in [a/2]:
                    self.add_atom(x, y, z, 'face')
        for x in [a/2]:
            for y in [0,a]:
                for z in [a/2]:
                    self.add_atom(x, y, z, 'face')

    def _plot_diamond(self, a: float) -> None:
        """Plot Diamond structure (FCC with tetrahedral positions)"""
        self._plot_fcc(a)
        # Add tetrahedral sites
        tetrahedral_positions = [
            (0.25, 0.25, 0.25), (0.75, 0.75, 0.25),
            (0.75, 0.25, 0.75), (0.25, 0.75, 0.75)
        ]
        for x, y, z in tetrahedral_positions:
            self.add_atom(x*a, y*a, z*a, 'center')

    def _plot_hcp(self, a: float) -> None:
        """Plot Hexagonal Close-Packed structure (ideal c/a ratio)"""
        c = 1.633 * a  
        angles = np.linspace(0, 2*np.pi, 7)[:-1]
        base_points = [(a * np.cos(theta), a * np.sin(theta)) for theta in angles]
        
        self.ax.clear()
        self.add_unit_cell_edges(a, c)
        
        # Base and top
        for x, y in base_points:
            self.add_atom(x, y, 0, 'corner')
            self.add_atom(x, y, c, 'corner')
        
        # Middle layer
        middle_points = [
            (a/2 * np.cos(theta + np.pi/6), a/2 * np.sin(theta + np.pi/6))
            for theta in angles[::2]
        ]
        for x, y in middle_points:
            self.add_atom(x, y, c/2, 'center')

    def _style_plot(self, title: str) -> None:
        """Apply styling to the plot"""
        self.ax.set_title(
            f'{title} Structure',
            pad=20,
            fontsize=14,
            fontweight='bold',
            color='#424242'
        )
        
        for axis in [self.ax.xaxis, self.ax.yaxis, self.ax.zaxis]:
            axis.line.set_linewidth(0.5)
            axis.label.set_color('#424242')
            axis.label.set_fontsize(10)
        
        self.ax.set_xlabel('X', labelpad=10)
        self.ax.set_ylabel('Y', labelpad=10)
        self.ax.set_zlabel('Z', labelpad=10)
        self.ax.set_box_aspect([1,1,1])
        
        # Set initial viewing angle
        self.ax.view_init(elev=20, azim=45)
        plt.draw()

    def update_atom_size(self, val: float) -> None:
        """Update the size of all atoms"""
        self.atom_size = val
        for atom in self.atoms:
            atom._sizes = [val]
        plt.draw()

    def on_mouse_press(self, event) -> None:
        """Handle mouse press event for rotation"""
        if event.button == 1 and event.xdata and event.ydata:
            self.is_rotating = True
            self.last_x = event.xdata
            self.last_y = event.ydata

    def on_mouse_release(self, event) -> None:
        """Handle mouse release event"""
        self.is_rotating = False

    def on_mouse_move(self, event) -> None:
        """Handle mouse movement for rotation"""
        if self.is_rotating and event.xdata and event.ydata:
            dx = event.xdata - self.last_x
            dy = event.ydata - self.last_y
            
            current_elev = self.ax.elev
            current_azim = self.ax.azim
            
            self.ax.view_init(
                elev=current_elev + dy * self.rotation_speed,
                azim=current_azim - dx * self.rotation_speed
            )
            
            self.last_x = event.xdata
            self.last_y = event.ydata
            plt.draw()


def analyze_and_plot_structure(wavelength: float, two_theta_angles: List[float]) -> Tuple[float, str]:
    """
    Given the wavelength and 2θ angles, compute the lattice parameter and
    return a plausible structure type. Here we default to 'FCC'.
    """
    theta_angles = [angle / 2 for angle in two_theta_angles]
    sin_theta = [np.sin(np.radians(theta)) for theta in theta_angles]
    sin2_theta = [sin_t**2 for sin_t in sin_theta]
    sin2_ratios = [sin2 / sin2_theta[0] for sin2 in sin2_theta]

    # Find best multiplication factor
    def find_multiplication_factor(ratios):
        best_factor = 1
        min_error = float('inf')
        best_rounded = None
        for factor in range(1, 21):
            rounded = [round(ratio * factor) for ratio in ratios]
            error = sum(abs(ratio * factor - round(ratio * factor)) for ratio in ratios)
            if error < min_error:
                min_error = error
                best_factor = factor
                best_rounded = rounded
        return best_factor, best_rounded

    factor, sigma_values = find_multiplication_factor(sin2_ratios)
    a_values = [wavelength * np.sqrt(sigma / (4 * sin2)) for sigma, sin2 in zip(sigma_values, sin2_theta)]
    a_average = float(np.mean(a_values))
    
    # Default structure assumption
    return a_average, 'FCC'


def main():
    """
    Main function that integrates:
    - Data input from the user
    - Diffraction analysis (steps)
    - Lattice parameter extraction
    - Structure plotting (2D ratio plot and 3D visualization)
    """
    try:
        print("\nX-ray Diffraction Pattern Analysis and Structure Visualization")
        print("-" * 60)

        # User input
        wavelength = float(input("Enter wavelength (Å): ").strip())
        angles_input = input("Enter 2θ angles separated by commas: ").strip()
        two_theta_angles = [float(angle.strip()) for angle in angles_input.replace(';', ',').split(',') if angle.strip()]

        # Perform analysis
        analysis = analyze_diffraction(wavelength, two_theta_angles)
        generate_step_by_step_report(wavelength, two_theta_angles, analysis)
        
        # Extract data for plotting
        _, _, _, sin2_ratios, a_average, structure = analysis
        
        # Plot sin²(θ) ratios
        plot_structure_ratios(two_theta_angles, sin2_ratios)

        # 3D Visualization
        visualizer = CrystalVisualizer()
        visualizer.plot_structure(structure, a_average)

        print(f"\nIdentified structure: {structure}")
        print(f"Calculated lattice parameter: {a_average:.4f} Å")
        print("\nInteractive controls:")
        print("- Use the slider to adjust atom size")
        print("- Click and drag to rotate the structure")
        print("- Close the window to exit")

        plt.show()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()

