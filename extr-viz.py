import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from itertools import permutations
from typing import List, Dict, Set, Tuple

def is_sc_allowed(h, k, l):
    return True

def is_bcc_allowed(h, k, l):
    return (h + k + l) % 2 == 0

def is_fcc_allowed(h, k, l):
    return (all(x % 2 == 0 for x in (h, k, l)) or 
            all(x % 2 == 1 for x in (h, k, l)))

def is_diamond_allowed(h, k, l):
    all_odd = all(x % 2 == 1 for x in (h, k, l))
    all_even = all(x % 2 == 0 for x in (h, k, l))
    sum_multiple_4 = (h + k + l) % 4 == 0
    return all_odd or (all_even and sum_multiple_4)

def is_hcp_allowed(h, k, l):
    if l % 2 == 0:  # L even
        return (h + 2*k) % 3 == 0
    else:  # L odd
        return (h + 2*k) % 3 != 0

def plot_crystal_structure(structure_type: str, a: float = 1.0):
    """
    Plot the crystal structure in 3D.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def add_atom(x, y, z, color='blue', size=100):
        ax.scatter(x, y, z, c=color, s=size)

    def add_unit_cell_edges(a, c=None):
        vertices = np.array([[0,0,0], [a,0,0], [a,a,0], [0,a,0],
                           [0,0,a], [a,0,a], [a,a,a], [0,a,a]])
        edges = [[0,1], [1,2], [2,3], [3,0],
                [4,5], [5,6], [6,7], [7,4],
                [0,4], [1,5], [2,6], [3,7]]
        for edge in edges:
            xs = [vertices[edge[0]][0], vertices[edge[1]][0]]
            ys = [vertices[edge[0]][1], vertices[edge[1]][1]]
            zs = [vertices[edge[0]][2], vertices[edge[1]][2]]
            ax.plot(xs, ys, zs, 'k-', alpha=0.5)

    if structure_type == 'SC':
        add_unit_cell_edges(a)
        for x in [0, a]:
            for y in [0, a]:
                for z in [0, a]:
                    add_atom(x, y, z)

    elif structure_type == 'BCC':
        add_unit_cell_edges(a)
        for x in [0, a]:
            for y in [0, a]:
                for z in [0, a]:
                    add_atom(x, y, z)
        add_atom(a/2, a/2, a/2, color='red')

    elif structure_type == 'FCC':
        add_unit_cell_edges(a)
        
        # Corner atoms (blue)
        for x in [0, a]:
            for y in [0, a]:
                for z in [0, a]:
                    add_atom(x, y, z, color='blue', size=100)
        
        # Face-centered atoms (red)
        # XY faces
        add_atom(a/2, a/2, 0, color='red', size=150)
        add_atom(a/2, a/2, a, color='red', size=150)
        
        # XZ faces
        add_atom(a/2, 0, a/2, color='red', size=150)
        add_atom(a/2, a, a/2, color='red', size=150)
        
        # YZ faces
        add_atom(0, a/2, a/2, color='red', size=150)
        add_atom(a, a/2, a/2, color='red', size=150)

    elif structure_type == 'DIAMOND':
        add_unit_cell_edges(a)
        for x in [0, a]:
            for y in [0, a]:
                for z in [0, a]:
                    add_atom(x, y, z)
        for x in [a/2]:
            for y in [0, a]:
                for z in [0, a]:
                    add_atom(x, y, z, color='red')
        for x in [0, a]:
            for y in [a/2]:
                for z in [0, a]:
                    add_atom(x, y, z, color='red')
        for x in [0, a]:
            for y in [0, a]:
                for z in [a/2]:
                    add_atom(x, y, z, color='red')
        offset = a/4
        for x in [offset, a-offset]:
            for y in [offset, a-offset]:
                for z in [offset, a-offset]:
                    add_atom(x, y, z, color='green')

    elif structure_type == 'HCP':
        # Calculate c parameter (ideal c/a ratio for HCP)
        c = 1.633 * a
        
        # Base hexagon
        hexagon_points = []
        for i in range(6):
            angle = i * np.pi/3
            x = a * np.cos(angle)
            y = a * np.sin(angle)
            hexagon_points.append((x, y))
            
            # Draw vertical edges
            ax.plot([x, x], [y, y], [0, c], 'k-', alpha=0.5)
        
        # Draw hexagonal bases
        for z in [0, c]:
            for i in range(6):
                j = (i + 1) % 6
                x1, y1 = hexagon_points[i]
                x2, y2 = hexagon_points[j]
                ax.plot([x1, x2], [y1, y2], [z, z], 'k-', alpha=0.5)
        
        # Bottom layer atoms (blue)
        for x, y in hexagon_points:
            add_atom(x, y, 0, color='blue', size=100)
        add_atom(0, 0, 0, color='blue', size=100)
        
        # Top layer atoms (blue)
        for x, y in hexagon_points:
            add_atom(x, y, c, color='blue', size=100)
        add_atom(0, 0, c, color='blue', size=100)
        
        # Middle layer atoms (red)
        middle_z = c/2
        for i in range(3):
            angle = (2*i + 1) * np.pi/3
            x = a/2 * np.cos(angle)
            y = a/2 * np.sin(angle)
            add_atom(x, y, middle_z, color='red', size=150)

    ax.set_title(f'{structure_type} Structure')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])
    
    # Make axes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    plt.show()

def get_all_permutations(h, k, l):
    perms = set(permutations([h, k, l]))
    return sorted(list(perms))

def generate_reflections_with_permutations(max_index, structure_type):
    sums_dict = {}
    for h in range(max_index + 1):
        for k in range(max_index + 1):
            for l in range(max_index + 1):
                if h == k == l == 0:
                    continue
                squared_sum = h*h + k*k + l*l
                
                is_allowed = False
                if structure_type == "sc" and is_sc_allowed(h, k, l):
                    is_allowed = True
                elif structure_type == "bcc" and is_bcc_allowed(h, k, l):
                    is_allowed = True
                elif structure_type == "fcc" and is_fcc_allowed(h, k, l):
                    is_allowed = True
                elif structure_type == "diamond" and is_diamond_allowed(h, k, l):
                    is_allowed = True
                elif structure_type == "hcp" and is_hcp_allowed(h, k, l):
                    is_allowed = True
                
                if is_allowed:
                    if squared_sum not in sums_dict:
                        sums_dict[squared_sum] = set()
                    
                    if structure_type == "hcp":
                        sums_dict[squared_sum].add((h, k, l))
                    else:
                        all_perms = get_all_permutations(h, k, l)
                        for perm in all_perms:
                            if structure_type == "sc" and is_sc_allowed(*perm):
                                sums_dict[squared_sum].add(perm)
                            elif structure_type == "bcc" and is_bcc_allowed(*perm):
                                sums_dict[squared_sum].add(perm)
                            elif structure_type == "fcc" and is_fcc_allowed(*perm):
                                sums_dict[squared_sum].add(perm)
                            elif structure_type == "diamond" and is_diamond_allowed(*perm):
                                sums_dict[squared_sum].add(perm)
    
    return sorted([(sum_, list(perms)) for sum_, perms in sums_dict.items()])

def get_structure_sequences(max_index=5, max_value=None):
    sequences = {}
    for structure_type in ["sc", "bcc", "fcc", "diamond", "hcp"]:
        reflections = generate_reflections_with_permutations(max_index, structure_type)
        sequence = [sum_ for sum_, _ in reflections]
        if max_value is not None:
            sequence = [x for x in sequence if x <= max_value]
        sequences[structure_type.upper()] = sequence
    return sequences

def identify_crystal_structure(sigma_values: List[int]) -> Dict[str, Dict]:
    max_sigma = max(sigma_values)
    structure_sequences = get_structure_sequences(max_index=5, max_value=max_sigma)
    
    sigma_set = set(sigma_values)
    matches = {}
    
    for structure, sequence in structure_sequences.items():
        ref_values = set(sequence)
        common_values = sigma_set.intersection(ref_values)
        matches[structure] = {
            'common_values': common_values,
            'allowed_reflections': sequence
        }
    
    return matches

def analyze_and_plot_structure(wavelength: float, two_theta_angles: List[float]):
    print("\nAnalysis Steps:")
    print("\n1. Converting 2θ to θ angles:")
    theta_angles = [angle/2 for angle in two_theta_angles]
    print([round(theta, 2) for theta in theta_angles])
    
    print("\n2. Calculating sin(θ):")
    sin_theta = [np.sin(np.radians(theta)) for theta in theta_angles]
    print([round(sin_t, 8) for sin_t in sin_theta])
    
    print("\n3. Calculating sin²(θ):")
    sin2_theta = [sin_t**2 for sin_t in sin_theta]
    print([round(sin2, 8) for sin2 in sin2_theta])
    
    print("\n4. Calculating sin²(θ)/sin²(θ₁) ratios:")
    sin2_ratios = [sin2/sin2_theta[0] for sin2 in sin2_theta]
    print([round(ratio, 8) for ratio in sin2_ratios])
    
    def find_multiplication_factor(ratios):
        best_factor = None
        min_error = float('inf')
        best_rounded = None
        
        for factor in range(1, 21):
            rounded = [round(ratio * factor) for ratio in ratios]
            error = sum(abs(ratio * factor - round(ratio * factor)) for ratio in ratios)
            
            if error < min_error:
                min_error = error
                best_factor = factor
                best_rounded = rounded
                
        return best_factor, best_rounded, min_error
    
    factor, sigma_values, min_error = find_multiplication_factor(sin2_ratios)
    print(f"\n5. Best multiplication factor: {factor}")
    print(f"   Resulting sigma values: {sigma_values}")
    print(f"   Average error: {min_error/len(sin2_ratios):.6f}")
    
    matches = identify_crystal_structure(sigma_values)
    
    print("\n6. Structure Analysis:")
    best_match = None
    best_percentage = 0
    
    for structure, data in matches.items():
        common_values = sorted(data['common_values'])
        sequence = data['allowed_reflections']
        
        n_common = len(common_values)
        n_sequence = len(sequence)
        n_experimental = len(sigma_values)
        match_percentage = (n_common / (n_sequence + n_experimental - n_common)) * 100
        
        print(f"\n{structure}:")
        print(f"  Allowed reflections: {sequence}")
        print(f"  Common values: {common_values}")
        print(f"  Match percentage: {match_percentage:.1f}%")
        
        if match_percentage > best_percentage:
            best_percentage = match_percentage
            best_match = structure
    
    print(f"\nBest matching structure: {best_match} with {best_percentage:.1f}% match")
    
    # Calculate lattice parameter
    a_values = [wavelength * np.sqrt(sigma/4/sin2) 
                for sigma, sin2 in zip(sigma_values, sin2_theta)]
    print("\n7. Individual lattice constants (Å):")
    print([round(a, 8) for a in a_values])
    
    a_average = float(np.mean(a_values))
    print(f"\n8. Average lattice constant: {round(a_average, 8)} Å")
    
    # Plot the best matching structure
    plot_crystal_structure(best_match, a_average)
    
    return a_average, best_match

def main():
    print("X-ray Diffraction Pattern Analysis and Structure Visualization")
    print("-" * 60)
    
    wavelength = float(input("Enter wavelength (Å): "))
    angles_input = input("Enter 2θ angles separated by commas: ")
    two_theta_angles = [float(angle.strip()) for angle in angles_input.replace(';', ',').split(',')]
    
    print(f"\nInput data:")
    print(f"Wavelength: {wavelength} Å")
    print(f"2θ angles: {[round(angle, 2) for angle in two_theta_angles]}")
    
    a, structure = analyze_and_plot_structure(wavelength, two_theta_angles)
    
    plt.show()  # Ensure plot is displayed
    input("\nPress Enter to close the visualization...")  # Keep window open until user input

if __name__ == "__main__":
    main()
