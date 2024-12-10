import numpy as np
from collections import Counter
from itertools import permutations
from typing import List, Dict, Set, Tuple

# Structure rule functions
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

STRUCTURE_RULES = {
    "SC": is_sc_allowed,
    "BCC": is_bcc_allowed,
    "FCC": is_fcc_allowed,
    "DIAMOND": is_diamond_allowed,
    "HCP": is_hcp_allowed
}

STRUCTURE_DESCRIPTIONS = {
    "SC": "All Miller indices (hkl) are allowed",
    "BCC": "Allowed: h + k + l = 2n (even)\nForbidden: h + k + l = 2n + 1 (odd)",
    "FCC": "Allowed: All h,k,l even or all odd\nForbidden: Mixed even and odd indices",
    "DIAMOND": "Allowed: All h,k,l odd, or all even with h+k+l = 4n\nForbidden: Mixed indices, all even with h+k+l ≠ 4n",
    "HCP": "For l even: h + 2k = 3n\nFor l odd: h + 2k ≠ 3n"
}

def get_all_permutations(h, k, l):
    return sorted(list(set(permutations([h, k, l]))))

def generate_reflections_with_permutations(max_index, structure_type):
    sums_dict = {}
    structure_rule = STRUCTURE_RULES[structure_type.upper()]
    
    for h in range(max_index + 1):
        for k in range(max_index + 1):
            for l in range(max_index + 1):
                if h == k == l == 0:
                    continue
                    
                squared_sum = h*h + k*k + l*l
                
                if structure_rule(h, k, l):
                    if squared_sum not in sums_dict:
                        sums_dict[squared_sum] = set()
                    
                    if structure_type.upper() == "HCP":
                        sums_dict[squared_sum].add((h, k, l))
                    else:
                        for perm in get_all_permutations(h, k, l):
                            if structure_rule(*perm):
                                sums_dict[squared_sum].add(perm)
    
    return sorted([(sum_, list(perms)) for sum_, perms in sums_dict.items()])

def analyze_diffraction_pattern(wavelength: float, two_theta_angles: List[float]) -> Dict[str, any]:
    """
    Analyzes an X-ray diffraction pattern and returns structure analysis results.
    """
    theta_angles = [angle/2 for angle in two_theta_angles]
    sin_theta = [np.sin(np.radians(theta)) for theta in theta_angles]
    sin2_theta = [sin_t**2 for sin_t in sin_theta]
    sin2_ratios = [sin2/sin2_theta[0] for sin2 in sin2_theta]
    
    # Find best multiplication factor
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
    
    # Identify crystal structure
    structure_matches = identify_crystal_structure(sigma_values)
    
    # Calculate lattice constants
    a_values = [wavelength * np.sqrt(sigma/4/sin2) 
                for sigma, sin2 in zip(sigma_values, sin2_theta)]
    a_average = float(np.mean(a_values))
    
    return {
        'sigma_values': sigma_values,
        'structure_matches': structure_matches,
        'lattice_constant': a_average,
        'error': min_error/len(sin2_ratios)
    }

def main():
    print("X-ray Diffraction Pattern Analysis")
    print("-" * 50)
    
    wavelength = float(input("Enter wavelength (Å): "))
    angles_input = input("Enter 2θ angles separated by commas: ")
    two_theta_angles = [float(angle.strip()) for angle in angles_input.replace(';', ',').split(',')]    
    
    print(f"\nWavelength: {wavelength} Å")
    
    # Analyze the pattern
    results = analyze_diffraction_pattern(wavelength, two_theta_angles)
    
    # Print results
    print("\nAnalysis Results:")
    print(f"Average lattice constant: {results['lattice_constant']:.8f} Å")
    print(f"Average error: {results['error']:.6f}")
    
    # Find best matching structure
    best_match = max(results['structure_matches'].items(), 
                    key=lambda x: len(x[1]['common_values']))
    structure = best_match[0]
    common_values = best_match[1]['common_values']
    
    # Analyze Miller indices for the best matching structure
    analyze_miller_indices(structure, common_values)

if __name__ == "__main__":
    main()
