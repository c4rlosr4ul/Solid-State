import numpy as np
from itertools import permutations
from typing import List, Dict, Set, Tuple

def get_miller_indices_for_value(squared_sum: int, max_index: int = 5) -> List[Tuple[int, int, int]]:
    """Generate all possible Miller indices combinations for a given h²+k²+l² value."""
    indices = []
    max_single_index = int(np.sqrt(squared_sum)) + 1
    
    for h in range(-max_single_index, max_single_index + 1):
        for k in range(-max_single_index, max_single_index + 1):
            for l in range(-max_single_index, max_single_index + 1):
                if h*h + k*k + l*l == squared_sum:
                    indices.append((h, k, l))
    return sorted(indices)

def check_structure_rules(h: int, k: int, l: int, structure: str) -> bool:
    """Check if Miller indices follow structure-specific rules."""
    if structure == "SC":
        return True
    elif structure == "BCC":
        return (h + k + l) % 2 == 0
    elif structure == "FCC":
        return (all(x % 2 == 0 for x in (h, k, l)) or 
                all(x % 2 == 1 for x in (h, k, l)))
    elif structure == "DIAMOND":
        all_odd = all(x % 2 == 1 for x in (h, k, l))
        all_even = all(x % 2 == 0 for x in (h, k, l))
        return all_odd or (all_even and (h + k + l) % 4 == 0)
    elif structure == "HCP":
        if l % 2 == 0:
            return (h + 2*k) % 3 == 0
        else:
            return (h + 2*k) % 3 != 0
    return False

def analyze_miller_indices(structure: str, common_values: List[int]):
    """Generate and analyze Miller indices for common values in the structure."""
    print(f"\n=== Miller Indices Analysis for {structure} Structure ===")
    print(f"Structure type: {structure}")
    print("\nReflection Rules:")
    
    # Print structure-specific rules
    rules = {
        "SC": "All Miller indices (hkl) are allowed",
        "BCC": "Allowed: h + k + l = 2n (even)\nForbidden: h + k + l = 2n + 1 (odd)",
        "FCC": "Allowed: All h,k,l even or all odd\nForbidden: Mixed even and odd indices",
        "DIAMOND": "Allowed: All h,k,l odd, or all even with h+k+l = 4n\nForbidden: Mixed indices, all even with h+k+l ≠ 4n",
        "HCP": "For l even: h + 2k = 3n\nFor l odd: h + 2k ≠ 3n"
    }
    print(rules[structure])
    
    print("\nAnalysis of Common Values:")
    for squared_sum in sorted(common_values):
        print(f"\nValue h² + k² + l² = {squared_sum}")
        print("Allowed Miller indices (h,k,l):")
        
        # Get all possible Miller indices for this squared sum
        all_indices = get_miller_indices_for_value(squared_sum)
        
        # Filter and group by rules
        allowed_indices = []
        for h, k, l in all_indices:
            if check_structure_rules(h, k, l, structure):
                allowed_indices.append((h, k, l))
        
        # Print results in a organized way
        if allowed_indices:
            # Group by absolute value combinations
            grouped_indices = {}
            for h, k, l in allowed_indices:
                key = tuple(sorted([abs(h), abs(k), abs(l)]))
                if key not in grouped_indices:
                    grouped_indices[key] = []
                grouped_indices[key].append((h, k, l))
            
            # Print grouped results
            for abs_group, indices in sorted(grouped_indices.items()):
                print(f"\n  |{abs_group}| permutations:")
                # Print in columns
                indices_str = [f"({h:2d},{k:2d},{l:2d})" for h, k, l in sorted(indices)]
                col_width = 12
                cols = 6
                for i in range(0, len(indices_str), cols):
                    print("    " + "".join(s.ljust(col_width) for s in indices_str[i:i+cols]))
        else:
            print("  No allowed indices found for this value")

def main():
    # Your existing main code here...
    # After finding the best structure match, add:
    
    # Example usage (replace with your actual values):
    structure = "FCC"  # This should come from your structure identification
    common_values = [3, 4, 8, 11, 12, 16, 19, 20, 24, 27, 32, 35, 36, 40]  # These should be your actual common values
    
    analyze_miller_indices(structure, common_values)

if __name__ == "__main__":
    main()
