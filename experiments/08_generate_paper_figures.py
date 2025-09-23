#!/usr/bin/env python3
# scripts/generate_figures.py
"""
Generate all publication-ready figures and tables for the research paper
"""

import sys
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def main():
    """Generate all figures for the paper"""
    print("🎨 PUBLICATION FIGURE GENERATION")
    print("=" * 50)
    
    try:
        # Import the visualization module
        from src.visualization.paper_figures import PaperFigureGenerator
        
        # Initialize generator
        generator = PaperFigureGenerator()
        
        # Generate all figures
        success = generator.generate_all_figures()
        
        if success:
            print("\n🎯 FIGURE GENERATION COMPLETE!")
            print("📁 Check data/results/paper_figures/ for outputs")
            print("📊 Files generated:")
            
            output_dir = Path("data/results/paper_figures")
            if output_dir.exists():
                for file in output_dir.glob("*"):
                    print(f"   📄 {file.name}")
        
        return success
        
    except Exception as e:
        print(f"❌ Figure generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
