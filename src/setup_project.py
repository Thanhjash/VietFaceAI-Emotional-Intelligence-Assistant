# D:\3.Project\VGU\CS_AGE\face_analysis\src\setup_project.py

import sys
from pathlib import Path

# Add CS_AGE to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from face_analysis.src.utils.project_structure import ProjectStructureManager

if __name__ == "__main__":
    # Initialize and setup project structure
    project_manager = ProjectStructureManager("D:/3.Project/VGU/CS_AGE")
    
    # Create directory structure
    project_manager.create_directory_structure()
    
    # Setup data source configuration
    project_manager.setup_data_source()
    
    # Validate structure
    if project_manager.validate_structure():
        project_manager.save_structure_info()