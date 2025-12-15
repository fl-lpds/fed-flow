"""
Utility script to generate architecture diagrams for existing result folders.
This script can be run to create diagrams for results that were generated before
the architecture diagram feature was added.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.util.graph_utils import generate_architecture_diagram
from app.config.logger import fed_logger


def generate_diagrams_for_existing_results(results_base_path='Results'):
    """
    Scan the Results directory and generate architecture diagrams for any
    result folders that contain a docker-compose.yml but don't have an architecture.png.
    """
    absolute_path = os.path.abspath(results_base_path)
    fed_logger.info(f"Resolved results path: {absolute_path}")
    if not os.path.exists(results_base_path):
        fed_logger.error(f"Results directory not found: {results_base_path}")
        return
    
    # Iterate through all subdirectories in Results
    for result_folder in os.listdir(results_base_path):
        fed_logger.info(f"Processing result folder: {result_folder}")
        result_path = os.path.join(results_base_path, result_folder)
        fed_logger.info(f"Result path: {result_path}")
        if not os.path.isdir(result_path):
            continue
        
        compose_file = os.path.join(result_path, 'docker-compose.yml')
        architecture_diagram = os.path.join(result_path, 'architecture.png')
        
        # Check if docker-compose.yml exists but architecture diagram doesn't
        if os.path.exists(compose_file) and not os.path.exists(architecture_diagram):
            fed_logger.info(f"Generating architecture diagram for: {result_folder}")
            try:
                generate_architecture_diagram(compose_file, result_path, 'architecture')
                fed_logger.info(f"Successfully generated diagram for: {result_folder}")
            except Exception as e:
                fed_logger.error(f"Failed to generate diagram for {result_folder}: {e}")
        elif os.path.exists(architecture_diagram):
            fed_logger.info(f"Architecture diagram already exists for: {result_folder}")
        else:
            fed_logger.warning(f"No docker-compose.yml found in: {result_folder}")
    
    fed_logger.info("Finished processing all result folders")


if __name__ == '__main__':
    # Allow custom results path as command line argument
    results_path = sys.argv[1] if len(sys.argv) > 1 else 'Results'
    fed_logger.info(f"Scanning results directory: {results_path}")
    generate_diagrams_for_existing_results(results_path)

