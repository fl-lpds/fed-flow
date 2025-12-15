# Results Visualization Features

This document describes the visualization and reporting features available in the Fed-Flow project.

## Overview

The project automatically generates comprehensive visualizations and data exports for each federated learning experiment run.

## Features

### 1. Automatic CSV Export

All numerical results are automatically saved to CSV files for easy analysis in spreadsheet software or custom analysis scripts.

**Location**: `Results/{timestamp} {scenario_description}/results-{node_name}.csv`

**Columns**:
- Round
- Training Time (s)
- Bandwidth (bytes/s)
- Accuracy (%)
- Neighbor Bandwidth (bytes/s) [if applicable]

### 2. Graphical Plots

The following plots are automatically generated:
- Training time per round
- Bandwidth usage per round
- Accuracy per round
- Accuracy over time (cumulative duration)
- Neighbor bandwidth (when available)

**Location**: `Results/{timestamp} {scenario_description}/`

### 3. Architecture Diagrams

Visual network topology diagrams are automatically generated from the docker-compose.yml configuration.

**Features**:
- **Intelligent color coding**:
  - Centralized/Hierarchical architectures:
    - 🔵 Blue: Server nodes (top layer)
    - 🟡 Yellow: Edge server nodes (middle layer)
    - 🎨 Cluster colors: Client nodes colored by edge cluster (different color per cluster)
  - D2D architectures:
    - 🔵 Blue: Server node (center)
    - 🎨 Cluster colors: Clients grouped and colored by cluster membership
- **Hierarchical layouts**:
  - Centralized: Tree structure with server at top, edges in middle, clients at bottom
  - D2D: Radial layout with server at center, clients arranged in clusters around it
- Node size reflects hierarchy (servers > edge > clients)
- Connections show neighbor relationships
- Automatic architecture detection and layout selection

**Location**: `Results/{timestamp} {scenario_description}/architecture.png`

## Usage

### Automatic Generation

All visualizations and exports are created automatically when running federated learning experiments. No additional configuration is required.

### Generating Diagrams for Existing Results

If you have existing result folders without architecture diagrams, you can generate them using:

```bash
python app/util/generate_architecture_diagrams.py
```

Or specify a custom results directory:

```bash
python app/util/generate_architecture_diagrams.py path/to/results
```

## Requirements

The following packages are required (already included in requirements.txt):
- matplotlib
- networkx
- pyyaml
- numpy
- pandas

To install/update dependencies:

```bash
pip install -r requirements.txt
```

## Architecture Diagram Details

### Supported Architectures

The diagram generator supports all federated learning architectures:

1. **Centralized**: Single server with multiple clients
2. **Semi-Decentralized**: Multiple edge servers with their own clients
3. **Decentralized**: Only clients communicating with each other
4. **D2D (Device-to-Device)**: Clustered clients with a central server

### Customization

The diagram generation can be customized by modifying:
- `app/util/graph_utils.py::generate_architecture_diagram()`
- Color scheme: Modify `color_map` dictionary
- Node sizes: Modify `size_map` dictionary
- Layout algorithm: Modify the layout selection logic

## File Structure

After running an experiment, your results folder will contain:

```
Results/{timestamp} {scenario_description}/
├── accuracy-{node}.png           # Accuracy plot
├── accuracy-duration-{node}.png  # Accuracy over time
├── training-time-{node}.png      # Training time plot
├── bandwidth-{node}.png          # Bandwidth plot
├── results-{node}.csv            # Numerical data (NEW)
├── architecture.png              # Network diagram (NEW)
└── docker-compose.yml            # Configuration file
```

## Examples

### Reading CSV Data

Python:
```python
import pandas as pd

df = pd.read_csv('Results/2025-11-24 03:17 /results-server1:8098.csv')
print(df.head())
```

Excel/Google Sheets:
- Open the CSV file directly for analysis and custom visualizations

### Programmatic Diagram Generation

```python
from app.util.graph_utils import generate_architecture_diagram

generate_architecture_diagram(
    compose_file_path='path/to/docker-compose.yml',
    save_path='path/to/output',
    diagram_name='custom_architecture'
)
```

## Troubleshooting

### Diagram Not Generated

1. Ensure docker-compose.yml exists in the results folder
2. Check that NODE_TYPE and NEIGHBORS fields are properly set in the compose file
3. Verify networkx and pyyaml are installed

### CSV Missing Data

- Ensure the experiment completed successfully
- Check log files for any errors during result generation

## Future Enhancements

Potential improvements:
- Interactive diagrams with plotly
- Additional metrics in CSV (loss, convergence rate, etc.)
- Comparison plots across multiple experiments
- Real-time visualization dashboard

