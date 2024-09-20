# MDScatter

**MDScatter** is a Python-based toolkit for analyzing molecular dynamics (MD) trajectory files of molecular materials, such as solvated electrolytes or solutions. The toolkit allows for comprehensive cluster distribution statistical analysis and small-angle X-ray scattering (SAXS) calculations based on these distributions.

This repository provides functionality for processing MD trajectories, identifying clusters, analyzing their statistical properties, and calculating SAXS profiles from the resulting cluster distributions.

## Key Features
1. **MD Trajectory File Conversion**: Converts MD trajectory files (e.g., `.xyz` or `.pdb` formats) into a format suitable for cluster analysis.
2. **Cluster Network Identification**: Identifies clusters in the molecular system based on user-defined parameters and generates PDB files for each cluster.
3. **Cluster Distribution Statistical Analysis**: Performs statistical analysis on the cluster distributions generated in the cluster identification step, including distribution statistics and coordination numbers.
4. **SAXS Calculations and Fitting**: Estimates and fits SAXS profiles from cluster distribution data derived from MD simulations.

## Notebooks

The repository contains four primary Jupyter notebooks, each designed for specific stages of the workflow:

### 1. **File Conversion**
   This notebook allows you to convert MD trajectory files (in `.xyz` or `.pdb` format) into a format usable for further cluster analysis. The converted files are prepared for efficient cluster identification and distribution analysis.

### 2. **Cluster Network Identification**
   In this notebook, you can parse clusters from the molecular system based on user-defined parameters such as cutoff distances and target atoms. This step generates PDB files for each identified cluster, which can be used for further analysis.

### 3. **Cluster Distribution Statistical Analysis**
   This notebook provides tools for calculating statistical information on the clusters identified in the previous step. You can calculate cluster size distributions, coordination numbers, bond lengths, and angles.

### 4. **SAXS Calculations and Fitting**
   Here, you can estimate SAXS profiles from the cluster distribution data generated across the MD trajectory. The notebook also allows for fitting the SAXS data to experimental scattering profiles.

## Core Classes

### 1. **PDBFileHandler**
   - Responsible for handling PDB files, parsing atoms, and writing updated PDB files.
   - **Methods**:
     - `read_pdb_file()`: Reads the PDB file and separates core and shell atoms.
     - `write_pdb_file()`: Writes the parsed atom data to a new PDB file.
     - `update_residue_names()`: Updates the residue names of atoms based on input.
     - `print_atom_details()`: Prints the details of atoms including ID, name, element, and coordinates.

### 2. **Atom**
   - Represents an individual atom in the molecular system.
   - **Attributes**:
     - `atom_id`: Unique identifier for the atom.
     - `atom_name`: The atom’s name.
     - `residue_name`: The residue to which the atom belongs.
     - `residue_number`: The residue's numerical identifier.
     - `coordinates`: A tuple containing the x, y, z coordinates of the atom.
     - `element`: The chemical element of the atom.
     - `network_id`: Identifier for the cluster (or network) the atom belongs to.

### 3. **ClusterNetwork**
   - Manages the identification and analysis of clusters (networks) in the molecular system.
   - **Methods**:
     - `analyze_networks()`: Identifies unique clusters based on user-defined parameters.
     - `get_connected_atoms()`: Finds atoms connected to a given atom within a distance threshold.
     - `rename_clusters_in_pdb()`: Assigns unique residue names to each identified cluster and writes the updated PDB file.
     - `calculate_coordination_numbers()`: Computes coordination numbers based on user-defined atom pairs and cutoff distances.
     - `calculate_bond_lengths_within_network()`: Measures bond lengths within identified clusters.
     - `calculate_bond_angles_within_network()`: Computes bond angles for specified atom triplets within clusters.
     - `visualize_networks()`: Generates a 3D plot for visualizing the identified clusters.
     - `calculate_and_plot_distributions()`: Plots bond length and angle distributions for clusters.

## Installation

To use this toolkit, clone the repository and install the necessary Python packages:

```bash
git clone https://github.com/kewh5868/MDScatter.git
cd MDScatter
pip install -r requirements.txt
