import os
from tqdm.notebook import tqdm

class TrajectoryProcessor:
    
    def __init__(self, input_file, base_dir, output_folder_name):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
        - input_file (str): Path to the input trajectory file (.xyz or .pdb).
        - base_dir (str): Base directory where the output folder will be created.
        - output_folder_name (str): Name of the folder to be created in the base directory for storing output files.
        """
        self.input_file = input_file
        self.output_dir = os.path.join(base_dir, output_folder_name)
        
        # Check file extension and set mode
        if input_file.endswith('.xyz'):
            self.file_type = 'xyz'
        elif input_file.endswith('.pdb'):
            self.file_type = 'pdb'
        else:
            raise ValueError("Unsupported file type. Please provide a .xyz or .pdb file.")

        # Ensure output directory does not already exist to avoid overwriting
        if os.path.exists(self.output_dir):
            raise FileExistsError(f"The directory {self.output_dir} already exists. Choose a different directory or folder name.")
        else:
            os.makedirs(self.output_dir)

    def estimate_frame_count_xyz(self):
        """
        Estimate the number of frames in the XYZ trajectory file by counting the 'frame' lines.
        
        Returns:
        - int: The total number of frames in the XYZ trajectory file.
        """
        frame_count = 0
        with open(self.input_file, 'r') as infile:
            for line in infile:
                if line.startswith('frame'):
                    frame_count += 1
        return frame_count

    def estimate_frame_count_pdb(self):
        """
        Estimate the number of frames in the PDB trajectory file by counting the 'MODEL' lines.
        
        Returns:
        - int: The total number of frames in the PDB trajectory file.
        """
        frame_count = 0
        with open(self.input_file, 'r') as infile:
            for line in infile:
                if line.startswith('MODEL'):
                    frame_count += 1
        return frame_count

    def split_and_preprocess_xyz(self):
        """
        Split an XYZ trajectory file into frames, preprocess them, and save only the preprocessed frames.
        """
        if self.file_type != 'xyz':
            raise ValueError("This method is only applicable for XYZ files.")
        
        # Estimate the number of frames in the XYZ file
        total_frames = self.estimate_frame_count_xyz()

        # Step 1: Split and preprocess the XYZ trajectory in one go, showing progress
        self.split_and_preprocess_xyz_frames(total_frames)

    def split_and_preprocess_xyz_frames(self, total_frames):
        """
        Split the XYZ trajectory file, preprocess the frames, and save them as 'frame_' with a progress bar.
        
        Parameters:
        - total_frames (int): The total number of frames estimated from the trajectory file.
        """
        with open(self.input_file, 'r') as infile:
            lines = infile.readlines()

        frame_num = 0
        frame_data = []
        in_frame = False
        atom_count = 0

        # Use tqdm progress bar to track frame processing
        with tqdm(total=total_frames, desc="Processing XYZ frames", unit="frame") as pbar:
            for line in lines:
                if line.strip().isdigit():
                    atom_count = int(line.strip())
                    continue
                if line.startswith('frame'):
                    if in_frame and frame_data:
                        # Preprocess and write the frame data directly
                        output_file = os.path.join(self.output_dir, f'frame_{frame_num:04d}.xyz')
                        self.preprocess_xyz_frame(frame_data, output_file, atom_count)
                        pbar.update(1)  # Update progress bar
                    frame_num = int(line.split()[1])
                    frame_data = [line]
                    in_frame = True
                elif in_frame:
                    frame_data.append(line)

            # Write and preprocess the last frame
            if frame_data:
                output_file = os.path.join(self.output_dir, f'frame_{frame_num:04d}.xyz')
                self.preprocess_xyz_frame(frame_data, output_file, atom_count)
                pbar.update(1)  # Update progress bar for the last frame

    def preprocess_xyz_frame(self, frame_data, output_file, atom_count):
        """
        Preprocess an XYZ frame, removing numeric suffixes from atom labels, and save it.

        Parameters:
        - frame_data (list): List of lines containing the XYZ frame data.
        - output_file (str): Path where the preprocessed XYZ frame will be saved.
        - atom_count (int): The number of atoms in the frame.
        """
        with open(output_file, 'w') as outfile:
            outfile.write(f"{atom_count}\n")
            outfile.write(frame_data[0])  # Comment line
            
            # Process each atom line in the frame
            for line in frame_data[1:]:
                parts = line.split()
                if len(parts) != 4:
                    continue
                atom_label = ''.join([i for i in parts[0] if not i.isdigit()]).capitalize()
                outfile.write(f"{atom_label:>4} {parts[1]:>10} {parts[2]:>10} {parts[3]:>10}\n")

    def split_pdb_trajectory(self):
        """
        Splits a PDB trajectory file into individual frames and saves them as separate .pdb files with a progress bar.
        """
        if self.file_type != 'pdb':
            raise ValueError("This method is only applicable for PDB files.")
        
        # Estimate the number of frames in the PDB file
        total_frames = self.estimate_frame_count_pdb()

        # Use tqdm progress bar to track frame processing
        with open(self.input_file, 'r') as infile:
            frame_data = []
            frame_num = 0

            with tqdm(total=total_frames, desc="Processing PDB frames", unit="frame") as pbar:
                for line in infile:
                    if line.startswith('MODEL'):
                        frame_num = int(line.split()[1])
                        frame_data = [line]  # Start a new frame
                    elif line.startswith('ENDMDL') or line.startswith('END'):
                        frame_data.append(line)
                        
                        # Check if frame contains only 'END' or 'ENDMDL' and skip if so
                        if len(frame_data) > 2:  # Must contain more than MODEL and ENDMDL/END
                            output_file = os.path.join(self.output_dir, f'frame_{frame_num:04d}.pdb')
                            with open(output_file, 'w') as outfile:
                                outfile.write(''.join(frame_data))
                        frame_data = []
                        pbar.update(1)  # Update progress bar
                    else:
                        if line.startswith('ATOM') or line.startswith('HETATM'):
                            atom_id = int(line[6:11].strip())
                            atom_name = line[12:16].strip()
                            residue_name = line[17:20].strip()
                            chain_id = line[21:22].strip()
                            residue_number = int(line[22:26].strip())
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())
                            element = self.get_element_name(atom_name)
                            line = self.format_pdb_line(atom_id, atom_name, residue_name, chain_id, residue_number, x, y, z, element)
                        frame_data.append(line)

                # Handle case where file does not end with ENDMDL/END
                if frame_data and len(frame_data) > 2:
                    output_file = os.path.join(self.output_dir, f'frame_{frame_num:04d}.pdb')
                    with open(output_file, 'w') as outfile:
                        outfile.write(''.join(frame_data))
                    pbar.update(1)  # Update progress bar for the last frame

    def get_element_name(self, atom_name):
        """
        Extract the element name from the atom name by stripping any digits.
        
        Parameters:
        - atom_name (str): The atom name (e.g., 'PB1', 'I1').
        
        Returns:
        - str: The corresponding element name (e.g., 'Pb', 'I').
        """
        element = ''.join([char for char in atom_name if not char.isdigit()])
        return element.capitalize()

    def format_pdb_line(self, atom_id, atom_name, residue_name, chain_id, residue_number, x, y, z, element):
        """
        Formats a PDB line ensuring proper spacing and alignment.
        
        Parameters:
        - atom_id (int): Atom ID.
        - atom_name (str): Atom name.
        - residue_name (str): Residue name.
        - chain_id (str): Chain identifier.
        - residue_number (int): Residue sequence number.
        - x (float): X coordinate.
        - y (float): Y coordinate.
        - z (float): Z coordinate.
        - element (str): Element symbol.
        
        Returns:
        - str: Properly formatted PDB line.
        """
        return (f"ATOM  {atom_id:5d}  {atom_name:<4}{residue_name:<3} {chain_id}{residue_number:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:<2}\n")
