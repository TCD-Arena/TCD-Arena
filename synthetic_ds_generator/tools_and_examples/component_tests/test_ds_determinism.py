#!/usr/bin/env python3
"""
Test Dataset Determinism

This script takes a folder path containing subfolders and hashes each file
in the subfolders to compare if the folders are identical. This is useful
for verifying that dataset generation or experiment outputs are deterministic
and produce identical results across different runs.

Usage:
    python test_ds_determinism.py <folder_path> [--algorithm sha256] [--verbose]
    python test_ds_determinism.py <folder1> <folder2> [--algorithm sha256]
"""

import sys
import hashlib
import argparse
from pathlib import Path
from typing import Dict
import json
import numpy as np
import pickle


class DatasetDeterminismTester:
    """Test determinism of datasets by comparing file hashes across folder structures."""
    
    def __init__(self, only_check):
        """
        Initialize the dataset determinism tester.
        
        Args:
            hash_algorithm: Hash algorithm to use (md5, sha1, sha256, sha512)
            verbose: Whether to print detailed progress information
        """
        self.hash_algorithm = "sha256"
        self.only_check = only_check
        self.verbose = False
        self.tolerance = 4  # tolerance for numpy array comparisons

    
    def hash_file(self, file_path: Path) -> str:
        """
        Calculate hash of a single file.
        
        Args:
            file_path: Path to the file to hash
            
        Returns:
            Hexadecimal hash string of the file contents
        """
        hasher = getattr(hashlib, self.hash_algorithm)()
        
        if "X.npy" in str(file_path):
            try:
                # We might run into numpy precision issues here, so we load the file and compare with a tolerance.
                f = np.load(file_path)
                f = np.round(f, self.tolerance)  # round to avoid precision issues
                # Compute a stable cryptographic hash of the array bytes (avoid Python's randomized hash())
                hasher.update(f.tobytes())
                hash_out = hasher.hexdigest()
                return hash_out
            except Exception as e:
                print(f"Error hashing numpy file {file_path}: {e}")
                return f"ERROR: {str(e)}"
        
        else:
            try:
                with open(file_path, 'rb') as f:
                    # Read file in chunks to handle large files efficiently
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
                return hasher.hexdigest()
            except Exception as e:
                print(f"Error hashing file {file_path}: {e}")
                return f"ERROR: {str(e)}"
        
    def scan_folder_structure(self, folder_path: Path) -> Dict[str, Dict[str, str]]:
        """
        Scan folder structure and hash all files in subfolders. 
        IMPORTANT: THIS REQUIRES SUBFOLDERS WITH the same names THAT WILL BE SORTED TO ENSURE CONSISTENT ORDERING.
        
        Args:
            folder_path: Root folder path to scan
            
        Returns:
            Dictionary mapping subfolder names to dictionaries of file hashes
            Format: {subfolder_name: {relative_file_path: hash}}
        """
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        if not folder_path.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")
        
        folder_hashes = {}
        
        
        # Get all subdirectories
        subdirs = [d for d in folder_path.iterdir() if d.is_dir()]
        
        if not subdirs:
            print(f"Warning: No subdirectories found in {folder_path}")
            return folder_hashes
        
        for n,subdir in enumerate(sorted(subdirs)):  # Sort for consistent ordering
            # Recursively find all files in this subdirectory
            hash_stack = []
            for file_path in sorted(subdir.rglob('*')):  # Sort for consistent ordering
                # remove config files as the might be differing depending on the pathing.
                if "config.yaml" in str(file_path):
                    continue
                if self.only_check == "ALL": 
                    # Calculate relative path from the subdirectory
                    file_hash = self.hash_file(file_path)
                    hash_stack.append(file_hash)
                else:
                    if self.only_check in str(file_path):
                        file_hash = self.hash_file(file_path)
                        hash_stack.append(file_hash)
            folder_hashes[n] = hash_stack
            
        return folder_hashes
    
    def compare_folder_structures(self, folder1_hashes: list, 
                                 folder2_hashes: list) -> bool:
        """
        Compare two folder hash structures for identical content.
        
        Args:
            folder1_hashes: Hash results from first folder
            folder2_hashes: Hash results from second folder
            
        Returns:
            True if folders are identical, False otherwise
        """
        # Sanity checks.      
        if len(folder1_hashes) != len(folder2_hashes):
            print("unequal length between folders")
            return False
        
        no_match = []
        
        for x in folder1_hashes.keys():
          
            subgroup1 = set(folder1_hashes[x])
            subgroup2 = set(folder2_hashes[x])
            diff = len(subgroup1 - subgroup2)
            if diff :
                no_match.append(x)

        if len(no_match) == 0 :
            print(f"  ✓ All {len(folder1_hashes)}  folders are identical")           
            return True
        else:
            print(f"  ✗, the following folders differ: {no_match}")
            return False

    
def main():
    """Main function to run the dataset determinism tester.

    Examples:
    # Compare two folders directly
    python test_ds_determinism.py --compare /path/to/folder1 /path/to/folder2
    # Load and compare wth saved hashes
    python test_ds_determinism.py /path/to/folder2 --compare-with hashes.json
    """
    parser = argparse.ArgumentParser(
    description="Test dataset determinism by comparing file hashes across folder structures",
    formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('folder_path1', nargs='?', 
                       help='Path to folder containing subfolders to analyze')
    parser.add_argument('folder_path2', nargs='?',
                       help='Path to folder containing subfolders to analyze')  
    parser.add_argument('--use_saved', action='store_true',
                       help='Compare folder with previously saved hashes')
    parser.add_argument('--save', action='store_true',
                       help='Save hash results to JSON file')
    parser.add_argument('--save_path', default="../data_hashes/ds_hashes.p",
                       help='Save hash results to JSON file')
    parser.add_argument('--only_check', default="ALL",
                       help='restrict comparisons')

    args = parser.parse_args()
    
    # Leverages previously saved hash

    if args.use_saved: 
        

        folder1_path = Path(args.folder_path1)
        folder1_datasets = sorted([d for d in folder1_path.iterdir() if d.is_dir()])

        saved_hashes = pickle.load(open(args.save_path, "rb"))
        
        if len(folder1_datasets) != len(saved_hashes):
            print("The two folders have an unequal number of datasets! ")
            return False
        
        tester = DatasetDeterminismTester(only_check=args.only_check)
        res = []
        save = []
        for x in range(len(folder1_datasets)):
            print(f"Comparing dataset {x+1}/{len(folder1_datasets)}:")
            p1 = folder1_datasets[x]  
            folder1_hashes = tester.scan_folder_structure(p1)
            has_compare = tester.compare_folder_structures(folder1_hashes,saved_hashes[x])
            if has_compare:
                res.append(has_compare)

        print(len(res), "datasets of", len(folder1_datasets),  "equal.")


    # compares two folders directly  
    else: 
        
        folder1_path = Path(args.folder_path1)
        folder1_datasets = sorted([d for d in folder1_path.iterdir() if d.is_dir()])

        folder2_path = Path(args.folder_path2)
        folder2_datasets = sorted([d for d in folder2_path.iterdir() if d.is_dir()])

        if len(folder1_datasets) != len(folder2_datasets):
            print("The two folders have an unequal number of datasets.")
            return False
        
    
        tester = DatasetDeterminismTester(only_check=args.only_check)
        res = []
        save = []
        for x in range(len(folder1_datasets)):
            print(f"Comparing dataset {x+1}/{len(folder1_datasets)}:", )
            p1 = folder1_datasets[x]
            p2 = folder2_datasets[x]
            print(f"Scanning folder 1: {folder1_datasets[x]}")
            

            folder1_hashes = tester.scan_folder_structure(p1)
            folder2_hashes = tester.scan_folder_structure(p2)
            save.append(folder1_hashes)
            has_compare = tester.compare_folder_structures(folder1_hashes, folder2_hashes)
            if has_compare:
                res.append(has_compare)


        print(len(res), "datasets of", len(folder1_datasets),  "equal.")

        if args.save:
            # save the first one for later comparison: 
            pickle.dump(save, open(args.save_path, "wb"))
            print("Hash results saved to: ", args.save_path)


if __name__ == "__main__":
    main()
