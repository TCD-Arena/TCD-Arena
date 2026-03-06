#!/bin/bash

# This script counts the number of subfolders in each subfolder of a given path
# Usage: ./count_files_in_subfolders.sh <path>
# Example: ./count_files_in_subfolders.sh /home/user/documents

# Check if path argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide a path as an argument"
    echo "Usage: $0 <path>"
    echo "Example: $0 /home/user/documents"
    exit 1
fi

TARGET_PATH="$1"

# Check if the provided path exists
if [ ! -d "$TARGET_PATH" ]; then
    echo "Error: Directory '$TARGET_PATH' does not exist"
    exit 1
fi

echo "Counting subfolders in each subfolder of: $TARGET_PATH"
echo "================================================"

# Initialize total counter
total_subfolders=0

# Find all subdirectories and count subfolders in each
find "$TARGET_PATH" -mindepth 1 -maxdepth 1 -type d | sort | while read -r subfolder; do
    # Count subdirectories (not files) in the subfolder
    subfolder_count=$(find "$subfolder" -mindepth 1 -maxdepth 1 -type d | wc -l)
    folder_name=$(basename "$subfolder")
    
    printf "%-40s %s subfolders\n" "$folder_name:" "$subfolder_count"
done

echo "================================================"

# Calculate total subfolders across all subfolders
total_subfolders=$(find "$TARGET_PATH" -mindepth 2 -maxdepth 2 -type d | wc -l)
echo "Total subfolders in all subfolders: $total_subfolders"

# Count direct subfolders in the target directory
direct_subfolders=$(find "$TARGET_PATH" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "Direct subfolders in '$TARGET_PATH': $direct_subfolders"
