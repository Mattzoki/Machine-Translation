import os

# Folders to ignore (you can add more here)
IGNORE_FOLDERS = {".git", ".ipynb_checkpoints"}


def generate_tree(start_path, max_files=10, prefix=""):
    tree_lines = []
    items = sorted(os.listdir(start_path))
    files = [f for f in items if os.path.isfile(os.path.join(start_path, f))]
    dirs = [
        d
        for d in items
        if os.path.isdir(os.path.join(start_path, d)) and d not in IGNORE_FOLDERS
    ]

    # List directories first
    for i, d in enumerate(dirs):
        connector = "├── " if i < len(dirs) - 1 or files else "└── "
        tree_lines.append(f"{prefix}{connector}{d}/")
        extension = "│   " if i < len(dirs) - 1 or files else "    "
        tree_lines.extend(
            generate_tree(os.path.join(start_path, d), max_files, prefix + extension)
        )

    # Then list files
    for i, f in enumerate(files[:max_files]):
        connector = "├── " if i < len(files[:max_files]) - 1 else "└── "
        tree_lines.append(f"{prefix}{connector}{f}")

    # Indicate extra files if any
    if len(files) > max_files:
        tree_lines.append(f"{prefix}└── ... ({len(files) - max_files} more files)")

    return tree_lines


if __name__ == "__main__":
    base_folder = os.path.dirname(os.path.abspath(__file__))
    tree = [os.path.basename(base_folder) + "/"]
    tree.extend(generate_tree(base_folder))

    output_file = os.path.join(base_folder, "folder_structure.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(tree))

    print(f"Folder structure saved to {output_file}")
