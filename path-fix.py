#!/usr/bin/env python3
"""
Path Reference Fix

This script specifically addresses the 'local variable Path referenced before assignment' error
in the training code by ensuring Path is properly imported and accessible in all methods.

Usage:
    python fix_path.py
"""
import os
import re
import logging
import shutil
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PathFix")

def find_training_panel():
    """Find the training_panel.py file"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Possible locations
    candidates = [
        os.path.join(base_dir, "ui", "training_panel.py"),
        os.path.join(base_dir, "io_wake_word", "ui", "training_panel.py"),
        os.path.join(str(Path.home()), ".io", "ui", "training_panel.py"),
    ]
    
    for path in candidates:
        if os.path.exists(path):
            logger.info(f"Found training_panel.py at: {path}")
            return path
    
    logger.error("Could not find training_panel.py")
    return None

def fix_path_reference(file_path):
    """Fix Path references in the file"""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    # Create backup
    backup_path = file_path + ".pathfix.bak"
    try:
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup at: {backup_path}")
    except Exception as e:
        logger.warning(f"Could not create backup: {e}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Ensure Path is imported at the top level
    if "from pathlib import Path" not in content:
        # Add after other imports
        import_section = re.search(r"import\s+.*?(?=\n\n|\nclass|\ndef)", content, re.DOTALL)
        if import_section:
            new_imports = import_section.group(0) + "\nfrom pathlib import Path\n"
            content = content.replace(import_section.group(0), new_imports)
            logger.info("Added Path import at the top level")
        else:
            # Add at the top after docstring
            docstring = re.search(r'""".*?"""', content, re.DOTALL)
            if docstring:
                new_content = content.replace(docstring.group(0), docstring.group(0) + "\nfrom pathlib import Path\n")
                content = new_content
                logger.info("Added Path import after docstring")
            else:
                # Last resort, add at the very top
                content = "from pathlib import Path\n" + content
                logger.info("Added Path import at the beginning of the file")
    
    # Find the TrainingThread class and ensure Path is used properly within it
    training_thread_match = re.search(r"class TrainingThread\(.*?\):(.*?)(?=\n\nclass|\Z)", content, re.DOTALL)
    if training_thread_match:
        thread_class = training_thread_match.group(0)
        
        # Check if Path is used in run method without global scope
        run_method = re.search(r"def run\(self\):(.*?)(?=\n    def|\Z)", thread_class, re.DOTALL)
        if run_method and "Path(" in run_method.group(1) and "from pathlib import Path" not in run_method.group(1):
            # Fix Path references in run method
            fixed_run = run_method.group(0).replace(
                "def run(self):",
                "def run(self):\n        # Ensure Path is available\n        from pathlib import Path"
            )
            content = content.replace(run_method.group(0), fixed_run)
            logger.info("Added local Path import in run method")
    
    # Find all methods that reference Path
    method_pattern = r"def (\w+)\(.*?\):(.*?)(?=\n    def|\n\nclass|\Z)"
    for method_match in re.finditer(method_pattern, content, re.DOTALL):
        method_name = method_match.group(1)
        method_body = method_match.group(2)
        
        # Check if method uses Path without importing it
        if "Path(" in method_body and "from pathlib import Path" not in method_body:
            # Add Path import to the method
            fixed_method = method_match.group(0).replace(
                f"def {method_name}(",
                f"def {method_name}(\n        # Ensure Path is available\n        from pathlib import Path\n        "
            )
            content = content.replace(method_match.group(0), fixed_method)
            logger.info(f"Added local Path import in {method_name} method")
    
    # Fix any direct references to Path.home()
    if "Path.home()" in content:
        # This ensures Path.home() works even if Path isn't in the local scope
        content = content.replace(
            "Path.home()",
            "__import__('pathlib').Path.home()"
        )
        logger.info("Fixed direct Path.home() references")
    
    # Write the fixed content back
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Successfully fixed Path references in {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error writing to file: {e}")
        return False

def main():
    """Main function"""
    logger.info("Path Reference Fix")
    logger.info("This script fixes 'local variable Path referenced before assignment' errors")
    
    # Find training_panel.py
    panel_path = find_training_panel()
    if not panel_path:
        print("ERROR: Could not find training_panel.py")
        return
    
    # Fix Path references
    success = fix_path_reference(panel_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print(" PATH REFERENCE FIX - SUMMARY ")
    print("=" * 60)
    
    if success:
        print("\n✅ Successfully fixed Path references in training_panel.py")
        print("\nThe 'local variable Path referenced before assignment' error should be resolved.")
    else:
        print("\n❌ Could not fix Path references")
        print("\nManual intervention required:")
        print("1. Open training_panel.py")
        print("2. Add 'from pathlib import Path' at the top of the file")
        print("3. Check all methods that use Path and ensure it's properly imported")
    
    print("\nNext steps:")
    print("  1. Try training your model again")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
