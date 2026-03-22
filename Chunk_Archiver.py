import os
import zipfile
import glob
import argparse
from tqdm import tqdm
import concurrent.futures
import multiprocessing

# ==========================================
# --- USER CONFIGURATION ---
# ==========================================

# MODE: 'pack' or 'unpack'
MODE = 'pack'  

# DELETE_RAW: Delete original files after zipping? (Only for 'pack')
DELETE_RAW = True

# CHUNK SIZE: Target size in MB
CHUNK_SIZE_MB = 50 

# ==========================================

TARGET_DIRS = [
    os.path.join("instances"),
    os.path.join("solutions"),
    os.path.join("visuals", "instances"),
    os.path.join("visuals", "solutions")
]

VALID_EXTENSIONS = {'.json', '.bin', '.png', '.jpg', '.txt', '.sol'}
MAX_BYTES = CHUNK_SIZE_MB * 1024 * 1024

def get_file_size(filepath):
    return os.path.getsize(filepath)

# --- WORKER FUNCTIONS (Must be top-level for multiprocessing) ---

def _pack_worker(args):
    """
    Worker process to create a single zip chunk.
    """
    zip_path, batch_files, base_directory, delete_raw = args
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in batch_files:
                # Store relative to the specific directory
                arcname = os.path.relpath(file_path, base_directory)
                zf.write(file_path, arcname)
        
        if delete_raw:
            for file_path in batch_files:
                try:
                    os.remove(file_path)
                except OSError:
                    pass
        return True
    except Exception as e:
        print(f"Error packing {zip_path}: {e}")
        return False

def _unpack_worker(args):
    """
    Worker process to extract a single zip chunk.
    """
    zip_path, extract_dir = args
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        return True
    except Exception as e:
        print(f"Error unpacking {zip_path}: {e}")
        return False

# --- MAIN LOGIC ---

def get_packing_tasks(root_dir):
    tasks = []
    print("\n[SCANNING] Generating file batches...")
    
    for rel_path in TARGET_DIRS:
        directory = os.path.join(root_dir, rel_path)
        if not os.path.exists(directory):
            continue

        # 1. Gather files
        all_files = []
        for root, _, files in os.walk(directory):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in VALID_EXTENSIONS:
                    all_files.append(os.path.join(root, f))
        
        if not all_files:
            continue
            
        all_files.sort() # Deterministic order

        # 2. Batching
        current_batch = []
        current_size = 0
        batch_idx = 0
        
        # Naming logic
        dir_name = os.path.basename(directory.rstrip(os.sep))
        parent_name = os.path.basename(os.path.dirname(directory).rstrip(os.sep))
        prefix = f"visuals_{dir_name}" if parent_name == "visuals" else dir_name

        for f_path in all_files:
            f_size = get_file_size(f_path)
            if current_size + f_size > MAX_BYTES and current_batch:
                zip_name = f"{prefix}_part_{batch_idx}.zip"
                tasks.append((
                    os.path.join(directory, zip_name),
                    current_batch,
                    directory,
                    DELETE_RAW
                ))
                current_batch = []
                current_size = 0
                batch_idx += 1
            
            current_batch.append(f_path)
            current_size += f_size
            
        if current_batch:
            zip_name = f"{prefix}_part_{batch_idx}.zip"
            tasks.append((
                os.path.join(directory, zip_name),
                current_batch,
                directory,
                DELETE_RAW
            ))
            
    return tasks

def get_unpacking_tasks(root_dir):
    tasks = []
    for rel_path in TARGET_DIRS:
        directory = os.path.join(root_dir, rel_path)
        if not os.path.exists(directory):
            continue
            
        zips = glob.glob(os.path.join(directory, "*.zip"))
        for z in zips:
            tasks.append((z, directory))
    return tasks

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    num_cores = os.cpu_count()
    print(f"--- Chunk Archiver (Mode: {MODE.upper()} | Cores: {num_cores}) ---")
    
    tasks = []
    
    if MODE == 'pack':
        tasks = get_packing_tasks(root_dir)
        worker_func = _pack_worker
        desc = "Packing Archives"
    elif MODE == 'unpack':
        tasks = get_unpacking_tasks(root_dir)
        worker_func = _unpack_worker
        desc = "Extracting Archives"
    else:
        print(f"Invalid mode: {MODE}")
        return

    if not tasks:
        print("No tasks found.")
        return

    print(f"Starting execution on {len(tasks)} tasks...")
    
    # Parallel Execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Use list(tqdm(...)) to force iteration and display progress
        results = list(tqdm(
            executor.map(worker_func, tasks), 
            total=len(tasks), 
            desc=desc,
            unit="file"
        ))

    print("\nDone.")

if __name__ == "__main__":
    # Windows support for multiprocessing requires freeze_support
    multiprocessing.freeze_support()
    main()