import os
import zipfile
import glob
import concurrent.futures
import multiprocessing
import subprocess
from tqdm import tqdm

# ==========================================
# --- USER CONFIGURATION ---
# ==========================================

# MODE: 'pack', 'unpack', 'upload', or 'push'
MODE = 'upload'

# DELETE_RAW: Delete original files after zipping? (Only for 'pack' and 'upload')
DELETE_RAW = True

# CHUNK SIZE: Target size in MB
CHUNK_SIZE_MB = 50 

# ==========================================

TARGET_DIRS = [
    os.path.join("instances"),
    os.path.join("solutions"),
    os.path.join("visuals", "instances"),
    os.path.join("visuals", "solutions"),
    os.path.join("Generalized_TSP_Analysis", "instances"),
    os.path.join("Generalized_TSP_Analysis", "solutions"),
    os.path.join("Generalized_TSP_Analysis", "visualizations")
]

VALID_EXTENSIONS = {'.json', '.bin', '.png', '.jpg', '.txt', '.sol'}
MAX_BYTES = CHUNK_SIZE_MB * 1024 * 1024

def get_file_size(filepath):
    return os.path.getsize(filepath)

# --- WORKER FUNCTIONS (Must be top-level for multiprocessing) ---

def pack_worker(args):
    zip_path, batch_files, base_directory, delete_raw = args
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file_context:
        for file_path in batch_files:
            relative_name = os.path.relpath(file_path, base_directory)
            zip_file_context.write(file_path, relative_name)
    
    if delete_raw:
        for file_path in batch_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                
    return True

def unpack_worker(args):
    zip_path, extract_dir = args
    
    with zipfile.ZipFile(zip_path, 'r') as zip_file_context:
        zip_file_context.extractall(extract_dir)
        
    return True

# --- MAIN LOGIC ---

def get_packing_tasks(root_dir):
    tasks = []
    
    for relative_path in TARGET_DIRS:
        directory = os.path.join(root_dir, relative_path)
        if not os.path.exists(directory):
            continue

        all_files = []
        for root, _, files in os.walk(directory):
            for file_name in files:
                extension = os.path.splitext(file_name)[1].lower()
                if extension in VALID_EXTENSIONS:
                    all_files.append(os.path.join(root, file_name))
        
        if not all_files:
            continue
            
        all_files.sort()

        current_batch = []
        current_size = 0
        batch_index = 0
        
        dir_name = os.path.basename(directory.rstrip(os.sep))
        parent_name = os.path.basename(os.path.dirname(directory).rstrip(os.sep))
        prefix = f"visuals_{dir_name}" if parent_name == "visuals" else dir_name

        for file_path in all_files:
            file_size = get_file_size(file_path)
            if current_size + file_size > MAX_BYTES and current_batch:
                zip_name = f"{prefix}_part_{batch_index}.zip"
                tasks.append((
                    os.path.join(directory, zip_name),
                    current_batch,
                    directory,
                    DELETE_RAW
                ))
                current_batch = []
                current_size = 0
                batch_index += 1
            
            current_batch.append(file_path)
            current_size += file_size
            
        if current_batch:
            zip_name = f"{prefix}_part_{batch_index}.zip"
            tasks.append((
                os.path.join(directory, zip_name),
                current_batch,
                directory,
                DELETE_RAW
            ))
            
    return tasks

def get_unpacking_tasks(root_dir):
    tasks = []
    
    for relative_path in TARGET_DIRS:
        directory = os.path.join(root_dir, relative_path)
        if not os.path.exists(directory):
            continue
            
        zip_files = glob.glob(os.path.join(directory, "*.zip"))
        for zip_file in zip_files:
            tasks.append((zip_file, directory))
            
    return tasks

def execute_git_lfs_upload():
    print("\nStarting Git LFS tracking and upload process...")
    subprocess.run(["git", "lfs", "install"], check=True)
    subprocess.run(["git", "lfs", "track", "*.zip"], check=True)
    subprocess.run(["git", "add", ".gitattributes"], check=True)
    
    for relative_path in TARGET_DIRS:
        if os.path.exists(relative_path):
            subprocess.run(["git", "add", relative_path], check=True)
            
    subprocess.run(["git", "commit", "-m", "Automated upload of packed archive chunks"], check=True)
    subprocess.run(["git", "push"], check=True)

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    number_of_cores = os.cpu_count()
    print(f"--- Chunk Archiver (Mode: {MODE.upper()} | Cores: {number_of_cores}) ---")
    
    tasks = []
    if MODE in ['pack', 'upload']:
        tasks = get_packing_tasks(root_dir)
        worker_function = pack_worker
        description = "Packing Archives"
    elif MODE == 'unpack':
        tasks = get_unpacking_tasks(root_dir)
        worker_function = unpack_worker
        description = "Extracting Archives"
    elif MODE == 'push':
        print("Skipping packaging/extraction tasks. Proceeding directly to Git LFS upload...")
    else:
        print(f"Invalid mode: {MODE}")
        return

    # Only run the ProcessPoolExecutor if there are tasks to process
    if tasks:
        print(f"Starting execution on {len(tasks)} tasks...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=number_of_cores) as executor:
            list(tqdm(
                executor.map(worker_function, tasks), 
                total=len(tasks), 
                desc=description,
                unit="file"
            ))
    elif MODE != 'push':
        print("No eligible files or tasks found.")
        return

    if MODE in ['upload', 'push']:
        execute_git_lfs_upload()
        
    print("\nDone.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()