import subprocess
from concurrent.futures import ThreadPoolExecutor

def optimize_file(file_path):
    executable = "C:/Program Files/FileOptimizer/FileOptimizer64.exe"
    subprocess.run([executable, file_path], check=True)

def optimize_files(files, num_threads):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(optimize_file, files)
