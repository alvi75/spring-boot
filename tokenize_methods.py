import os
import re
import time
from tqdm import tqdm
from multiprocessing import Process, Queue

# Function to extract method signatures and bodies from a Java file
def extract_methods_from_file(file_path, queue):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

            # Refined regex for Java method extraction
            method_pattern = re.compile(
                r'\b(public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\(.*?\)\s*\{', re.DOTALL)

            methods = method_pattern.findall(content)
            queue.put([method[1] for method in methods])  # Extract method names
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        queue.put([])

# Function to find all Java files in a directory
def find_java_files(directory):
    java_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.java'):
                java_files.append(os.path.join(root, file))
    return java_files

# Function to tokenize method body (split into meaningful tokens)
def tokenize_method(method):
    tokens = re.findall(r'\w+|\S', method)
    return tokens

# Timeout function for method extraction
def extract_methods_with_timeout(file_path, timeout=30):
    queue = Queue()
    process = Process(target=extract_methods_from_file, args=(file_path, queue))
    process.start()
    process.join(timeout)

    # If the process exceeds the timeout, terminate it
    if process.is_alive():
        print(f"Terminating long-running process for file {file_path}")
        process.terminate()
        process.join()

    if not queue.empty():
        return queue.get()
    else:
        return []

# Directory set to the current folder
directory = '.'

# Get all Java files
java_files = find_java_files(directory)

# Set a file size threshold to skip large files (e.g., 10 MB)
FILE_SIZE_THRESHOLD = 10 * 1024 * 1024  # 10 MB

# Open file to write extracted methods
with open("tokenized_methods.txt", "w") as outfile:
    total_methods = 0
    skipped_files = 0

    for java_file in tqdm(java_files, desc="Processing Java Files", unit="file"):
        try:
            file_size = os.path.getsize(java_file)
            if file_size > FILE_SIZE_THRESHOLD:
                print(f"Skipping {java_file} due to large file size: {file_size / (1024 * 1024):.2f} MB")
                skipped_files += 1
                continue

            # Extract methods with a timeout
            methods = extract_methods_with_timeout(java_file, timeout=30)

            if len(methods) == 0:
                continue  # Skip files with no valid methods

            total_methods += len(methods)
            for method in methods:
                tokens = tokenize_method(method)
                if len(tokens) > 0:
                    outfile.write(' '.join(tokens) + '\n')
                else:
                    print(f"Method has no tokens: {method}")

        except Exception as e:
            print(f"Error processing {java_file}: {e}")

print(f"Total methods extracted: {total_methods}")
print(f"Files skipped due to size or no methods: {skipped_files}")
print("Method signatures saved to 'tokenized_methods.txt'")
