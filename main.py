import os
import glob
import json
import torch
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from pathlib import Path
import multiprocessing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("repo_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration (Change these as needed)
input_dir = "./aos/"        # Path to your repository
output_dir = "./save-aos/"       # Path to save the dataset
model_name = "unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit"  # Quantized model for memory efficiency
num_gpus = 5                 # Number of GPUs to use
batch_size = 10              # Number of files to process before saving

# Process all file extensions - no filtering based on extension

# Files to ignore (common patterns to skip)
ignore_patterns = [
    'node_modules', '.git', '.idea', '.vscode', '__pycache__', 
    'venv', 'env', '.env', 'dist', 'build', '.DS_Store', ".wasm"
]

# Maximum file size to process (in bytes)
max_file_size = 300 * 1024  # 100KB

# Create output directory
os.makedirs(output_dir, exist_ok=True)
checkpoint_dir = os.path.join(output_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

def setup_model_on_gpu(gpu_id, model_name):
    """Load the model on a specific GPU"""
    try:
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Loading model on GPU {gpu_id}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Check if using a BitsAndBytes quantized model
        if "bnb" in model_name or "4bit" in model_name or "8bit" in model_name:
            print(f"Using quantized model {model_name}")
            
            # Import bitsandbytes if needed
            try:
                import bitsandbytes as bnb
                print("bitsandbytes is installed")
            except ImportError:
                print("Warning: bitsandbytes is not installed. Installing it may improve performance with quantized models.")
            
            # Load quantized model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                trust_remote_code=True,
            )
        else:
            # Load standard model with fp16
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=torch.float16  # Use fp16 to fit in GPU memory
            )
        
        return {"model": model, "tokenizer": tokenizer, "device": device, "gpu_id": gpu_id}
    except Exception as e:
        print(f"Error loading model on GPU {gpu_id}: {str(e)}")
        return None

def get_file_type(file_path):
    """Determine the type of file based on extension and content heuristics"""
    ext = os.path.splitext(file_path)[1].lower()
    
    # Common extensions by category
    code_extensions = ['.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp', '.cs', 
                      '.go', '.rs', '.rb', '.php', '.html', '.css', '.jsx', '.tsx',
                      '.swift', '.kt', '.m', '.sh', '.bat', '.ps1']
    
    doc_extensions = ['.md', '.txt', '.rst', '.org', '.adoc', '.wiki', '.tex']
    
    data_extensions = ['.json', '.yaml', '.yml', '.toml', '.xml', '.csv', '.tsv', 
                      '.xls', '.xlsx', '.db', '.sqlite', '.dat']
    
    binary_extensions = ['.bin', '.exe', '.dll', '.so', '.dylib', '.class', '.jar',
                        '.war', '.ear', '.pyc', '.pyo', '.o', '.obj',
                        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.ico',
                        '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.wav',
                        '.zip', '.tar', '.gz', '.7z', '.rar', '.pdf', '.doc', '.docx',
                        '.ppt', '.pptx', '.xls', '.xlsx']
    
    if ext in binary_extensions:
        return "binary"
    elif ext in code_extensions:
        return "code"
    elif ext in doc_extensions:
        return "documentation"
    elif ext in data_extensions:
        return "data"
    else:
        # For unknown extensions, try to infer from content later
        return "unknown"

def get_prompt_template(file_type, file_ext):
    """Get the appropriate prompt template based on file type"""
    if file_type == "code":
        return """Below is source code. Please analyze it and provide:
1. A high-level summary of what this code does
2. Key functions/classes and their purposes
3. Any notable design patterns, algorithms, or techniques used
4. Potential issues or areas for improvement

Source Code:
{file_content}

Analysis:
"""
    elif file_type == "documentation":
        return """Below is documentation content. Please analyze it and provide a structured summary
of the key concepts, examples, and technical information it contains.

Documentation Content:
{file_content}

Summary:
"""
    elif file_type == "data":
        return """Below is structured data. Please analyze it and provide:
1. The overall structure and schema of this data
2. Key entities and their relationships
3. Any notable patterns or insights from the data structure

Data Content:
{file_content}

Analysis:
"""
    elif file_type == "binary":
        return """This is a binary file (not text-based) with file extension '{file_ext}'.
Based on the filename and extension, provide a short description of what this file
likely contains and its purpose in the repository.

Filename: {file_path}

Analysis:
"""
    else:
        return """Below is content from a file. Please analyze it and provide a summary
of what this file contains and its likely purpose in the repository.

File Content:
{file_content}

Analysis:
"""

def process_file(file_path, model_info):
    """Process a single file with the model"""
    try:
        gpu_id = model_info["gpu_id"]
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        device = model_info["device"]
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > max_file_size:
            print(f"File too large ({file_size} bytes): {file_path}, skipping")
            return None
        
        # Get the file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Initial file type determination based on extension
        file_type = get_file_type(file_path)
        
        # For binary files, skip processing
        if file_type == "binary":
            print(f"Binary file detected: {file_path}, skipping")
            return None
        
        # Try to read the file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try another common encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except Exception:
                print(f"Unable to read file: {file_path}, skipping")
                return None
                
        # For unknown file types, try to infer based on content
        if file_type == "unknown":
            # Simple heuristics to determine file type from content
            if any(pattern in content[:1000] for pattern in ['<html', '<!DOCTYPE', '<xml']):
                file_type = "code"  # HTML or XML
            elif any(pattern in content[:1000] for pattern in ['import ', 'function ', 'class ', 'def ', '#include']):
                file_type = "code"
            elif any(pattern in content[:1000] for pattern in ['{', ':', '":', ']', '}']):
                if content.strip()[0] in ['{', '['] and content.strip()[-1] in ['}', ']']:
                    file_type = "data"  # Likely JSON or similar
            else:
                file_type = "documentation"  # Default for text files
        
        # Skip empty files
        if not content.strip():
            print(f"Empty file: {file_path}, skipping")
            return None
        
        # Create prompt with the file content
        prompt_template = get_prompt_template(file_type, file_ext)
        prompt = prompt_template.format(file_content=content, file_ext=file_ext, file_path=file_path)
        
        # Generate using the model
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
        
        # Create dataset entry
        result = {
            "file_path": str(file_path),
            "file_type": file_type,
            "file_extension": file_ext,
            "content": content,
            "processed_content": response,
            "timestamp": time.time()
        }
        
        return result
    
    except Exception as e:
        print(f"Error processing file {file_path} on GPU {gpu_id}: {str(e)}")
        return None

def save_batch(batch_results, output_dir, batch_num):
    """Save a batch of results to a JSON file"""
    if not batch_results:
        return
    
    output_file = os.path.join(output_dir, f"batch_{batch_num}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, ensure_ascii=False, indent=2)
    
    print(f"Saved batch {batch_num} with {len(batch_results)} entries to {output_file}")

def should_process_file(file_path):
    """Check if a file should be processed based on ignore patterns"""
    # Check if file matches any ignore pattern
    for pattern in ignore_patterns:
        if pattern in str(file_path):
            return False
    
    # Process all files regardless of extension
    return True

def find_files(directory):
    """Find all files to process in the directory recursively"""
    files_to_process = []
    
    for root, dirs, files in os.walk(directory):
        # Skip directories that match ignore patterns
        dirs[:] = [d for d in dirs if not any(pattern in d for pattern in ignore_patterns)]
        
        for file in files:
            file_path = os.path.join(root, file)
            if should_process_file(file_path):
                files_to_process.append(file_path)
    
    print(f"Found {len(files_to_process)} files to process in {directory}")
    return files_to_process

def process_files_with_gpu(files, gpu_id, model_name, output_dir, batch_size, checkpoint_file):
    """Process a subset of files on a specific GPU"""
    # Load the latest checkpoint if it exists
    processed_files = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed_files = set(json.load(f))
        print(f"Loaded {len(processed_files)} processed files from checkpoint on GPU {gpu_id}")
    
    # Filter out already processed files
    files_to_process = [f for f in files if str(f) not in processed_files]
    print(f"GPU {gpu_id} has {len(files_to_process)} files to process")
    
    if not files_to_process:
        print(f"No files to process on GPU {gpu_id}")
        return
    
    # Set up model on this GPU
    model_info = setup_model_on_gpu(gpu_id, model_name)
    if model_info is None:
        print(f"Failed to set up model on GPU {gpu_id}")
        return
    
    # Process files in batches
    batch_results = []
    batch_num = 0
    
    for i, file_path in enumerate(tqdm(files_to_process, desc=f"GPU {gpu_id}")):
        result = process_file(file_path, model_info)
        
        if result:
            batch_results.append(result)
            processed_files.add(str(file_path))
            
            # Save checkpoint of processed files
            with open(checkpoint_file, 'w') as f:
                json.dump(list(processed_files), f)
            
            # Save batch if we've reached batch_size
            if len(batch_results) >= batch_size:
                batch_output_dir = os.path.join(output_dir, f"gpu_{gpu_id}")
                os.makedirs(batch_output_dir, exist_ok=True)
                save_batch(batch_results, batch_output_dir, batch_num)
                batch_results = []
                batch_num += 1
    
    # Save any remaining results
    if batch_results:
        batch_output_dir = os.path.join(output_dir, f"gpu_{gpu_id}")
        os.makedirs(batch_output_dir, exist_ok=True)
        save_batch(batch_results, batch_output_dir, batch_num)
    
    print(f"GPU {gpu_id} completed processing")

# Find all files to process
files_to_process = find_files(input_dir)

# Check available GPUs
available_gpus = min(torch.cuda.device_count(), num_gpus)
if available_gpus == 0:
    print("No GPUs available")
else:
    print(f"Using {available_gpus} GPUs")
    
    # Distribute files across GPUs
    files_per_gpu = [[] for _ in range(available_gpus)]
    for i, file_path in enumerate(files_to_process):
        gpu_idx = i % available_gpus
        files_per_gpu[gpu_idx].append(file_path)
    
    # This function helps run the process in a separate process
    def run_gpu_process(gpu_id):
        checkpoint_file = os.path.join(checkpoint_dir, f"gpu_{gpu_id}_checkpoint.json")
        process_files_with_gpu(
            files_per_gpu[gpu_id],
            gpu_id,
            model_name,
            output_dir,
            batch_size,
            checkpoint_file
        )
    
    # Create a process for each GPU
    processes = []
    for gpu_id in range(available_gpus):
        p = multiprocessing.Process(target=run_gpu_process, args=(gpu_id,))
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("All processing complete")
    
    # Combine all batches into a final dataset
    final_dataset = []
    for gpu_id in range(available_gpus):
        gpu_dir = os.path.join(output_dir, f"gpu_{gpu_id}")
        if os.path.exists(gpu_dir):
            batch_files = glob.glob(os.path.join(gpu_dir, "batch_*.json"))
            for batch_file in batch_files:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                    final_dataset.extend(batch_data)
    
    # Save final dataset
    final_output = os.path.join(output_dir, "final_dataset.json")
    with open(final_output, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Final dataset with {len(final_dataset)} entries saved to {final_output}")

# Generate statistics about processed files
if os.path.exists(os.path.join(output_dir, "final_dataset.json")):
    with open(os.path.join(output_dir, "final_dataset.json"), 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Count files by type
    file_types = {}
    file_extensions = {}
    
    for entry in dataset:
        file_type = entry.get("file_type", "unknown")
        file_extension = entry.get("file_extension", "unknown")
        
        file_types[file_type] = file_types.get(file_type, 0) + 1
        file_extensions[file_extension] = file_extensions.get(file_extension, 0) + 1
    
    # Save statistics
    stats = {
        "total_files": len(dataset),
        "file_types": file_types,
        "file_extensions": file_extensions
    }
    
    with open(os.path.join(output_dir, "statistics.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print("Statistics generated and saved to statistics.json")
