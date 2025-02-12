import os
import subprocess
import platform

def run_trec_eval(qrel_file, result_file, eval_tool="trec_eval"):
    """Runs trec_eval or dyn_eval depending on the OS."""
    if platform.system() == "Windows":
        eval_tool = "trec_eval.exe"  # Use dyn_eval for Windows
    
    if not os.path.exists(eval_tool):
        print(f"Error: {eval_tool} not found. Ensure it's installed and in the current directory.")
        return
    
    try:
        # Run evaluation and capture the output
        result = subprocess.run([eval_tool, qrel_file, result_file], capture_output=True, text=True)
        print(f"\nEvaluation for {result_file} using {eval_tool}:")
        print(result.stdout)
    except Exception as e:
        print(f"Error running {eval_tool}: {e}")

# Define relevance judgments file
qrel_file = "cranqrel.trec.txt"

# Run evaluation for each model
for model, result_file in {"TF-IDF": "tfidf_results.txt", "BM25": "bm25_results.txt", "LM": "lm_results.txt"}.items():
    run_trec_eval(qrel_file, result_file)
