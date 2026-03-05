import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Flush after every print to ensure output appears before a segfault
import functools
print = functools.partial(print, flush=True)

print("=== Testing project module imports one by one ===")

print("1. importing configs.config ...", end=" ")
from configs.config import Config, set_seed
print("OK")

print("2. importing utils.helpers ...", end=" ")
from utils.helpers import clear_memory
print("OK")

print("3. importing data.data_loader ...", end=" ")
from data.data_loader import prepare_datasets
print("OK")

print("4. importing utils.visualizer ...", end=" ")
from utils.visualizer import plot_latent_pca, plot_loss_curve
print("OK")

print("5. importing models.coconut ...", end=" ")
from models.coconut import initialize_qwen_model
print("OK")

print("6. importing core.trainer ...", end=" ")
from core.trainer import train_phase1
print("OK")

print("7. importing core.extractor ...", end=" ")
from core.extractor import extract_truth_vector
print("OK")

print("8. importing core.evaluator ...", end=" ")
from core.evaluator import analyze_confidence, evaluate_with_iti, print_sample_outputs
print("OK")

print("\nAll project imports passed. No segfault.")
