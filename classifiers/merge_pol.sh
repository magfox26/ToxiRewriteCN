PROJECT_ROOT=$(python -c "from utils.path_utils import get_project_root; print(get_project_root())")

swift export \
    --adapters "$PROJECT_ROOT/classifiers/output/pol_ratio121/" \  # Replace with path to your LoRA adapters
    --merge_lora true \  
    --output_dir "$PROJECT_ROOT/classifiers/output/qwen3-32b-style-polarity/"  # Replace with path to save merged model
