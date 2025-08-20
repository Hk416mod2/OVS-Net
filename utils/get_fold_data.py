import json
import os

_fold_data_cache = {}

def load_fold_data(data_root, fold):
    cache_key = f"{data_root}_{fold}"
    
    if cache_key not in _fold_data_cache:
        json_path = os.path.join(data_root, 'train_val_fold.json')
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                all_folds_data = json.load(f)
            
            fold_key = f"fold_{fold}"
            if fold_key in all_folds_data["folds"]:
                fold_data = all_folds_data["folds"][fold_key]
                _fold_data_cache[cache_key] = {
                    'train': fold_data['train'],
                    'val': fold_data['val']
                }
            else:
                raise ValueError(f"Fold {fold} not found in {json_path}")
        else:
            raise FileNotFoundError(f"JSON file not found at {json_path}")
    
    return _fold_data_cache[cache_key]
