import os
import sys
import torch

# Add project root to Python path to allow direct execution of examples
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llmdoctor.configs.config_loader import load_config
from llmdoctor.data.utils import PreferenceDataset, prepare_data_for_reward_acquisition
from llmdoctor.models.model_loader import load_model_and_tokenizer
from llmdoctor.core.reward_acquisition import get_behavioral_variant_logprobs, calculate_token_importance, assign_token_level_rewards

# Mock models and tokenizers for now, as actual model loading is heavy
# In a real run, these would be loaded by load_model_and_tokenizer
class MockPatientModel:
    def __init__(self, device='cpu'):
        self.device = device
        print(f"MockPatientModel initialized on {device}. This is a placeholder.")
    
    # The get_behavioral_variant_logprobs function expects a model and tokenizer,
    # but currently it has its own internal simulation.
    # If get_behavioral_variant_logprobs were to use this model directly, it would need methods like:
    # def __call__(self, input_ids, attention_mask=None): # etc.
    # For now, its mere presence is for API consistency if we were to make get_behavioral_variant_logprobs call it.
    pass

class MockTokenizer:
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        self.pad_token_id = 0 # Example
        print("MockTokenizer initialized. This is a placeholder.")

    def encode(self, text, add_special_tokens=False, return_tensors=None, truncation=True, max_length=None):
        # Simplified deterministic encoding for mock
        tokens = [min(ord(c) % self.vocab_size, self.vocab_size-1) for c in text]
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
        if return_tensors == "pt":
            return torch.tensor([tokens], dtype=torch.long)
        return tokens
        
    def batch_decode(self, sequences, skip_special_tokens=True):
        return [f"decoded_{seq}" for seq in sequences]

    def decode(self, token_ids, skip_special_tokens=True):
        return f"decoded_{token_ids}"

def main():
    print("--- Running LLMdoctor: Stage 1 - Token-Level Reward Acquisition ---")

    # 1. Load Configuration
    print("\n1. Loading configuration...")
    # Assuming default_config.yaml exists in llmdoctor/configs/
    # You might need to adjust the path if running from a different directory
    # or ensure default_config.yaml has a valid path for preference_dataset_jsonl
    config_path = os.path.join(project_root, 'llmdoctor', 'configs', 'default_config.yaml')
    if not os.path.exists(config_path):
        print(f"Warning: Default config not found at {config_path}. Using load_config's internal default.")
        config = load_config() # Relies on load_config to create a minimal default if necessary
    else:
        config = load_config(config_path)

    if not config:
        print("Configuration could not be loaded. Exiting.")
        return

    # Extract relevant parameters
    data_params = config.get('data_paths', {})
    reward_acq_params = config.get('reward_acquisition_params', {})
    training_settings = config.get('training_settings', {})
    # model_paths = config.get('model_paths', {}) # For actual model loading

    device = torch.device("cuda" if torch.cuda.is_available() and training_settings.get('device') != 'cpu' else "cpu")
    if training_settings.get('device') == 'auto':
        print(f"Device set to 'auto'. Using: {device}")
    else:
        print(f"Device set to: {device} (based on config and availability)")


    # 2. Load Dataset
    print("\n2. Loading preference dataset...")
    preference_file_path = data_params.get('preference_dataset_jsonl', 'dummy_prefs.jsonl')
    
    # Create a dummy dataset if the configured one doesn't exist, for demonstration
    if not os.path.exists(preference_file_path) or preference_file_path == 'path/to/your/preference_data.jsonl':
        print(f"Warning: Preference data file '{preference_file_path}' not found or is a placeholder.")
        print("Creating a dummy dataset for demonstration: dummy_reward_acq_prefs.jsonl")
        preference_file_path = "dummy_reward_acq_prefs.jsonl"
        dummy_jsonl_content = [
            {"prompt": "Explain gravity.", "chosen": "Gravity is a force that attracts objects.", "rejected": "Gravity is a type of food."},
            {"prompt": "What is Python?", "chosen": "Python is a programming language.", "rejected": "Python is a snake."}
        ]
        with open(preference_file_path, 'w') as f:
            for entry in dummy_jsonl_content:
                f.write(json.dumps(entry) + "\n")
        # Update config to reflect the dummy path for this run
        data_params['preference_dataset_jsonl'] = preference_file_path


    dataset = PreferenceDataset.load_from_jsonl(preference_file_path)
    if not dataset: # PreferenceDataset.load_from_jsonl returns cls([]) on error, so check len
        print(f"Failed to load dataset from {preference_file_path} or dataset is empty. Exiting.")
        if os.path.exists(preference_file_path) and preference_file_path == "dummy_reward_acq_prefs.jsonl":
             os.remove(preference_file_path) # Clean up dummy if it was made but failed to load
        return
    if len(dataset) == 0:
        print(f"Loaded dataset from {preference_file_path}, but it's empty. Exiting.")
        if os.path.exists(preference_file_path) and preference_file_path == "dummy_reward_acq_prefs.jsonl":
             os.remove(preference_file_path)
        return

    print(f"Loaded {len(dataset)} items from {preference_file_path}")

    # 3. Load Patient Model and Tokenizer (Using Mocks for this example script)
    print("\n3. Loading patient model and tokenizer (using Mocks for this script)...")
    # In a real scenario, you would use:
    # patient_model_name = model_paths.get('patient_model_name', 'Qwen/Qwen1.5-72B-Chat')
    # patient_model, patient_tokenizer = load_model_and_tokenizer(
    #     patient_model_name,
    #     use_quantization=training_settings.get('use_quantization_patient', True),
    #     low_cpu_mem_usage=True
    # )
    # if patient_model is None or patient_tokenizer is None:
    #     print("Failed to load patient model or tokenizer. Exiting.")
    #     return
    # patient_model.to(device)
    
    # Using mocks to avoid heavy model loading in this example script
    patient_model = MockPatientModel(device=device) # Placeholder
    patient_tokenizer = MockTokenizer() # Placeholder
    
    print("Mock patient model and tokenizer loaded.")

    # 4. Prepare Data for Reward Acquisition
    # This step primarily tokenizes. Actual reward calculation needs the model.
    print("\n4. Preparing data for reward acquisition...")
    # `prepare_data_for_reward_acquisition` is mostly for tokenization,
    # the core reward logic is in the loop below.
    tokenized_data = prepare_data_for_reward_acquisition(dataset, patient_tokenizer)
    if not tokenized_data:
        print("No data prepared for reward acquisition. Exiting.")
        if os.path.exists(preference_file_path) and preference_file_path == "dummy_reward_acq_prefs.jsonl":
             os.remove(preference_file_path)
        return

    # 5. Process Data to Acquire Token-Level Rewards
    print("\n5. Acquiring token-level rewards...")
    all_items_with_rewards = []

    for i, item_data in enumerate(tokenized_data):
        print(f"\nProcessing item {i+1}/{len(tokenized_data)}: '{item_data['prompt_text'][:50]}...'")
        prompt_text = item_data['prompt_text']
        
        # --- Preferred Response ---
        pref_response_text = item_data['preferred_response_text']
        print(f"  Preferred response: '{pref_response_text[:50]}...'")
        
        # Simulate getting log_probs (currently mock implementation inside the function)
        # The `patient_model` and `patient_tokenizer` are passed for API consistency,
        # but `get_behavioral_variant_logprobs` in `reward_acquisition.py` currently simulates its own.
        logprobs_pos_pref = get_behavioral_variant_logprobs(prompt_text, pref_response_text, "positive", patient_model, patient_tokenizer)
        logprobs_neg_pref = get_behavioral_variant_logprobs(prompt_text, pref_response_text, "negative", patient_model, patient_tokenizer)

        if logprobs_pos_pref.numel() > 0 and logprobs_neg_pref.numel() > 0:
            importance_pref = calculate_token_importance(
                logprobs_pos_pref.to(device), 
                logprobs_neg_pref.to(device),
                epsilon=reward_acq_params.get('epsilon', 1e-6),
                tau=reward_acq_params.get('tau', 1.0)
            )
            rewards_pref = assign_token_level_rewards(
                importance_pref, 
                preference_sign=1, 
                sparsity_threshold=reward_acq_params.get('sparsity_threshold', 0.1)
            )
            item_data['preferred_rewards'] = rewards_pref.cpu() # Store rewards
            print(f"    Calculated rewards for preferred response (sum: {rewards_pref.sum():.2f}, count: {len(rewards_pref)})")
        else:
            item_data['preferred_rewards'] = torch.tensor([])
            print("    Could not calculate rewards for preferred response (empty tokens/logprobs).")

        # --- Unpreferred Response ---
        unpref_response_text = item_data['unpreferred_response_text']
        print(f"  Unpreferred response: '{unpref_response_text[:50]}...'")
        
        logprobs_pos_unpref = get_behavioral_variant_logprobs(prompt_text, unpref_response_text, "positive", patient_model, patient_tokenizer)
        logprobs_neg_unpref = get_behavioral_variant_logprobs(prompt_text, unpref_response_text, "negative", patient_model, patient_tokenizer)

        if logprobs_pos_unpref.numel() > 0 and logprobs_neg_unpref.numel() > 0:
            importance_unpref = calculate_token_importance(
                logprobs_pos_unpref.to(device), 
                logprobs_neg_unpref.to(device),
                epsilon=reward_acq_params.get('epsilon', 1e-6),
                tau=reward_acq_params.get('tau', 1.0)
            )
            rewards_unpref = assign_token_level_rewards(
                importance_unpref, 
                preference_sign=-1, 
                sparsity_threshold=reward_acq_params.get('sparsity_threshold', 0.1)
            )
            item_data['unpreferred_rewards'] = rewards_unpref.cpu() # Store rewards
            print(f"    Calculated rewards for unpreferred response (sum: {rewards_unpref.sum():.2f}, count: {len(rewards_unpref)})")

        else:
            item_data['unpreferred_rewards'] = torch.tensor([])
            print("    Could not calculate rewards for unpreferred response (empty tokens/logprobs).")
            
        all_items_with_rewards.append(item_data)

    print(f"\n--- Reward Acquisition Stage Complete ---")
    print(f"Processed {len(all_items_with_rewards)} items and attached reward information.")
    if all_items_with_rewards:
        print("\nExample of first processed item with rewards:")
        first_item_example = all_items_with_rewards[0]
        print(f"  Prompt: {first_item_example['prompt_text'][:50]}...")
        print(f"  Preferred Response: {first_item_example['preferred_response_text'][:50]}...")
        print(f"  Preferred Rewards (sum): {first_item_example['preferred_rewards'].sum().item() if first_item_example['preferred_rewards'].numel() > 0 else 'N/A'}")
        print(f"  Unpreferred Response: {first_item_example['unpreferred_response_text'][:50]}...")
        print(f"  Unpreferred Rewards (sum): {first_item_example['unpreferred_rewards'].sum().item() if first_item_example['unpreferred_rewards'].numel() > 0 else 'N/A'}")

    output_file = "processed_data_with_rewards.json" 
    # For saving (example, not run by default to keep example simple):
    # serializable_data = []
    # for item in all_items_with_rewards:
    #     new_item = {}
    #     for k, v in item.items():
    #         if isinstance(v, torch.Tensor): new_item[k] = v.tolist()
    #         else: new_item[k] = v
    #     serializable_data.append(new_item)
    # with open(output_file, 'w') as f:
    #    json.dump(serializable_data, f, indent=2)
    # print(f"\nSerialized data with rewards (example) would be saved to {output_file}")


    # Clean up dummy dataset file if created
    if data_params.get('preference_dataset_jsonl') == "dummy_reward_acq_prefs.jsonl" and os.path.exists("dummy_reward_acq_prefs.jsonl"):
        os.remove("dummy_reward_acq_prefs.jsonl")
        print("\nCleaned up dummy_reward_acq_prefs.jsonl")

if __name__ == "__main__":
    # Need to import json for the dummy data creation part
    import json # This import is correctly placed inside the guard for script execution
    main()
