import os
import sys
import torch
import json # For handling data if we save/load rewards as JSON
import random

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llmdoctor.configs.config_loader import load_config
from llmdoctor.data.utils import prepare_data_for_tfpo_training # Using this to generate trajectories
from llmdoctor.models.model_loader import load_model_and_tokenizer # For actual doctor model
from llmdoctor.core.tfpo_tuner import DoctorModel, ValueFunction, TFPOTrainer 
# We'll use the placeholder DoctorModel and ValueFunction from tfpo_tuner.py for this example,
# as they don't require actual pre-trained weights for the script to run.

# Mock Tokenizer for data preparation if not loading a real doctor model
class MockDoctorTokenizer:
    def __init__(self, vocab_size=50257): # Default to GPT2-like vocab size
        self.vocab_size = vocab_size
        self.pad_token_id = 0 # Example
        print("MockDoctorTokenizer initialized for TFPO training example.")

    def encode(self, text, add_special_tokens=False, return_tensors=None, truncation=True, max_length=None):
        tokens = [min(ord(c) % self.vocab_size, self.vocab_size-1) for c in text]
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
        if return_tensors == "pt": # Not used by prepare_data_for_tfpo_training directly but good for consistency
            return torch.tensor([tokens], dtype=torch.long)
        return tokens # prepare_data_for_tfpo_training expects list of ints from encode for its internal logic

    def decode(self, token_ids, skip_special_tokens=True): # Not strictly needed by this script
        return f"decoded_{token_ids}"


def create_dummy_processed_reward_data(num_items=5, max_prompt_len=10, max_resp_len=15, vocab_size=100):
    """
    Creates dummy data that mimics the output of the reward acquisition stage.
    Each item will have: prompt_ids, preferred_response_ids, preferred_rewards,
                         unpreferred_response_ids, unpreferred_rewards.
    """
    dummy_data = []
    for i in range(num_items):
        p_len = torch.randint(1, max_prompt_len + 1, (1,)).item()
        pref_r_len = torch.randint(1, max_resp_len + 1, (1,)).item()
        unpref_r_len = torch.randint(1, max_resp_len + 1, (1,)).item()

        item = {
            'prompt_text': f'Dummy prompt {i}', # For reference
            'prompt_ids': torch.randint(0, vocab_size, (p_len,)),
            'preferred_response_text': f'Dummy preferred response {i}',
            'preferred_response_ids': torch.randint(0, vocab_size, (pref_r_len,)),
            'preferred_rewards': torch.rand(pref_r_len) * 2 - 1, # Rewards between -1 and 1
            'unpreferred_response_text': f'Dummy unpreferred response {i}',
            'unpreferred_response_ids': torch.randint(0, vocab_size, (unpref_r_len,)),
            'unpreferred_rewards': torch.rand(unpref_r_len) * -2 + 1 # Biased towards negative
        }
        dummy_data.append(item)
    print(f"Created {len(dummy_data)} dummy items for TFPO training input.")
    return dummy_data

def main():
    print("--- Running LLMdoctor: Stage 2 - TFPO-based Fine-grained Preference Tuning ---")

    # 1. Load Configuration
    print("\n1. Loading configuration...")
    config_path = os.path.join(project_root, 'llmdoctor', 'configs', 'default_config.yaml')
    if not os.path.exists(config_path):
        print(f"Warning: Default config not found at {config_path}. Using load_config's internal default.")
        config = load_config()
    else:
        config = load_config(config_path)
    
    if not config:
        print("Configuration could not be loaded. Exiting.")
        return

    tfpo_params = config.get('tfpo_params', {})
    training_settings = config.get('training_settings', {})
    model_paths_config = config.get('model_paths', {}) # For actual doctor model if not using placeholder

    device = torch.device("cuda" if torch.cuda.is_available() and training_settings.get('device') != 'cpu' else "cpu")
    if training_settings.get('device') == 'auto':
        print(f"Device set to 'auto'. Using: {device}")
    else:
        print(f"Device set to: {device} (based on config and availability)")

    torch.manual_seed(training_settings.get('seed', 42))
    random.seed(training_settings.get('seed', 42)) # For shuffling trajectories
    print(f"Set random seed to: {training_settings.get('seed', 42)}")

    # 2. Load Processed Data (from Reward Acquisition Stage)
    print("\n2. Loading data with token-level rewards (using dummy data for this script)...")
    # In a real pipeline, this data would be loaded from a file produced by run_reward_acquisition.py
    # e.g., all_items_with_rewards = torch.load("all_items_with_rewards.pt")
    # For this example, we'll generate dummy data.
    
    # Using a small vocab_size for dummy data consistent with placeholder DoctorModel
    dummy_vocab_size_for_data = tfpo_params.get('doctor_model_vocab_size_for_dummy', 1000) 
    processed_reward_data = create_dummy_processed_reward_data(vocab_size=dummy_vocab_size_for_data)

    # 3. Initialize Doctor Model, Value Function, and Tokenizer
    print("\n3. Initializing Doctor Model, Value Function (using placeholders from tfpo_tuner.py)...")
    
    doctor_model_vocab_size = dummy_vocab_size_for_data # Ensure consistency
    doctor_model = DoctorModel(vocab_size=doctor_model_vocab_size).to(device)
    
    value_function_input_size = tfpo_params.get('value_function_input_size_for_dummy', 10)
    value_function = ValueFunction(input_size=value_function_input_size).to(device)
    
    doctor_tokenizer = MockDoctorTokenizer(vocab_size=doctor_model_vocab_size)
    
    print("Placeholder DoctorModel, ValueFunction, and MockDoctorTokenizer initialized.")

    # 4. Prepare Data for TFPO Trainer
    print("\n4. Preparing data for TFPO trainer...")
    trajectories_for_subtb, data_for_value_loss = prepare_data_for_tfpo_training(
        processed_reward_data,
        doctor_tokenizer, 
        max_seq_length_doctor=tfpo_params.get('max_seq_length_doctor', 256)
    )

    if not trajectories_for_subtb:
        print("No trajectories created for SubTB loss. This might be due to empty responses or reward data issues.")
        return

    if tfpo_params.get('lambda_val_loss', 0.1) > 0 and not data_for_value_loss:
        print("`data_for_value_loss` is empty. Creating dummy data for value discrimination loss demonstration.")
        num_value_loss_pairs = 10 
        data_for_value_loss = [
            (torch.randn(1, value_function_input_size).to(device), torch.randn(1, value_function_input_size).to(device)) 
            for _ in range(num_value_loss_pairs)
        ]
        print(f"Created {len(data_for_value_loss)} dummy pairs for value discrimination loss.")


    # 5. Initialize TFPO Trainer
    print("\n5. Initializing TFPO Trainer...")
    tfpo_trainer = TFPOTrainer(
        doctor_model,
        value_function,
        lambda_val=tfpo_params.get('lambda_val_loss', 0.1),
        learning_rate=tfpo_params.get('learning_rate', 1e-5),
        device=device
    )
    print("TFPOTrainer initialized.")

    # 6. Run Training Loop
    print("\n6. Starting TFPO training loop...")
    epochs = tfpo_params.get('epochs', 1) 
    batch_size = tfpo_params.get('batch_size', 2)

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        epoch_total_loss = 0
        epoch_subtb_loss = 0
        epoch_val_disc_loss = 0
        num_batches = 0

        random.shuffle(trajectories_for_subtb) 

        for i in range(0, len(trajectories_for_subtb), batch_size):
            batch_trajectories = trajectories_for_subtb[i : i + batch_size]
            
            current_value_loss_data = []
            if data_for_value_loss: # Sample from value_loss_data if available
                current_value_loss_data = random.sample(data_for_value_loss, min(len(data_for_value_loss), batch_size))


            total_loss, subtb_loss, val_disc_loss = tfpo_trainer.train_step(
                batch_trajectories,
                current_value_loss_data, 
                gamma_val_loss=tfpo_params.get('gamma_val_loss', 0.1)
            )
            
            print(f"  Batch {i//batch_size + 1}/{len(trajectories_for_subtb)//batch_size + 1}: "
                  f"Total Loss: {total_loss:.4f}, SubTB Loss: {subtb_loss:.4f}, ValDisc Loss: {val_disc_loss:.4f}")
            
            epoch_total_loss += total_loss
            epoch_subtb_loss += subtb_loss
            epoch_val_disc_loss += val_disc_loss
            num_batches += 1
        
        avg_epoch_total_loss = epoch_total_loss / num_batches if num_batches > 0 else 0
        avg_epoch_subtb_loss = epoch_subtb_loss / num_batches if num_batches > 0 else 0
        avg_epoch_val_disc_loss = epoch_val_disc_loss / num_batches if num_batches > 0 else 0

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Total Loss: {avg_epoch_total_loss:.4f}")
        print(f"  Average SubTB Loss: {avg_epoch_subtb_loss:.4f}")
        print(f"  Average Value Discrimination Loss: {avg_epoch_val_disc_loss:.4f}")

    print("\n--- TFPO Training Stage Complete ---")
    print("Note: This script used placeholder models and dummy data. Losses are illustrative.")

if __name__ == "__main__":
    main()
