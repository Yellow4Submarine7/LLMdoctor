import yaml
import os

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'default_config.yaml')

def load_config(config_path: str = None) -> dict:
    """
    Loads a configuration from a YAML file.

    Args:
        config_path (str, optional): Path to the YAML configuration file. 
                                     If None, loads the 'default_config.yaml' from the same directory.

    Returns:
        dict: A dictionary containing the configuration parameters.
              Returns an empty dict if loading fails.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
        print(f"No config path provided, using default: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        # Attempt to create a default one if it's the default path that's missing
        if config_path == DEFAULT_CONFIG_PATH:
            print("Attempting to create a minimal default_config.yaml as it's missing.")
            try:
                minimal_config = {
                    'model_paths': {
                        'patient_model_name': 'Qwen/Qwen1.5-72B-Chat',
                        'doctor_model_name': 'Qwen/Qwen1.5-4B-Chat'
                    },
                    'data_paths': {
                        'preference_dataset_jsonl': 'path/to/your/preference_data.jsonl'
                    },
                    'reward_acquisition_params': {
                        'epsilon': 1e-6,
                        'tau': 1.0,
                        'sparsity_threshold': 0.1
                    },
                    'tfpo_params': {
                        'lambda_val_loss': 0.1,
                        'gamma_val_loss': 0.1,
                        'learning_rate': 1e-5,
                        'epochs': 3,
                        'batch_size': 4,
                        'max_seq_length_doctor': 256
                    },
                    'online_alignment_params': {
                        'alpha': 1.0,
                        'beta_values': [0.5],
                        'max_generation_length': 50,
                        'temperature': 0.7,
                        'top_k': 50
                    },
                    'training_settings': {
                        'device': 'auto', # 'auto', 'cuda', 'cpu'
                        'use_quantization_patient': True,
                        'use_quantization_doctor': False,
                        'seed': 42
                    }
                }
                with open(DEFAULT_CONFIG_PATH, 'w') as f:
                    yaml.dump(minimal_config, f, sort_keys=False)
                print(f"Created minimal default_config.yaml at {DEFAULT_CONFIG_PATH}. Please review and update paths.")
                return minimal_config
            except Exception as e_create:
                print(f"Could not create minimal default config: {e_create}")
                return {}
        else:
            return {} # Only create default if default path was used.

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from {config_path}")
        return config if config else {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        return {}
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return {}

if __name__ == '__main__':
    print("--- Testing Config Loader ---")

    # Test loading default config (it might be created if not present)
    print("\n1. Testing loading default config:")
    config_default = load_config() # Uses DEFAULT_CONFIG_PATH
    if config_default:
        print("Default config loaded/created successfully:")
        # print(json.dumps(config_default, indent=2)) # For pretty print
        assert 'model_paths' in config_default
        assert 'data_paths' in config_default
    else:
        print("Failed to load or create default config.")

    # Create a dummy temporary config file for further testing
    dummy_cfg_path = "temp_test_config.yaml"
    dummy_content = {
        'model_paths': {'patient_model_name': 'test_patient', 'doctor_model_name': 'test_doctor'},
        'tfpo_params': {'learning_rate': 0.001}
    }
    with open(dummy_cfg_path, 'w') as f:
        yaml.dump(dummy_content, f)

    print(f"\n2. Testing loading specific config file: {dummy_cfg_path}")
    config_specific = load_config(dummy_cfg_path)
    if config_specific:
        print("Specific config loaded successfully:")
        # print(json.dumps(config_specific, indent=2))
        assert config_specific['model_paths']['patient_model_name'] == 'test_patient'
        assert config_specific['tfpo_params']['learning_rate'] == 0.001
    else:
        print(f"Failed to load specific config from {dummy_cfg_path}")

    # Test loading non-existent specific config
    print("\n3. Testing loading non-existent specific config:")
    config_non_existent = load_config("non_existent_config.yaml")
    if not config_non_existent:
        print("Correctly returned empty dict for non-existent specific config.")
        assert config_non_existent == {}
    else:
        print("Error: Should have returned empty dict for non-existent specific file.")
        
    # Clean up dummy config file
    if os.path.exists(dummy_cfg_path):
        os.remove(dummy_cfg_path)
        print(f"\nRemoved temporary config file: {dummy_cfg_path}")
        
    # Verify default config file (default_config.yaml) was created if it didn't exist
    # This check should ideally be after the first load_config() call if default was expected to be created.
    # If default_config.yaml is created by this script's execution, this check is valid here.
    if os.path.exists(DEFAULT_CONFIG_PATH):
        print(f"\nDefault config file exists at: {DEFAULT_CONFIG_PATH}")
    else:
        print(f"\nWarning: Default config file was NOT created at: {DEFAULT_CONFIG_PATH} (This might be an issue if it was expected).")


    print("\n--- Config Loader Test Complete ---")
