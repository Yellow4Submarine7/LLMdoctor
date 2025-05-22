import json
import torch
# from transformers import PreTrainedTokenizer
# from llmdoctor.core.reward_acquisition import get_behavioral_variant_logprobs, calculate_token_importance, assign_token_level_rewards

# Define a type hint for the tokenizer for clarity
# In a real setup, you'd import PreTrainedTokenizer from transformers
class PreTrainedTokenizer: # Mock for type hinting
    def encode(self, text, add_special_tokens=False, return_tensors=None): pass
    def pad_token_id(self): pass

class PreferenceDataset:
    def __init__(self, data: list[dict]):
        """
        Represents a preference dataset.

        Args:
            data (list[dict]): A list of preference items. 
                               Each item is a dictionary with keys like 'prompt', 
                               'preferred_response', 'unpreferred_response'.
                               Example: [{"prompt": "Hello", "preferred_response": "Hi there!", "unpreferred_response": "Go away."}]
        """
        self.data = data
        if not isinstance(data, list):
            raise ValueError("Dataset must be a list of dictionaries.")
        for item in data:
            if not all(k in item for k in ['prompt', 'preferred_response', 'unpreferred_response']):
                raise ValueError(f"Dataset item missing required keys: {item}")
        
        print(f"PreferenceDataset initialized with {len(data)} items.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @classmethod
    def load_from_jsonl(cls, file_path: str):
        """
        Loads a preference dataset from a JSONL file.
        Each line in the file should be a JSON object representing one preference item.
        Required keys: "prompt", "chosen" (for preferred_response), "rejected" (for unpreferred_response).
        Example line: {"prompt": "What is AI?", "chosen": "AI is...", "rejected": "I don't know."}
        """
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        # Adapt common key names like "chosen"/"rejected" or "response_j"/"response_k"
                        if 'prompt' not in item:
                            # Try to find a prompt-like key (e.g. "context", "instruction")
                            prompt_keys = ['context', 'instruction', 'query']
                            found_prompt_key = None
                            for pk in prompt_keys:
                                if pk in item:
                                    found_prompt_key = pk
                                    break
                            if not found_prompt_key:
                                print(f"Warning: Skipping item, 'prompt' key not found and no alternatives: {item}")
                                continue
                            item['prompt'] = item.pop(found_prompt_key)


                        if 'chosen' in item and 'rejected' in item:
                            item['preferred_response'] = item.pop('chosen')
                            item['unpreferred_response'] = item.pop('rejected')
                        elif 'response_j' in item and 'response_k' in item: # Another common format
                            item['preferred_response'] = item.pop('response_j')
                            item['unpreferred_response'] = item.pop('response_k')
                        elif 'preferred' in item and 'not_preferred' in item:
                            item['preferred_response'] = item.pop('preferred')
                            item['unpreferred_response'] = item.pop('not_preferred')
                        else:
                            # If specific keys are not found, check if 'preferred_response' and 'unpreferred_response' already exist
                            if not ('preferred_response' in item and 'unpreferred_response' in item):
                                print(f"Warning: Skipping item, preferred/unpreferred keys not found: {item}")
                                continue
                        
                        data.append(item)
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping line, invalid JSON: {line.strip()}")
                        continue
            print(f"Loaded {len(data)} items from {file_path}")
            return cls(data)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return cls([]) # Return empty dataset
        except Exception as e:
            print(f"Error loading dataset from {file_path}: {e}")
            return cls([])


def tokenize_preference_item(item: dict, tokenizer: PreTrainedTokenizer, max_length: int = 512):
    """
    Tokenizes a single preference item (prompt, preferred_response, unpreferred_response).

    Args:
        item (dict): A dictionary with 'prompt', 'preferred_response', 'unpreferred_response'.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        max_length (int): Maximum token length for truncation.

    Returns:
        dict: A dictionary with tokenized versions of the inputs, e.g.,
              'prompt_ids', 'preferred_response_ids', 'unpreferred_response_ids',
              and corresponding attention masks.
              Returns None if tokenization fails for critical parts.
    """
    try:
        prompt_tokens = tokenizer.encode(item['prompt'], add_special_tokens=False)
        pref_response_tokens = tokenizer.encode(item['preferred_response'], add_special_tokens=False)
        unpref_response_tokens = tokenizer.encode(item['unpreferred_response'], add_special_tokens=False)

        return {
            "prompt_text": item['prompt'],
            "preferred_response_text": item['preferred_response'],
            "unpreferred_response_text": item['unpreferred_response'],
            "prompt_ids": torch.tensor(prompt_tokens, dtype=torch.long),
            "preferred_response_ids": torch.tensor(pref_response_tokens, dtype=torch.long),
            "unpreferred_response_ids": torch.tensor(unpref_response_tokens, dtype=torch.long),
        }
    except Exception as e:
        print(f"Error tokenizing item {item.get('prompt', 'Unknown prompt')}: {e}")
        return None

def prepare_data_for_reward_acquisition(dataset: PreferenceDataset, 
                                        patient_tokenizer: PreTrainedTokenizer,
                                       ):
    """
    Prepares data for the token-level reward acquisition stage.
    This involves tokenizing prompts and responses.
    """
    processed_data = []
    for item in dataset:
        tokenized_item = tokenize_preference_item(item, patient_tokenizer)
        if tokenized_item:
            processed_data.append(tokenized_item)
            
    print(f"Prepared {len(processed_data)} items for reward acquisition.")
    return processed_data


def prepare_data_for_tfpo_training(processed_reward_data: list,
                                   doctor_tokenizer: PreTrainedTokenizer, # Usually same as patient for TFPO
                                   max_seq_length_doctor: int = 256):
    """
    Prepares data for TFPO training.
    """
    trajectories_for_subtb = []
    data_for_value_loss = [] # Placeholder for now

    for item in processed_reward_data:
        prompt_ids = item['prompt_ids']
        
        for response_type in ["preferred", "unpreferred"]:
            response_ids_key = f"{response_type}_response_ids"
            rewards_key = f"{response_type}_rewards" 

            if response_ids_key not in item or rewards_key not in item:
                continue

            response_ids = item[response_ids_key]
            token_rewards_for_response = item[rewards_key] 

            if len(response_ids) != len(token_rewards_for_response):
                print(f"Warning: Mismatch between token IDs ({len(response_ids)}) and rewards ({len(token_rewards_for_response)}) for item. Skipping this response.")
                continue
            if not response_ids.numel(): 
                continue

            current_trajectory = []
            # s_0 is the prompt. The first token y_1 is the first token of the response.
            # So, the first prefix s_0 for y_1 is just prompt_ids.
            current_prefix_ids = prompt_ids.clone() 

            for i in range(len(response_ids)):
                s_k_ids = current_prefix_ids.clone() 
                y_k_plus_1_id = response_ids[i].unsqueeze(0) 
                r_yk_plus_1 = token_rewards_for_response[i]  

                if s_k_ids.size(0) > max_seq_length_doctor:
                    s_k_ids = s_k_ids[-max_seq_length_doctor:]
                
                current_trajectory.append((s_k_ids, y_k_plus_1_id, r_yk_plus_1))
                
                current_prefix_ids = torch.cat((current_prefix_ids, y_k_plus_1_id), dim=0)
            
            if current_trajectory:
                trajectories_for_subtb.append(current_trajectory)
    
    print(f"Prepared {len(trajectories_for_subtb)} trajectories for SubTB loss.")
    print(f"Data for value loss is currently a placeholder: {len(data_for_value_loss)} items.")
    return trajectories_for_subtb, data_for_value_loss


# Example Usage
if __name__ == '__main__':
    # --- Mock Tokenizer for testing ---
    class MockTokenizerForDataUtils:
        def __init__(self, vocab_size=100, pad_token_id=0):
            self._vocab_size = vocab_size
            self._pad_token_id = pad_token_id
            print("MockTokenizerForDataUtils initialized.")

        def encode(self, text, add_special_tokens=False, return_tensors=None): 
            encoded = [min(ord(c) % self._vocab_size, self._vocab_size -1) for c in text if ord(c) < self._vocab_size]
            if not encoded and add_special_tokens: 
                 encoded = [1] 
            elif not encoded:
                 encoded = []
            return encoded 

        @property
        def pad_token_id(self):
            return self._pad_token_id

    mock_tokenizer = MockTokenizerForDataUtils(vocab_size=30000)

    # --- Test PreferenceDataset Loading ---
    print("\n--- Testing PreferenceDataset Loading ---")
    dummy_jsonl_content = [
        {"prompt": "Question 1", "chosen": "Good answer 1", "rejected": "Bad answer 1"},
        {"prompt": "Question 2", "context": "This should be ignored", "response_j": "Preferred response", "response_k": "Worse response"},
        {"instruction": "Question 3", "preferred": "Nice one", "not_preferred": "Not nice"},
        {"prompt": "Question 4", "preferred_response": "Already in right format", "unpreferred_response": "This too"},
        {"no_prompt": "Test 5", "chosen": "A", "rejected": "B"}, 
        "This is not a valid JSON line" 
    ]
    dummy_file_path = "dummy_prefs.jsonl"
    with open(dummy_file_path, 'w') as f:
        for entry in dummy_jsonl_content:
            if isinstance(entry, dict):
                f.write(json.dumps(entry) + "\n")
            else:
                f.write(entry + "\n")

    dataset = PreferenceDataset.load_from_jsonl(dummy_file_path)
    print(f"Loaded dataset with {len(dataset)} items.")
    if len(dataset) > 0:
        print("First item:", dataset[0])
    assert len(dataset) == 4, f"Expected 4 items, got {len(dataset)}"

    # --- Test Tokenization ---
    print("\n--- Testing Tokenization ---")
    if len(dataset) > 0:
        item0 = dataset[0]
        tokenized_item0 = tokenize_preference_item(item0, mock_tokenizer)
        print("Tokenized first item:", tokenized_item0)
        assert 'prompt_ids' in tokenized_item0
        if tokenized_item0['preferred_response_text']: # only assert if text is not empty
             assert tokenized_item0['preferred_response_ids'].numel() > 0
        else: # if text is empty, numel should be 0
             assert tokenized_item0['preferred_response_ids'].numel() == 0


    # --- Test Preparation for Reward Acquisition ---
    print("\n--- Testing Preparation for Reward Acquisition ---")
    prepared_for_reward = prepare_data_for_reward_acquisition(dataset, mock_tokenizer)
    if prepared_for_reward:
        print("First item prepared for reward acquisition:", prepared_for_reward[0])
        assert 'preferred_response_ids' in prepared_for_reward[0]
        assert 'unpreferred_response_text' in prepared_for_reward[0]

    # --- Test Preparation for TFPO Training ---
    print("\n--- Testing Preparation for TFPO Training ---")
    dummy_reward_data = []
    for item_data in prepared_for_reward:
        if item_data['preferred_response_ids'].numel() > 0: # Only add rewards if there are tokens
            item_data['preferred_rewards'] = torch.rand(item_data['preferred_response_ids'].size(0))
        else:
            item_data['preferred_rewards'] = torch.tensor([])

        if item_data['unpreferred_response_ids'].numel() > 0:
            item_data['unpreferred_rewards'] = torch.rand(item_data['unpreferred_response_ids'].size(0)) * -1 
        else:
            item_data['unpreferred_rewards'] = torch.tensor([])
            
        dummy_reward_data.append(item_data)
    
    if dummy_reward_data:
        trajectories, value_loss_data_placeholder = prepare_data_for_tfpo_training(dummy_reward_data, mock_tokenizer)
        print(f"Number of trajectories for SubTB: {len(trajectories)}")
        if trajectories:
            print("Example trajectory (first one, first step):", trajectories[0][0] if trajectories[0] else "Empty trajectory")
            if trajectories[0]:
                s_k, y_k1, r_yk1 = trajectories[0][0]
                print(f"  s_k shape: {s_k.shape}, y_k+1 shape: {y_k1.shape}, r_yk+1: {r_yk1.item()}")
                assert s_k.ndim == 1
                assert y_k1.ndim == 1 and y_k1.numel() == 1 
                assert isinstance(r_yk1, torch.Tensor)
        assert len(value_loss_data_placeholder) == 0 

    print("\n--- Data Utils Test Complete ---")
    import os
    os.remove(dummy_file_path)
    print(f"Removed dummy file: {dummy_file_path}")
