# LLMdoctor: Test-Time Alignment for Large Language Models

LLMdoctor is a Python framework implementing the three-stage methodology described in the paper ("LLMdoctor: A Three-Stage Framework for Efficient and Effective Test-Time Alignment"). This framework aims to align large language models (LLMs) with human preferences at test-time, reducing the need for costly retraining.

**Note:** This codebase provides the structural implementation of the LLMdoctor framework, including core algorithms with placeholder components and example scripts to demonstrate the workflow. Key interactions with actual Large Language Models (patient and doctor models) are currently implemented as placeholders or with mock objects in the example scripts. Making this framework fully operational for research or production would require replacing these placeholders with real model loading, inference, and potentially complex prompting strategies.

## Framework Overview

The LLMdoctor framework consists of three main stages:

1.  **Token-Level Reward Acquisition**:
    *   This stage aims to extract fine-grained token-level reward signals from a large pre-trained "patient" model.
    *   It involves creating behavioral variants of the patient model (e.g., "positive face" and "negative face") through prompting.
    *   Token importance is measured by comparing log-probabilities of tokens under these contrasting behaviors.
    *   These importance scores are combined with human preference labels from a dataset (e.g., $y_+, y_-$ pairs) to assign directional token-level rewards ($r_t$).
    *   Implemented in: `llmdoctor/core/reward_acquisition.py`
    *   Example: `llmdoctor/examples/run_reward_acquisition.py` (uses mock model logic)

2.  **TFPO-based Fine-grained Preference Tuning**:
    *   The token-level rewards ($r_t$) obtained from Stage 1 are used to train a smaller "doctor" model ($\hat{\pi}_	heta$).
    *   This is achieved through Token-level Flow-guided Preference Optimization (TFPO), which incorporates a value function ($V_\phi$) and applies flow conservation principles (Subtrajectory Balance - SubTB) to token sequences.
    *   The SubTB loss ($\mathcal{L}_{	ext{SubTB}}$) trains the doctor model and value function to satisfy flow consistency across all token subsequences.
    *   An additional Value Discrimination Loss ($\mathcal{L}_{	ext{value}}$) helps the value function distinguish between preferred and less-preferred next tokens.
    *   Implemented in: `llmdoctor/core/tfpo_tuner.py` (DoctorModel and ValueFunction are nn.Module placeholders)
    *   Example: `llmdoctor/examples/run_tfpo_training.py` (uses placeholder models and dummy data)

3.  **Online Alignment with Flow-Guided Reward Model**:
    *   The trained doctor model acts as a flow-guided reward model during inference.
    *   It provides token-level preference signals ($\log \pi_r(y_{t+1}|s_t)$) that guide the generation of the larger, frozen patient model.
    *   The patient model's original log-probabilities ($\pi_{	ext{base}}$) are combined with the doctor model's signals:
        $\pi_{	ext{decode}}(y_{t+1}\mid s_t) \;\propto\; igl[\pi_{	ext{base}}(y_{t+1}\mid s_t)igr]^{\,lpha} \;\cdot\; \prod_i igl[\pi_{r}^{(i)}(y_{t+1}\mid s_t)igr]^{\,eta_i}$.
    *   This allows for flexible, multi-dimensional alignment at inference time without retraining the patient model.
    *   Implemented in: `llmdoctor/core/online_alignment.py`
    *   Example: `llmdoctor/examples/run_online_alignment.py` (uses mock models)

## Current Status & Limitations

*   **Framework Structure**: The core directory structure, module separation, configuration loading, and data utility classes are implemented.
*   **Algorithm Placeholders**:
    *   The core mathematical formulations for reward calculation, SubTB loss, and value discrimination loss are present in `core/reward_acquisition.py` and `core/tfpo_tuner.py`.
    *   However, critical components like the `DoctorModel` and `ValueFunction` in `tfpo_tuner.py` are `torch.nn.Module` placeholders with dummy layers. They need to be replaced with actual transformer model architectures (e.g., loading a pre-trained Qwen-3B for the DoctorModel).
    *   Interactions with LLMs (getting log-probabilities, hidden states) in `core/reward_acquisition.py` and `core/online_alignment.py` are simulated or use mock objects.
*   **Data Handling**:
    *   Loading preference datasets (JSONL) is supported.
    *   Preparation of data for TFPO training (`prepare_data_for_tfpo_training`) correctly structures trajectories for the SubTB loss but has a placeholder for generating data needed for the value discrimination loss.
    *   The definition and calculation of $Q(s_t)$ (prefix scores) and state representations for $V_\phi(s_t)$ in `tfpo_tuner.py` are currently based on simplified/dummy logic and require careful implementation based on the paper's details.
*   **Configuration**: A configuration system using YAML files is in place (`llmdoctor/configs/`).
*   **Example Scripts**: Example scripts in `llmdoctor/examples/` demonstrate the workflow for each stage but rely on mock models and dummy data. They are useful for understanding the code structure and intended data flow.
*   **Missing Key Implementations for Full Functionality**:
    *   **Real LLM Integration**: Replacing all mock model interactions with actual Hugging Face model loading and inference for both patient and doctor models.
    *   **Behavioral Variant Prompting**: Designing and implementing the specific prompting strategies to elicit "positive face" and "negative face" behaviors from the patient model (Stage 1). This is model-dependent and crucial for meaningful reward signals.
    *   **State Representations for Value Function**: Implementing the method to extract or compute meaningful state representations (e.g., hidden states from the DoctorModel) for the ValueFunction $V_\phi$ in TFPO.
    *   **Vocabulary Alignment**: The `OnlineAlignment` module notes the challenge of vocabulary mismatch between potentially different patient and doctor models. The current placeholder solution (padding/truncation) is naive and needs a robust strategy if models don't share vocabularies.
    *   **Value Discrimination Data**: Populating `data_for_value_loss` in `data/utils.py` with meaningful pairs for training the value function.

## Project Structure

```
llmdoctor/
├── configs/                 # Configuration files (YAML) and loader
│   ├── default_config.yaml
│   └── config_loader.py
├── core/                    # Core logic for the three stages
│   ├── reward_acquisition.py
│   ├── tfpo_tuner.py
│   ├── online_alignment.py
│   └── __init__.py
├── data/                    # Data loading and preprocessing utilities
│   ├── utils.py
│   └── __init__.py
├── examples/                # Example scripts to run each stage
│   ├── run_reward_acquisition.py
│   ├── run_tfpo_training.py
│   ├── run_online_alignment.py
│   └── __init__.py
├── models/                  # Model loading utilities
│   ├── model_loader.py
│   └── __init__.py
├── __init__.py
└── README.md                # This file
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd llmdoctor
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scriptsctivate
    ```

3.  **Install dependencies:**
    The core dependencies are PyTorch and Transformers. Other libraries like `pyyaml` are also needed.
    ```bash
    pip install torch torchvision torchaudio
    pip install transformers
    pip install pyyaml
    pip install accelerate bitsandbytes # For quantization and faster loading of large models
    ```
    Ensure you install versions compatible with your hardware (e.g., CUDA version for PyTorch if using GPU).

## Configuration

Before running the example scripts, review and customize the configuration:

1.  **Copy and rename the default config (optional but recommended):**
    ```bash
    cp llmdoctor/configs/default_config.yaml llmdoctor/configs/my_config.yaml
    ```
2.  **Edit `my_config.yaml` (or `default_config.yaml`):**
    *   **`model_paths`**:
        *   `patient_model_name`: Set to the Hugging Face model name or local path for your large patient LLM (e.g., "Qwen/Qwen1.5-72B-Chat").
        *   `doctor_model_name`: Set to the Hugging Face model name or local path for your smaller doctor LLM (e.g., "Qwen/Qwen1.5-4B-Chat").
        *   For initial testing with example scripts (which use mocks), these paths are not strictly loaded by the examples but are good to set for future use.
    *   **`data_paths`**:
        *   `preference_dataset_jsonl`: **Crucially, update this** to the actual path of your preference dataset in JSONL format. The example scripts include logic to create a dummy JSONL file if this path is not found, allowing them to run out-of-the-box for a quick demonstration.
    *   Review other parameters for reward acquisition, TFPO training, online alignment, and training settings. The defaults are provided as a starting point.

## Running the Example Scripts

The example scripts in `llmdoctor/examples/` demonstrate the workflow of each stage. They currently use mock objects for actual LLM interactions and dummy data where necessary, so they can run without requiring large model downloads or extensive datasets.

1.  **Run Token-Level Reward Acquisition:**
    ```bash
    python llmdoctor/examples/run_reward_acquisition.py
    ```
    This script will:
    *   Load configuration.
    *   Load a (dummy or specified) preference dataset.
    *   Use mock logic (from `core.reward_acquisition.py`) to simulate generating behavioral variant log-probabilities.
    *   Calculate and print token-level rewards.

2.  **Run TFPO-based Fine-grained Preference Tuning:**
    ```bash
    python llmdoctor/examples/run_tfpo_training.py
    ```
    This script will:
    *   Load configuration.
    *   Generate dummy "processed reward data" (output from Stage 1).
    *   Initialize placeholder `DoctorModel` and `ValueFunction` (from `core.tfpo_tuner.py`).
    *   Prepare trajectories for SubTB loss using `data.utils.prepare_data_for_tfpo_training`.
    *   Run a mock training loop, printing illustrative loss values.

3.  **Run Online Alignment:**
    ```bash
    python llmdoctor/examples/run_online_alignment.py
    ```
    This script will:
    *   Load configuration.
    *   Initialize mock Patient and Doctor models.
    *   Use the `OnlineAlignment` module to generate text for example prompts, demonstrating how the patient and doctor model outputs would be combined. The generated text will be random due to mock models.

**Expected Output from Examples:**
The scripts will print detailed logs of their operations, including configuration loaded, data processing steps, and mock model interactions. Since they use placeholders, the actual outputs (rewards, generated text) will be illustrative of the process rather than meaningful results.

## Future Development & Contributions

To make LLMdoctor a fully functional tool, the following areas need development:

*   **Integration of Real LLMs**: Replace mock logic with actual model loading (using `models/model_loader.py`) and inference calls within the core modules.
*   **Behavioral Prompting Strategy**: Develop and implement robust prompting techniques for eliciting distinct positive and negative behaviors from the patient LLM in Stage 1.
*   **Doctor Model & Value Function Implementation**: Define and implement the actual architectures for `DoctorModel` and `ValueFunction` (e.g., using Hugging Face transformers for the Doctor Model).
*   **State Representation & Q-Value Calculation**: Implement sophisticated methods for deriving state representations for the $V_\phi$ and ensure the $Q(s_t)$ calculation aligns with the theoretical underpinnings of TFPO.
*   **Value Discrimination Data**: Implement the logic in `data/utils.py` to correctly prepare data pairs for the value discrimination loss.
*   **Vocabulary Alignment**: Develop a robust solution for aligning outputs if patient and doctor models have different vocabularies.
*   **Evaluation Harness**: Add scripts and utilities for evaluating the alignment performance.
*   **Checkpointing and Resuming**: Implement saving and loading for TFPO training.

Contributions are welcome! Please refer to the paper for detailed algorithmic descriptions.

## Citation

If you use or refer to the LLMdoctor framework or the underlying methodology, please cite the original paper (details to be added once available).
