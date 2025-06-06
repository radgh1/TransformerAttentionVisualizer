title: TransformerAttentionVisualizer
emoji: 🦀
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.27.0
app_file: app.py
pinned: false
short_description: TransformerAttentionVisualizer

# Transformer Attention Visualizer\n\nAn interactive tool to visualize attention mechanisms in DistilGPT-2, deployed on Hugging Face Spaces at [https://huggingface.co/spaces/raddev1/TransformerAttentionVisualizer](https://huggingface.co/spaces/raddev1/TransformerAttentionVisualizer).\n\n## Features\n- Visualizes embeddings, raw attention scores, and attention weights as heatmaps.\n- Includes a color key: Purple (small attention), Green (medium), Yellow (big attention).\n- Shows attention on predicted words with [PRED] labels.\n\n## Setup\n1. Clone the repository.\n2. Install dependencies: pip install -r requirements.txt\n3. Run the app: python app.py\n\n## Usage\nEnter a sentence (e.g., 'The cat sits') and a Look Ahead value (e.g., 3) to see the model's attention patterns.

### How It Works
    This application uses DistilGPT-2, a lightweight version of GPT-2, to analyze and visualize attention mechanisms. Enter a sentence and specify how many words to predict (Look Ahead). The app will:
    - Predict the next words based on the input sentence.
    - Display the updated sentence (input + predicted words).
    - Generate three visualizations:
        # Default Mode Network (DMN)   
        Brain Role: Active during rest, self-reflection, and mind-wandering; can disrupt focus with negative self-talk if unregulated.  
        Transformer Parallel: The transformer’s “baseline” state—its pre-trained weights and embeddings before task-specific attention kicks in. Excessive “noise” in these weights (e.g., irrelevant features) could mirror an overactive DMN.  
        Visualizer Tie-In: Show how the model’s initial token embeddings (pre-attention) might carry broad, unfocused information, which attention then refines.
        
        # Salience Network (SN)  
        Brain Role: Filters internal/external inputs, assigns value to what’s important (value tagging).  
        Transformer Parallel: The attention scores before softmax normalization—raw values that indicate which tokens the model deems salient.  
        Visualizer Tie-In: Display these raw scores alongside the final weights to show the “filtering” process, like the SN picking what matters.

        # Attention Network (AN)  
        Brain Role: Directs (DAN) and sustains (VAN) focus, filtering distractions.  
        Transformer Parallel: The softmax-normalized attention weights and their application across layers, focusing the model on key inputs while ignoring noise.  
        Visualizer Tie-In: Visualize attention heatmaps as the AN in action—showing how focus shifts across tokens and layers.

    The color key in the heatmaps indicates attention intensity: **Purple** (low), **Green** (medium), **Yellow** (high).
    