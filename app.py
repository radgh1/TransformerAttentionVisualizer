# app.py
import gradio as gr
import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Load a smaller model for hosting
try:
    print("Loading DistilGPT-2 models...")
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2', force_download=True)
    model = GPT2Model.from_pretrained('distilgpt2', force_download=True)
    model_lm = GPT2LMHeadModel.from_pretrained('distilgpt2', force_download=True)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

def get_attention_data(sentence, layer_idx=0):
    try:
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
        raw_attention = outputs.attentions[layer_idx][0, 0].detach().numpy()
        norm_attention = torch.softmax(torch.tensor(raw_attention), dim=-1).numpy()
        embeddings = outputs.last_hidden_state[0].detach().numpy()
        return raw_attention, norm_attention, embeddings, inputs
    except Exception as e:
        print(f"Error in get_attention_data: {e}")
        raise

def predict_next_words(sentence, num_words=1):
    try:
        current_sentence = sentence
        predicted_words = []
        for _ in range(num_words):
            inputs = tokenizer(current_sentence, return_tensors="pt")
            outputs = model_lm(**inputs)
            logits = outputs.logits[:, -1, :]
            probabilities = torch.softmax(logits, dim=-1)
            predicted_token_id = torch.argmax(probabilities, dim=-1).item()
            predicted_word = tokenizer.decode(predicted_token_id).strip()
            predicted_words.append(predicted_word)
            current_sentence = f"{current_sentence} {predicted_word}".strip()
        return " ".join(predicted_words), predicted_words
    except Exception as e:
        print(f"Error in predict_next_words: {e}")
        raise

def plot_heatmap(data, labels, title, for_gradio=False):
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.matshow(data, cmap='viridis')
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        ax.set_title(title, pad=20)
        plt.colorbar(im)

        plt.figtext(0.5, 0.02, "Color Key: Purple = Small (not much attention), Green = Medium, Yellow = Big (lots of attention)", 
                    ha="center", fontsize=8, wrap=True, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])

        if for_gradio:
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            rgba = np.asarray(canvas.buffer_rgba())
            img_array = rgba[:, :, :3]
            plt.close()
            return img_array
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.5)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error in plot_heatmap: {e}")
        raise

def interactive_visualize(sentence, look_ahead):
    try:
        look_ahead = max(1, int(look_ahead))
        # Predict the next words
        next_words_str, next_words_list = predict_next_words(sentence, look_ahead)
        # Combine the input sentence and predicted words
        combined_sentence = f"{sentence} {next_words_str}".strip()
        # Get attention data for the combined sentence
        raw_att, norm_attention, embeddings, inputs = get_attention_data(combined_sentence)
        # Get tokens for the combined sentence
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        # Create labels with a marker to distinguish input tokens from predicted tokens
        input_tokens = tokenizer.tokenize(sentence)
        labels = input_tokens + ['[PRED] ' + token for token in next_words_list]
        # Plot heatmaps
        dmn_img = plot_heatmap(embeddings.T[:10], tokens, "DMN: Embeddings (Input + Predicted)", for_gradio=True)
        sn_img = plot_heatmap(raw_att, tokens, "SN: Raw Scores (Input + Predicted)", for_gradio=True)
        an_img = plot_heatmap(norm_attention, tokens, "AN: Attention Weights (Input + Predicted)", for_gradio=True)
        return combined_sentence, next_words_str, dmn_img, sn_img, an_img, combined_sentence
    except Exception as e:
        print(f"Error in interactive_visualize: {e}")
        raise

# Gradio interface using gr.Blocks
print("Launching Gradio interface...")
with gr.Blocks(title="Interactive Transformer Attention Visualizer") as demo:
    gr.Markdown("# Interactive Transformer Attention Visualizer")
    gr.Markdown("""
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
    """)
    
    # Inputs
    sentence_input = gr.Textbox(label="Sentence", value="The cat sits", interactive=True)
    look_ahead_input = gr.Number(label="Look Ahead", value=1)
    submit_button = gr.Button("Submit")
    
    # Outputs
    updated_sentence_output = gr.Textbox(label="Updated Sentence (Input + Predicted)")
    predicted_words_output = gr.Textbox(label="Predicted Next Words")
    dmn_output = gr.Image(label="DMN: Embeddings (Input + Predicted)")
    sn_output = gr.Image(label="SN: Raw Scores (Input + Predicted)")
    an_output = gr.Image(label="AN: Attention Weights (Input + Predicted)")
    
    # Event handler
    submit_button.click(
        fn=interactive_visualize,
        inputs=[sentence_input, look_ahead_input],
        outputs=[
            updated_sentence_output,
            predicted_words_output,
            dmn_output,
            sn_output,
            an_output,
            sentence_input
        ]
    )

demo.launch(share=False)
print("Gradio interface launched. Check the URL above to access the dashboard.")