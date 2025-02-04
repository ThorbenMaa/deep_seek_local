from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr


# load model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="./",
    device_map="auto",
    low_cpu_mem_usage=True  # Essential for 8GB RAM
).eval()  # Set to eval mode immediately

# host webapp
def generate_text(prompt, max_length=100, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio UI
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=3, placeholder="Enter your prompt..."),
        gr.Slider(50, 500, value=100, label="Max Length"),
        gr.Slider(0.1, 1.0, value=0.7, label="Temperature")
    ],
    outputs="text",
    title="DeepSeek-R1-Distill-Qwen-1.5B Demo",
    description="A distilled 1.5B parameter model for efficient local AI."
)

demo.launch(share=False)  # Access via http://localhost:7860