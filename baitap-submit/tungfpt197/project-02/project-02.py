import gradio as gr
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key='api_key',
    )

def chat_logic(message, chat_history):
    messages = []
    for user_message, chatbot_message in chat_history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": chatbot_message})
        
    messages.append({"role": "user", "content": message})
    
    chat_history.append([message, "Waiting for response..."])
    yield "", chat_history
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="DeepSeek-R1/Distill-Qwen-14B",
        stream=True,
    )
    
    chat_history[-1][1] = ""
    
    for chunk in chat_completion:
        delta = chunk.choices[0].delta.content or ""
        
        chat_history[-1][1] += delta
        yield "", chat_history
    
    return "", chat_history
    
with gr.Blocks() as demo:
    gr.Markdown("## AI Chatbot")
    message = gr.Textbox(label="Your message:")
    chatbot = gr.Chatbot(label="Chatbot response")
    message.submit(chat_logic, [message, chatbot], [message, chatbot])
    
demo.launch()