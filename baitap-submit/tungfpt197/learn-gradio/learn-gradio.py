from openai import OpenAI
import gradio as gr

def greet(user_input):
    client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key='api_key',
    )

    
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": user_input,
        }
    ],
    model="DeepSeek-R1/Distill-Qwen-14B",
    )

    return chat_completion.choices[0].message.content
        
demo = gr.Interface(fn=greet, inputs="text", outputs="text")

demo.launch()



