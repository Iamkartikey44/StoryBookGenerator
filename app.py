import streamlit as st
import json
import torch
from diffusers import DiffusionPipeline
from openai import Openai
from PIL import Image,ImageDraw,ImageFont
from dotenv import load_dotenv
st.title("Story Book Generator")

model_name = "stabilityai/stable-diffusion-xl-base-1.0"

lora_path =  "StorybookRedmondV2-KidsBook-KidsRedmAF.safetensors"

load_dotenv()
client = Openai()

def wrap_text(text,font,max_width):
    lines = []
    words = text.split()

    while words:
        line = ""
        while words and font.getmask(line + words[0]).getbbox()[2] <= max_width:
            line += (words.pop(0) + ' ')
        lines.append(line.strip())

    return lines        
    
def text_on_image(image,text,font_path=None,position='top',font_size=50,
                  bg_color=(0,0,0,128),line_spacing=20,padding_top=20,padding_bottom=20,
                  padding_sides=0):
    draw = ImageDraw.Draw(image)

    try:
        if font_path:
            font = ImageFont.truetype(font_path,font_size)
    except IOError:
        print("Default font will be used as specified font was not found.")
        font = ImageFont.load_default()

    max_width = image.width - 2 *padding_sides
    wrapped_text = wrap_text(text,font,max_width)

    text_heights = [font.getmask(line).getbbox()[3] for line in wrapped_text]
    text_block_height = sum(text_heights) + line_spacing * (len(wrapped_text)-1) 

    y = padding_top if position=='top' else image.height - text_block_height - padding_bottom

    bg_rectangle_top = y-padding_top
    bg_rectangle_bottom = y + text_block_height + padding_bottom
    bg_rectangle_left = padding_sides
    bg_rectangle_right = image.width - padding_sides
    if position == "top":
        draw.rectangle([(bg_rectangle_left, bg_rectangle_top), (bg_rectangle_right, bg_rectangle_bottom)],
                       fill=bg_color)
    else:
        draw.rectangle([(bg_rectangle_left, image.height - text_block_height - padding_top - padding_bottom),
                        (bg_rectangle_right, image.height)], fill=bg_color)
 
    # Reset y position for text
    y = padding_top if position == "top" else image.height - text_block_height - padding_bottom
 
    # Draw text on image
    for line in wrapped_text:
        line_width = font.getmask(line).getbbox()[2]
        x = (image.width - line_width) / 2
        draw.text((x, y), line, fill=(255, 255, 255), font=font)
        y += font.getmask(line).getbbox()[3] + line_spacing
 
    # Display the image
    return image           



@st.cache_resource()
def load_image_model(model_name,model_path_refiner=None,lora_path=None,torch_dtype=torch.float16,variant='fp16',use_safetensors=True):
    base = DiffusionPipeline.from_pretrained(model_name,
                                             torch_dtype=torch_dtype,
                                             variant=variant,
                                             use_safetensors=use_safetensors)
    
    if lora_path:
        base.load_lora_weights(lora_path)
    
    if model_path_refiner:
        refiner = DiffusionPipeline.from_pretrained(
            model_path_refiner,
            text_encoder_2=base.text_encoder_2,
            vae = base.vae,
            use_safetensors=use_safetensors,
            variant=variant
        )
        return base,refiner    
    else:
        refiner=None


    return base,refiner
def generate_image(base,prompt,refiner=None,num_inference_steps=20,guidance_scale=15,
                   high_noise_frac=0.8,output_type='latent',verbose=False,temperature=0.7):

    if refiner:
        image = base(prompt=prompt,num_inference_steps=num_inference_steps,denoising_end=high_noise_frac,
                     output_type=output_type,verbose=verbose,guidance_scale=guidance_scale,temperature=temperature).images
        image = refiner(prompt=prompt,num_inference_steps=num_inference_steps,denoising_end=high_noise_frac,
                        image=image,verbose=verbose).images[0]
    else:
        image = base(prompt=prompt,num_inference_steps=num_inference_steps,
                 guidance_scale=guidance_scale).images[0]
    
    return image
def generate_description(client,prompt,text_area_placeholder=None,
                                model='gpt-3.5-turbo',temperature=0.7,
                                max_tokens=3000,top_p=1,frequency_penalty=0,
                                presence_penalty=0,stream=True,html=False):
    
    response = client.chat.completions.create(
        model=model,
        messages=[{'role':'user','content':prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p = top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stream=stream
    )
    complete_response  = []

    for chunk in response:
        if chunk.choices[0].delta.content:
            complete_response.append(chunk.choices[0].delta.content)
            result_string = ''.join(complete_response)

            #Auto Scroll
            lines = result_string.count('\n') + 1
            avg_chars_per_line = 80
            lines += len(result_string)//avg_chars_per_line
            height_per_line = 20
            total_height = lines * height_per_line

            if text_area_placeholder:
                if html:
                    text_area_placeholder.markdown(result_string,unsafe_allow_html=True)
                else:
                    text_area_placeholder.text_area("Generated Text",value=result_string,height=total_height) 
    result_string = ''.join(complete_response)
    words = len(result_string.split())
    st.text(f"Total Words Generated: {words}")

    return result_string                    

user_input = st.text_input("Enter your prompt",value='A princess that saved the village from dragon')
base,refiner = load_image_model(model_name,None,lora_path)
lora_trigger = 'Kids Book'
prompt_user = user_input + lora_trigger

prompt_input = """
write a 5 page story for children with a character. 
- Each page will have a single line.
- For each page give a description that will be used to generate an image for that page. Don't use names in the description of the image. 
- Describe the appearance of the character in each image description, as the AI does not have a memory.
- Don't use words like the same character because the ai model does not have any memory. 
- Use white skin color for the character and mention it in each image description.
- follow the output format as follows:
{
"page1":"page 1 line",
"page2":"page 2 line",
"page3":"page 3 line",
"page4":"page 4 line",
"page5":"page 5 line",
"page1_image_description":"page1 image description",
"page2_image_description":"page2 image description",
"page3_image_description":"page3 image description",
"page4_image_description":"page4 image description",
"page5_image_description":"page5 image description",
}
 
Here are the details: 
"""
prompt =  prompt_input + prompt_user

page1_text = "Once upon a time, in a faraway kingdom, there lived a brave princess. "
page1_img = Image.open('img.jpg')

if st.button("Generate Story"):
    with st.spinner("Generating Story Book..."):

        story = generate_description(client,prompt)

        story_json = json.loads(story)

        for i in range(1,6):
            st.text(f"Page {str(i)}")
            st.write(story_json['page' + str(i)])
            st.write(story_json['page' + str(i) + "_image_description"])
            page_prompt = story_json['page' + str(i) + '_image_description']
            page_text = story_json['page' + str(i)]
            generate_img = generate_image(base,page_prompt)
            image = text_on_image(generate_img,page_text,font_page="one_trick_pony_tt.ttf",
                                  position ='top' )
            st.image(image,channels='BGR',use_column_width=True)

