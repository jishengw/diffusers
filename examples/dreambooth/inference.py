#
# # model_id = "./test_model/model_i"
# #             #"logs/dreamsbooth-lora/events.out.tfevents.1711036363.DESKTOP-HINRCFR.17392.0")
from diffusers import StableDiffusionPipeline
import os
# model_id = "stabilityai/stable-diffusion-2-1"
# pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")
# # pipe.load_lora_weights("./fined_tuned_model/checkpoint-50") # Choose a checkpoint
model_address = "model_four_cups_300"
model_address = "model_cube_lr_5e5"
prompt = "a picture of rubik's cube"

# for i in range(10):
#     pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0].save(prefix+"/"+str(i)+".png")
#
from diffusers import AutoPipelineForText2Image
import torch
if not os.path.exists(model_address+"_result"):
    os.makedirs(model_address+"_result")

prefix = model_address + "_result/default_"+prompt
if not os.path.exists(prefix):
    os.mkdir(prefix)
pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16).to("cuda")
for i in range(50):
    pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0].save(prefix+"/"+str(i)+".png")
# prefix = model_address + "_result/fine_tuned_"+prompt
# if not os.path.exists(prefix):
#     os.mkdir(prefix)
# pipeline.load_lora_weights(model_address, weight_name="pytorch_lora_weights.safetensors")
# for i in range(50):
#     pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0].save(prefix+"/"+str(i)+".png")


# model_address = "model_four_cups_300"
# prompt_ini = "a picture of {} cups"
# l = ["one", "two", "three", "four", "five", "six", "seven", "eight", "night", "ten"]
# if not os.path.exists(model_address+"_result_round"):
#     os.makedirs(model_address+"_result_round")
# for i in l:
#     prompt = prompt_ini.format(i)
#     prefix = model_address+"_result_round/default_"+prompt
#     if not os.path.exists(prefix):
#         os.mkdir(prefix)
#     pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-2-1",
#                                                          torch_dtype=torch.float16).to("cuda")
#     for i in range(20):
#         pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0].save(prefix + "/" + str(i) + ".png")
#     prefix = model_address + "_result_round/fine_tuned_" + prompt
#     if not os.path.exists(prefix):
#         os.mkdir(prefix)
#     pipeline.load_lora_weights(model_address, weight_name="pytorch_lora_weights.safetensors")
#     for i in range(20):
#         pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0].save(prefix + "/" + str(i) + ".png")
