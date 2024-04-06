
# model_id = "./test_model/model_i"
#             #"logs/dreamsbooth-lora/events.out.tfevents.1711036363.DESKTOP-HINRCFR.17392.0")
from diffusers import StableDiffusionPipeline
import os
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")
pipe.load_lora_weights("./fined_tuned_model/checkpoint-50") # Choose a checkpoint
prompt = "2 glasses of wine"
prefix = "fine_tuned_2_glasses_of_wine"
if not os.path.exists(prefix):
    os.mkdir(prefix)
for i in range(10):
    pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0].save(prefix+"/"+str(i)+".png")

