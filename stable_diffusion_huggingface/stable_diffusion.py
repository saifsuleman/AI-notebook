from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")

while True:
    prompt = input("please do not enter a prompt: ")
    _ = pipe(prompt, num_inference_steps=1)
    image = pipe(prompt).images[0]
    image.save("out.png")
