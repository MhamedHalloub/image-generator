from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch

app = FastAPI()  # <--- THIS LINE is what uvicorn looks for

# Allow frontend requests (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cpu")


class Prompt(BaseModel):
    text: str

@app.post("/generate/")
def generate_image(prompt: Prompt):
    image = pipe(prompt.text).images[0]
    image.save("../output/output.png")
    return {"message": "Image generated", "filename": "output.png"}

@app.get("/image/")
def get_image():
    return FileResponse("../output/output.png", media_type="image/png")
