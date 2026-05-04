from inference.nora import Nora
from PIL import Image
import numpy as np

nora = Nora(device='cuda')

imarray = np.random.rand(640, 640, 3) * 255

image: Image.Image = Image.fromarray(imarray.astype('uint8')).convert('RGB')

instruction = "pick the object"

actions = nora.inference(
    image=image,
    instruction=instruction,
    skip_unnorm=True,  # environment uses normalized actions; skip dataset stats lookup
)

print(actions)