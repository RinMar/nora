import os
from nora.inference.nora import Nora
from PIL import Image
import numpy as np
import gymnasium as gym
import panda_gym  # Noqa

# Create observations directory
output_dir = "observations"
os.makedirs(output_dir, exist_ok=True)

nora = Nora(device='cuda')

env = gym.make('PandaPickAndPlace-v3', render_mode='rgb_array')
obs, info = env.reset()

instruction = "pick one cube"

for i in range(20):
    # Capture and save observation
    img_array = env.render()
    image: Image.Image = Image.fromarray(img_array.astype('uint8')).convert('RGB')
    image.save(os.path.join(output_dir, f"step_{i:02d}.png"))

    # Inference
    actions = nora.inference(
        image=image,
        instruction=instruction,
        skip_unnorm=True,  # environment uses normalized actions; skip dataset stats lookup
    )

    action = actions[0]
    # Map Nora action to panda-gym action format [x, y, z, gripper]
    env_action = np.array([action[0], action[1], action[2], action[-1]])

    # Environment step
    obs, reward, terminated, truncated, info = env.step(env_action)

    print(f"Step {i:02d}: Reward: {reward}")

    if terminated or truncated:
        print(f"Episode finished at step {i:02d}")
        break

env.close()
