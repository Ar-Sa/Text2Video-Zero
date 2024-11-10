import torch
from model import Model

model = Model(device = "cuda", dtype = torch.float16)
prompt = "animated video for below story- Once in a bright, cheerful forest, a clever little rabbit named Benny loved to boast about how fast he could run. One sunny day, he challenged his friend Tilly the tortoise to a race. Confident he would win easily, Benny dashed ahead and decided to take a nap under a shady tree. Meanwhile, Tilly kept moving slowly but steadily. When Benny woke up, he saw Tilly nearing the finish line. Panicking, he sprinted, but it was too late! Tilly won the race. Benny learned that boasting doesn’t lead to success; hard work and perseverance do. Always believe in yourself!"
params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 8}

out_path, fps = f"./text2video.mp4", 4
model.process_text2video(prompt, fps = fps, path = out_path, **params)
