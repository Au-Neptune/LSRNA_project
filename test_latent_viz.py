import os
import torch
import random
import json
from torchvision import transforms
from diffusers import DDIMScheduler, AutoencoderKL
from pipeline_lsrna_demofusion_sdxl import DemoFusionLSRNASDXLPipeline
from PIL import Image

# ==================== 全域常數設定 ====================
# 輸出設定
OUTPUT_DIR = "latent_viz_output"

# 模型設定
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
MODEL_CKPT = "stabilityai/stable-diffusion-xl-base-1.0"
LSR_PATH = "lsr/swinir-liif-latent-sdxl.pth"
DEVICE = "cuda"
DTYPE = torch.float16

# 圖像設定
HEIGHT = 2048
WIDTH = 2048
INPUT_SIZE = 1024

# 資料集設定
FOLDER = "/home/m11215122/datasets/OpenImages/valid/"
CAPTION_FILE = "/home/m11215122/datasets/OpenImages/captions.json"

# 生成設定
SEED = 42
NEGATIVE_PROMPT = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
VIEW_BATCH_SIZE = 8
STRIDE_RATIO = 0.5
INVERSION_DEPTH = 30
RNA_MAX_STD = 1.8
VIEW_LATENTS = True

# 輸出標籤
OUTPUT_LABELS = [
    "1_1x_reference",
    "2_upsampled_latent",
    "3_noisy_latent",
    "4_final_image",
]


# ==================== 輔助函式 ====================
def load_and_process_image(pil_image):
    transform = transforms.Compose(
        [
            # 1. Resize 輸入單個數字 1024：
            # PyTorch 會自動將「短邊」縮放到 1024，長邊則按比例放大（會大於 1024）
            transforms.Resize(1024), 
            
            # 2. CenterCrop：
            # 從圖片正中心切出 1024x1024，長邊多餘的部分會被自動丟掉
            transforms.CenterCrop(1024), 

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = transform(pil_image)
    image = image.unsqueeze(0).half()
    return image


def pad_image(image):
    """將圖像填充為正方形"""
    w, h = image.size
    if w == h:
        return image
    elif w > h:
        new_image = Image.new(image.mode, (w, w), (0, 0, 0))
        pad_h = (w - h) // 2
        new_image.paste(image, (0, pad_h))
        return new_image
    else:
        new_image = Image.new(image.mode, (h, h), (0, 0, 0))
        pad_w = (h - w) // 2
        new_image.paste(image, (pad_w, 0))
        return new_image


def load_pipeline():
    """載入並初始化 pipeline"""
    vae = AutoencoderKL.from_pretrained(VAE_MODEL, torch_dtype=DTYPE)
    scheduler = DDIMScheduler.from_pretrained(MODEL_CKPT, subfolder="scheduler")
    
    print("Loading pipeline...")
    pipe = DemoFusionLSRNASDXLPipeline.from_pretrained(
        MODEL_CKPT, scheduler=scheduler, vae=vae, torch_dtype=DTYPE
    ).to(DEVICE)
    pipe.vae.enable_tiling()
    
    return pipe


def load_image_and_prompt():
    """隨機選取圖像並載入對應的 prompt"""
    image_file_name = random.choice(os.listdir(FOLDER))
    
    with open(CAPTION_FILE, "r") as f:
        captions_data = json.load(f)
    
    prompt = captions_data[image_file_name]
    input_image = Image.open(os.path.join(FOLDER, image_file_name)).convert("RGB")
    # padded_image = pad_image(input_image).resize((INPUT_SIZE, INPUT_SIZE)).convert("RGB")
    image_lr = load_and_process_image(input_image).to(DEVICE)
    
    return prompt, image_lr


def save_images(images):
    """儲存生成的圖像"""
    print(f"Generation complete. Received {len(images)} images.")
    
    # 預期 images 列表包含：
    # 0: 1X Reference Image
    # 1: Upsampled Latent Visualization (if scale > 1)
    # 2: Final Image
    # 3: Final Latent Visualization
    
    if len(images) > len(OUTPUT_LABELS):
        # 如果有更多圖片，就用數字命名
        for i, img in enumerate(images):
            img.save(os.path.join(OUTPUT_DIR, f"output_{i}.png"))
            print(f"Saved output_{i}.png")
    else:
        for i, img in enumerate(images):
            label = OUTPUT_LABELS[i] if i < len(OUTPUT_LABELS) else f"output_{i}"
            img.save(os.path.join(OUTPUT_DIR, f"{label}.png"))
            print(f"Saved {label}.png")


def generate_images(pipe, prompt, image_lr):
    """執行圖像生成"""
    print("Starting generation with latent visualization...")
    
    with torch.inference_mode():
        images = pipe(
            prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=HEIGHT,
            width=WIDTH,
            view_batch_size=VIEW_BATCH_SIZE,
            stride_ratio=STRIDE_RATIO,
            lsr_path=LSR_PATH,
            inversion_depth=INVERSION_DEPTH,
            rna_max_std=RNA_MAX_STD,
            view_latents=VIEW_LATENTS,
            image_lr=image_lr
        )
    
    return images


# ==================== 主程式 ====================
def main():
    # 設定輸出路徑
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 設定隨機種子
    torch.manual_seed(SEED)
    
    # 載入模型
    pipe = load_pipeline()
    
    # 載入圖像和 prompt
    prompt, image_lr = load_image_and_prompt()
    
    # 生成圖像
    images = generate_images(pipe, prompt, image_lr)
    
    # 儲存圖像
    save_images(images)


if __name__ == "__main__":
    main()