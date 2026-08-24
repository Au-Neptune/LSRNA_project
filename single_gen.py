import os
import random
import shutil
import numpy as np
import torch
from diffusers import DDIMScheduler, AutoencoderKL
from pipeline_demofusion_sdxl import DemoFusionSDXLPipeline
from pipeline_lsrna_demofusion_sdxl import DemoFusionLSRNASDXLPipeline


# ==================== 全域常數設定 ====================
# 輸出設定
OUTPUT_PATH = "test"

# 模型設定
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
MODEL_CKPT = "stabilityai/stable-diffusion-xl-base-1.0"

LSR_PATHS = {
        "DRCT": "lsr_training/save/drct-liif-latent-sdxl/iter_last.pth",
        "DAT": "lsr_training/save/dat-liif-latent-sdxl/iter_last.pth",
        "HAT": "lsr_training/save/hat-liif-latent-sdxl/iter_last.pth",
        "swinIR": "lsr_training/save/swinir-liif-latent-sdxl/iter_last.pth"
}
DEVICE = "cuda"
DTYPE = torch.float16

# 圖像設定
HEIGHT = 4096
WIDTH = 4096
# CAPTION = "A bright red apple on a smooth white table, glistening skin, with a leaf and a tiny stem."
CAPTION = "Detailed close-up profile of a raptor with white feathers and gray wings, peering through wire mesh against blurred jungle foliage."
NEGATIVE_PROMPT = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
SEED = 42

# ==================== 輔助函式 ====================
def set_seed(seed):
    """設定所有隨機種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pipeline(lsrna = True):
    """載入並初始化 pipeline"""    
    vae = AutoencoderKL.from_pretrained(VAE_MODEL, torch_dtype=DTYPE)
    scheduler = DDIMScheduler.from_pretrained(MODEL_CKPT, subfolder="scheduler")
    
    if lsrna:
        print("載入 LSRNA 模型...")
        pipe = DemoFusionLSRNASDXLPipeline.from_pretrained(
            MODEL_CKPT,
            scheduler=scheduler,
            vae=vae,
            torch_dtype=DTYPE
        ).to(DEVICE)
    else:
        print("載入 Demofusion 基本模型...")
        pipe = DemoFusionSDXLPipeline.from_pretrained(
            MODEL_CKPT, 
            scheduler=scheduler, 
            vae=vae, 
            torch_dtype=DTYPE
        ).to(DEVICE)

    pipe.vae.enable_tiling()
    print("模型載入完成")
    return pipe

def generate_single_image(pipe, lsr_path, prompt):
    """
    生成單張圖像 (加入錯誤處理)
    
    Returns:
        PIL.Image 或 None (如果失敗)
    """
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    try:
        if lsr_path is not None:
            with torch.inference_mode():
                image = pipe(
                    prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    generator=generator,
                    lsr_path=lsr_path,
                    height=HEIGHT,
                    width=WIDTH,
                )
        else:
            with torch.inference_mode():
                image = pipe(
                    prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    generator=generator,
                    height=HEIGHT,
                    width=WIDTH,
                )
    except Exception as e:
        print(f"生成圖像時發生錯誤: {e}")
        return None
    finally:
        torch.cuda.empty_cache()
    return image
    
# ==================== 主程式 ====================
def main():
    """主程式入口"""
    
    # 設定隨機種子
    set_seed(SEED)
    print(f"Random seed: {SEED}")

    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)

    os.makedirs(OUTPUT_PATH)
    
    # 生成 lsrna 模塊圖像
    pipe = load_pipeline(lsrna = True)
    for key, path in LSR_PATHS.items():
        print(f"正在使用 LSR 模型: {key}")
        image = generate_single_image(pipe, path, CAPTION)
        for i, item in enumerate(image):
            output_path = os.path.join(OUTPUT_PATH, f"{key}_{i+1}.png")
            item.save(output_path)
        print(f" LSR 模型 {key} 生成完成\n")

    del pipe
    torch.cuda.empty_cache()

    # 生成基礎模型圖像
    pipe = load_pipeline(lsrna = False)
    image = generate_single_image(pipe, None, CAPTION)
    
    for i, item in enumerate(image):
        output_path = os.path.join(OUTPUT_PATH, f"Demofusion_{i+1}.png")
        item.save(output_path)
    print("基礎模型生成完成")
    
    del pipe
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()