import os
import json
import random
import time
import shutil
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from diffusers import DDIMScheduler, AutoencoderKL
from pipeline_lsrna_demofusion_sdxl import DemoFusionLSRNASDXLPipeline
from pipeline_demofusion_sdxl import DemoFusionSDXLPipeline
from cleanfid import fid
from concurrent.futures import ProcessPoolExecutor, as_completed


# ==================== 全域常數設定 ====================
# 輸出設定
OUTPUT_PATH = "eval_results/dat"

# 模型設定
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
MODEL_CKPT = "stabilityai/stable-diffusion-xl-base-1.0"
LSR_PATH = "lsr_training/save/dat-liif-latent-sdxl/iter_last.pth"
DEVICE = "cuda"
DTYPE = torch.float16

# 圖像設定
HEIGHT = 2048
WIDTH = 2048
INPUT_SIZE = 1024

# 資料集設定
HR_PATH = "/home/m11215122/datasets/OpenImages/valid/HR"
CAPTION_FILE = "/home/m11215122/datasets/OpenImages/captions.json"

# 生成設定
SEED = 42
NEGATIVE_PROMPT = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
VIEW_BATCH_SIZE = 16
STRIDE_RATIO = 0.5
INVERSION_DEPTH = 30
RNA_MIN_STD = 0.0
RNA_MAX_STD = 1.2
COSINE_SCALE_1 = 3
COSINE_SCALE_2 = 1
COSINE_SCALE_3 = 1
SIGMA = 0.8

# Patch 設定
PATCH_SIZE = 512
TOTAL_PATCHES = 50000

# 控制開關
SKIP_GEN = False  # 設為 True 跳過生成步驟
TEXT_TO_IMAGE = True  # 設為 True 使用 text-to-image 模式

# clean-fid 設定
CLEANFID_MODE = "clean"  # "clean" 或 "legacy"
CLEANFID_NUM_WORKERS = 4
CLEANFID_BATCH_SIZE = 32


# ==================== 輔助函式 ====================
def set_seed(seed):
    """設定所有隨機種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_and_process_image(pil_image, target_size=INPUT_SIZE):
    """
    改進的圖像預處理: 使用填充方式保持長寬比
    避免裁切導致的資訊損失
    """
    # 計算縮放比例 (保持長寬比)
    w, h = pil_image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 縮放圖片
    img = pil_image.resize((new_w, new_h), Image.BICUBIC)
    
    # 建立黑色背景並居中貼上
    padded = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    padded.paste(img, (paste_x, paste_y))
    
    # 轉換為 tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    image_tensor = transform(padded).unsqueeze(0).half()
    return image_tensor


def crop_single_image(args):
    """
    單一圖片的 patch 切割函數 (用於多進程)
    
    Args:
        args: (filename, src_dir, dst_dir, patches_per_img, patch_size, img_idx)
    
    Returns:
        切割的 patch 數量
    """
    filename, src_dir, dst_dir, patches_per_img, patch_size, img_idx = args
    
    try:
        img_path = os.path.join(src_dir, filename)
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        # 如果圖片太小，跳過
        if w < patch_size or h < patch_size:
            return 0
        
        # 隨機切割 patches
        patch_count = 0
        for i in range(patches_per_img):
            x = random.randint(0, w - patch_size)
            y = random.randint(0, h - patch_size)
            crop = img.crop((x, y, x + patch_size, y + patch_size))
            
            # 使用統一命名格式: idx_patchnum_filename
            output_name = f"{img_idx:05d}_{i:03d}_{filename}"
            crop.save(os.path.join(dst_dir, output_name))
            patch_count += 1
        
        return patch_count
        
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return 0


def prepare_patches_parallel(src_dir, dst_dir, patch_size=PATCH_SIZE, 
                             total_patches=TOTAL_PATCHES, max_workers=8):
    """
    並行化的 Patch 切割函數
    使用多進程大幅加速處理速度
    
    Args:
        src_dir: 來源圖片資料夾
        dst_dir: 目標 patch 資料夾
        patch_size: patch 尺寸
        total_patches: 總共要切割的 patch 數量
        max_workers: 最大進程數
    """
    # 清空並重建目標資料夾
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)
    
    # 收集所有圖片檔案
    files = sorted([
        f for f in os.listdir(src_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    if not files:
        print(f"⚠️ {src_dir} 中沒有圖片")
        return 0
    
    print(f"🔪 正在從 {src_dir} 切割 Patch...")
    print(f"   總圖片數: {len(files)}")
    
    # 計算每張圖要切多少 patches
    patches_per_img = max(1, total_patches // len(files))
    print(f"   每張圖切割: {patches_per_img} patches")
    print(f"   使用 {max_workers} 個進程並行處理")
    
    # 準備任務列表
    tasks = [
        (filename, src_dir, dst_dir, patches_per_img, patch_size, idx)
        for idx, filename in enumerate(files)
    ]
    
    # 使用進程池並行處理
    total_patches_created = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任務
        futures = {
            executor.submit(crop_single_image, task): task[0] 
            for task in tasks
        }
        
        # 顯示進度條
        with tqdm(total=len(files), desc="Cropping Patches") as pbar:
            for future in as_completed(futures):
                count = future.result()
                total_patches_created += count
                pbar.update(1)
    
    print(f"✅ 完成! 總共切割了 {total_patches_created} 個 patches")
    return total_patches_created


def load_pipeline():
    """載入並初始化 pipeline"""
    print("🔧 正在載入模型...")
    
    vae = AutoencoderKL.from_pretrained(VAE_MODEL, torch_dtype=DTYPE)
    scheduler = DDIMScheduler.from_pretrained(MODEL_CKPT, subfolder="scheduler")
    
    pipe = DemoFusionLSRNASDXLPipeline.from_pretrained(
        MODEL_CKPT, 
        scheduler=scheduler, 
        vae=vae, 
        torch_dtype=DTYPE
    ).to(DEVICE)
    
    pipe.vae.enable_tiling()
    
    print("✅ 模型載入完成")
    return pipe


def load_captions():
    """載入 caption 資料"""
    print(f"📖 正在載入 captions: {CAPTION_FILE}")
    with open(CAPTION_FILE, "r", encoding='utf-8') as f:
        captions_data = json.load(f)
    print(f"✅ 載入了 {len(captions_data)} 個 captions")
    return captions_data


def generate_single_image(pipe, prompt, image_lr):
    """
    生成單張圖像 (加入錯誤處理)
    
    Returns:
        PIL.Image 或 None (如果失敗)
    """
    try:
        with torch.inference_mode():
            image = pipe(
                prompt,
                negative_prompt=NEGATIVE_PROMPT,
                height=HEIGHT,
                width=WIDTH,
                view_batch_size=VIEW_BATCH_SIZE,
                stride_ratio=STRIDE_RATIO,
                lsr_path=LSR_PATH,
                inversion_depth=INVERSION_DEPTH,
                rna_min_std=RNA_MIN_STD,
                rna_max_std=RNA_MAX_STD,
                cosine_scale_1=COSINE_SCALE_1,
                cosine_scale_2=COSINE_SCALE_2,
                cosine_scale_3=COSINE_SCALE_3,
                sigma=SIGMA,
                image_lr=image_lr
            )[1]  # 取第二張圖
        return image
        
    except torch.cuda.OutOfMemoryError:
        print("❌ GPU 記憶體不足，清理快取...")
        torch.cuda.empty_cache()
        return None
        
    except Exception as e:
        print(f"❌ 生成失敗: {e}")
        return None


def generate_images(pipe, captions_data, gen_dir):
    """
    批量生成圖像 (支援中斷續傳)
    """
    print("🚀 開始批量生成圖片...")
    
    # 收集所有待處理的圖片
    hr_files = sorted([
        f for f in os.listdir(HR_PATH) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    # 檢查已生成的圖片 (支援續傳)
    existing_files = set()
    if os.path.exists(gen_dir):
        existing_files = set([
            f for f in os.listdir(gen_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
    
    remaining_files = [f for f in hr_files if f not in existing_files]
    
    print(f"📊 統計:")
    print(f"   總圖片數: {len(hr_files)}")
    print(f"   已完成: {len(existing_files)}")
    print(f"   剩餘: {len(remaining_files)}")
    
    if not remaining_files:
        print("✅ 所有圖片都已生成!")
        return 0.0
    
    inference_times = []
    successful_count = 0
    failed_count = 0
    
    try:
        for index, filename in enumerate(remaining_files):
            print(f"\n[{index+1}/{len(remaining_files)}] 處理: {filename}")
            
            # 從 caption 檔案中讀取對應的 prompt
            if filename in captions_data:
                current_prompt = captions_data[filename]
            else:
                print(f"⚠️ 找不到 caption，使用預設 prompt")
                current_prompt = "high resolution photography"
            
            print(f"   Prompt: {current_prompt[:80]}...")
            
            # 讀取並處理輸入圖像
            if not TEXT_TO_IMAGE:
                print("   使用 Image-to-Image 模式")
                try:
                    input_image = Image.open(os.path.join(HR_PATH, filename)).convert("RGB")
                    image_lr = load_and_process_image(input_image).to(DEVICE)
                except Exception as e:
                    print(f"❌ 圖片載入失敗: {e}")
                    failed_count += 1
                    continue
            else:
                print("   使用 Text-to-Image 模式")
                image_lr = None
            
            # 計時開始
            torch.cuda.synchronize()
            start_time = time.time()
            
            # 生成圖像
            image = generate_single_image(pipe, current_prompt, image_lr)
            
            if image is None:
                print(f"❌ 生成失敗，跳過")
                failed_count += 1
                continue
            
            # 計時結束
            torch.cuda.synchronize()
            end_time = time.time()
            elapsed = end_time - start_time
            inference_times.append(elapsed)
            
            # 存檔
            output_path = os.path.join(gen_dir, filename)
            image.save(output_path)
            successful_count += 1
            
            print(f"✅ 完成 (耗時: {elapsed:.2f}s)")
            
            # 每 10 張顯示平均時間
            if (index + 1) % 10 == 0:
                avg_so_far = sum(inference_times) / len(inference_times)
                print(f"\n📈 中間統計 ({index+1}/{len(remaining_files)}):")
                print(f"   平均生成時間: {avg_so_far:.4f} 秒/張")
                print(f"   成功: {successful_count}, 失敗: {failed_count}")
    
    except KeyboardInterrupt:
        print("\n⚠️ 使用者中斷，儲存統計資料...")
    
    # 計算並儲存統計
    if inference_times:
        avg_time = sum(inference_times) / len(inference_times)
        print(f"\n⏱️ 最終統計:")
        print(f"   成功生成: {successful_count} 張")
        print(f"   失敗: {failed_count} 張")
        print(f"   平均生成時間: {avg_time:.4f} 秒/張")
        
        # 儲存時間統計
        with open(os.path.join(OUTPUT_PATH, 'time_result.txt'), 'w') as f:
            f.write(f"Total images: {len(hr_files)}\n")
            f.write(f"Generated: {successful_count}\n")
            f.write(f"Failed: {failed_count}\n")
            f.write(f"Average Time: {avg_time:.4f} sec\n")
            f.write(f"Min Time: {min(inference_times):.4f} sec\n")
            f.write(f"Max Time: {max(inference_times):.4f} sec\n")
        
        return avg_time
    else:
        print("⚠️ 沒有成功生成任何圖片")
        return 0.0


def count_images(directory):
    """計算資料夾中的圖片數量"""
    if not os.path.exists(directory):
        return 0
    files = [
        f for f in os.listdir(directory) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    return len(files)


def calculate_metrics_with_cleanfid(gen_dir, real_dir):
    """
    使用 clean-fid 計算所有指標
    
    Args:
        gen_dir: 生成圖片資料夾
        real_dir: 真實圖片資料夾
    
    Returns:
        dict: 包含所有指標的字典
    """
    print("\n" + "="*60)
    print("📊 使用 clean-fid 計算評估指標")
    print("="*60)
    
    # 檢查資料夾
    gen_count = count_images(gen_dir)
    real_count = count_images(real_dir)
    
    print(f"\n📁 資料夾統計:")
    print(f"   生成圖片: {gen_count} 張 ({gen_dir})")
    print(f"   真實圖片: {real_count} 張 ({real_dir})")
    
    if gen_count == 0 or real_count == 0:
        raise ValueError("圖片資料夾為空!")
    
    metrics = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ========================================
    # 1. 計算 FID (Full Image)
    # ========================================
    print("\n" + "-"*60)
    print("📐 計算 FID (Fréchet Inception Distance)")
    print("-"*60)
    
    try:
        fid_score = fid.compute_fid(
            gen_dir,
            real_dir,
            mode=CLEANFID_MODE,
            num_workers=CLEANFID_NUM_WORKERS,
            batch_size=CLEANFID_BATCH_SIZE,
            device=device,
            verbose=True
        )
        metrics['FID'] = fid_score
        print(f"✅ FID: {fid_score:.4f}")
        
    except Exception as e:
        print(f"❌ FID 計算失敗: {e}")
        metrics['FID'] = float('nan')
    
    # ========================================
    # 2. 計算 KID (Full Image)
    # ========================================
    print("\n" + "-"*60)
    print("📐 計算 KID (Kernel Inception Distance)")
    print("-"*60)
    
    try:
        kid_score = fid.compute_kid(
            gen_dir,
            real_dir,
            mode=CLEANFID_MODE,
            num_workers=CLEANFID_NUM_WORKERS,
            batch_size=CLEANFID_BATCH_SIZE,
            device=device,
            verbose=True
        )
        metrics['KID'] = kid_score
        print(f"✅ KID: {kid_score:.6f}")
        
    except Exception as e:
        print(f"❌ KID 計算失敗: {e}")
        metrics['KID'] = float('nan')
    
    # ========================================
    # 3. 計算 pFID (Patch-based FID)
    # ========================================
    print("\n" + "-"*60)
    print("📐 計算 pFID (Patch-based FID)")
    print("-"*60)
    
    gen_patch_dir = os.path.join(OUTPUT_PATH, 'patches_gen')
    real_patch_dir = os.path.join(OUTPUT_PATH, 'patches_real')
    
    try:
        # 切割 patches (並行化加速)
        print("\n🔪 切割生成圖片的 patches...")
        prepare_patches_parallel(
            gen_dir, 
            gen_patch_dir,
            patch_size=PATCH_SIZE,
            total_patches=TOTAL_PATCHES,
            max_workers=8
        )
        
        print("\n🔪 切割真實圖片的 patches...")
        prepare_patches_parallel(
            real_dir, 
            real_patch_dir,
            patch_size=PATCH_SIZE,
            total_patches=TOTAL_PATCHES,
            max_workers=8
        )
        
        # 檢查 patch 數量
        gen_patch_count = count_images(gen_patch_dir)
        real_patch_count = count_images(real_patch_dir)
        print(f"\n📁 Patch 統計:")
        print(f"   生成圖片 patches: {gen_patch_count}")
        print(f"   真實圖片 patches: {real_patch_count}")
        
        # 計算 pFID
        print("\n計算 pFID...")
        pfid_score = fid.compute_fid(
            gen_patch_dir,
            real_patch_dir,
            mode=CLEANFID_MODE,
            num_workers=CLEANFID_NUM_WORKERS,
            batch_size=CLEANFID_BATCH_SIZE,
            device=device,
            verbose=True
        )
        metrics['pFID'] = pfid_score
        print(f"✅ pFID: {pfid_score:.4f}")
        
    except Exception as e:
        print(f"❌ pFID 計算失敗: {e}")
        import traceback
        traceback.print_exc()
        metrics['pFID'] = float('nan')
    
    # ========================================
    # 4. 計算 pKID (Patch-based KID)
    # ========================================
    print("\n" + "-"*60)
    print("📐 計算 pKID (Patch-based KID)")
    print("-"*60)
    
    try:
        # patches 已經在 pFID 階段切好了
        if os.path.exists(gen_patch_dir) and os.path.exists(real_patch_dir):
            pkid_score = fid.compute_kid(
                gen_patch_dir,
                real_patch_dir,
                mode=CLEANFID_MODE,
                num_workers=CLEANFID_NUM_WORKERS,
                batch_size=CLEANFID_BATCH_SIZE,
                device=device,
                verbose=True
            )
            metrics['pKID'] = pkid_score
            print(f"✅ pKID: {pkid_score:.6f}")
        else:
            raise ValueError("Patch 資料夾不存在")
            
    except Exception as e:
        print(f"❌ pKID 計算失敗: {e}")
        metrics['pKID'] = float('nan')
    
    finally:
        # 清理 patch 暫存檔以節省空間
        print("\n🧹 清理暫存 patches...")
        if os.path.exists(gen_patch_dir):
            shutil.rmtree(gen_patch_dir)
        if os.path.exists(real_patch_dir):
            shutil.rmtree(real_patch_dir)
        print("✅ 清理完成")
    
    return metrics


def save_results(metrics, avg_time=None):
    """儲存並顯示評估結果"""
    print("\n" + "="*60)
    print("🎉 最終評估結果 (Final Results)")
    print("="*60)
    
    # 顯示結果
    if avg_time is not None:
        print(f"⏱️  Average Inference Time: {avg_time:.4f} sec/image")
    
    print(f"\n📊 Full Image Metrics:")
    print(f"   FID     : {metrics.get('FID', float('nan')):.4f}")
    print(f"   KID     : {metrics.get('KID', float('nan')):.6f}")
    
    print(f"\n📊 Patch-based Metrics:")
    print(f"   pFID    : {metrics.get('pFID', float('nan')):.4f}")
    print(f"   pKID    : {metrics.get('pKID', float('nan')):.6f}")
    
    print("="*60)
    
    # 儲存到 JSON
    results_dict = {
        'metrics': metrics,
        'settings': {
            'height': HEIGHT,
            'width': WIDTH,
            'input_size': INPUT_SIZE,
            'patch_size': PATCH_SIZE,
            'total_patches': TOTAL_PATCHES,
            'cleanfid_mode': CLEANFID_MODE,
            'seed': SEED
        }
    }
    
    if avg_time is not None:
        results_dict['avg_inference_time'] = avg_time
    
    results_file = os.path.join(OUTPUT_PATH, 'evaluation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 結果已儲存至: {results_file}")
    
    # 也儲存成易讀的文字檔
    txt_file = os.path.join(OUTPUT_PATH, 'evaluation_results.txt')
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        
        if avg_time is not None:
            f.write(f"Average Inference Time: {avg_time:.4f} sec/image\n\n")
        
        f.write("Full Image Metrics:\n")
        f.write(f"  FID  : {metrics.get('FID', float('nan')):.4f}\n")
        f.write(f"  KID  : {metrics.get('KID', float('nan')):.6f}\n\n")
        
        f.write("Patch-based Metrics:\n")
        f.write(f"  pFID : {metrics.get('pFID', float('nan')):.4f}\n")
        f.write(f"  pKID : {metrics.get('pKID', float('nan')):.6f}\n")
    
    print(f"💾 結果已儲存至: {txt_file}")


# ==================== 主程式 ====================
def main():
    """主程式入口"""
    print("\n" + "="*60)
    print("🚀 圖像生成模型評估系統 (使用 clean-fid)")
    print("="*60 + "\n")
    
    # 設定隨機種子
    set_seed(SEED)
    print(f"🎲 Random seed: {SEED}")
    
    # 建立輸出資料夾
    gen_dir = os.path.join(OUTPUT_PATH, 'generated_images')
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    
    avg_time = None
    
    # ==========================================
    # 第一階段：批量生成圖片
    # ==========================================
    if not SKIP_GEN:
        print("\n" + "="*60)
        print("🎨 階段 1: 圖像生成")
        print("="*60)
        
        # 載入模型
        pipe = load_pipeline()
        
        # 載入 captions
        captions_data = load_captions()
        
        # 生成圖像
        avg_time = generate_images(pipe, captions_data, gen_dir)
        
        # 清理 GPU 記憶體
        del pipe
        torch.cuda.empty_cache()
        print("\n✅ 圖像生成階段完成")
        
    else:
        print("\n⏩ 跳過生成階段，使用現有圖片進行評估")
    
    # ==========================================
    # 第二階段：計算評估指標
    # ==========================================
    print("\n" + "="*60)
    print("📊 階段 2: 評估指標計算")
    print("="*60)
    
    try:
        metrics = calculate_metrics_with_cleanfid(gen_dir, HR_PATH)
    except Exception as e:
        print(f"\n❌ 評估失敗: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ==========================================
    # 第三階段：儲存結果
    # ==========================================
    print("\n" + "="*60)
    print("💾 階段 3: 儲存結果")
    print("="*60)
    
    save_results(metrics, avg_time)
    
    print("\n✅ 所有任務完成!")


if __name__ == '__main__':
    main()