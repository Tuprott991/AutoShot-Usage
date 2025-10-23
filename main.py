import sys
import av
import pickle
import numpy as np
from utils import mAP_f1_p_fix_r
from utils import evaluate_scenes, predictions_to_scenes
from utils import get_frames, get_batches, scenes2zero_one_representation, visualize_predictions
import os
import pickle
import cv2
import numpy as np
import torch
import shutil
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

def get_frames(video_path):
    frames = []
    container = av.open(video_path)

    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")  # PyAV trả ndarray RGB
        img_resized = cv2.resize(img, (48, 27))  # resize về 48x27 như code cũ
        frames.append(img_resized)

    container.close()
    return np.array(frames)

def get_frames_parallel(video_path, num_workers=4, chunk_size=500, use_gpu=False):
    """Memory-efficient frame extraction with streaming decode and parallel resizing"""
    logger = logging.getLogger(__name__)
    
    # Try GPU-accelerated decoding if requested and CUDA is available
    if use_gpu and torch.cuda.is_available():
        try:            
            logger.info("Attempting GPU-accelerated decoding with CUDA...")
            #print("🔧 Attempting GPU-accelerated decoding with CUDA...", flush=True)
            frames = get_frames_gpu(video_path, num_workers, chunk_size)
            if frames is not None:
                logger.info(f"GPU decoding successful! Processed {len(frames)} frames")
                #print(f"✓ GPU decoding successful! Processed {len(frames)} frames", flush=True)
                return frames
            else:
                logger.warning("GPU decoding failed, falling back to CPU...")
                #print("⚠️ GPU decoding failed, falling back to CPU...", flush=True)
        except Exception as e:
            logger.warning(f"GPU decoding error: {e}, falling back to CPU...")
            #print(f"⚠️ GPU decoding error: {e}, falling back to CPU...", flush=True)
    
    # CPU decoding (fallback or default)
    container = av.open(video_path)
    
    logger.info("Decoding and resizing video frames (CPU)...")
    #print("🔧 Decoding and resizing video frames (CPU)...", flush=True)
    sys.stdout.flush()  # Force flush for Kaggle
    
    frames = []
    chunk = []
    frame_idx = 0
    
    def resize_frame(frame):
        return cv2.resize(frame, (48, 27))
    
    # Process frames in chunks to avoid RAM overflow
    for frame in container.decode(video=0):
        chunk.append(frame.to_ndarray(format="rgb24"))
        frame_idx += 1
        
        # When chunk is full, process it in parallel
        if len(chunk) >= chunk_size:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                resized_chunk = list(executor.map(resize_frame, chunk))
            frames.extend(resized_chunk)
            
            # Update every 5000 frames
            if len(frames) % 5000 == 0:
                logger.info(f"Processed {len(frames)} frames so far...")
                #print(f"📊 Processed {len(frames)} frames so far...", flush=True)
                sys.stdout.flush()
            
            chunk = []  # Clear chunk to free memory
    
    # Process remaining frames
    if chunk:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            resized_chunk = list(executor.map(resize_frame, chunk))
        frames.extend(resized_chunk)
    
    container.close()
    
    total_frames = len(frames)
    logger.info(f"✓ Frame decoding completed: {total_frames} frames")
    #print(f"✓ Frame decoding completed: {total_frames} frames", flush=True)
    sys.stdout.flush()  # Force flush for Kaggle
    
    return np.array(frames)

def get_frames_gpu(video_path, num_workers=4, chunk_size=500):
    """GPU-accelerated frame extraction using CUDA with PyAV"""
    logger = logging.getLogger(__name__)
    
    try:
        # Try to open with hardware acceleration
        container = av.open(video_path, options={'hwaccel': 'nvdec'})
        
        frames = []
        chunk = []
        
        def resize_frame(frame):
            return cv2.resize(frame, (48, 27))
        
        # Decode with GPU
        for frame in container.decode(video=0):
            chunk.append(frame.to_ndarray(format="rgb24"))
            
            if len(chunk) >= chunk_size:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    resized_chunk = list(executor.map(resize_frame, chunk))
                frames.extend(resized_chunk)
                
                # Update every 5000 frames
                if len(frames) % 5000 == 0:
                    logger.info(f"Processed {len(frames)} frames so far (GPU)...")
                    #print(f"📊 Processed {len(frames)} frames so far (GPU)...", flush=True)
                
                chunk = []
        
        if chunk:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                resized_chunk = list(executor.map(resize_frame, chunk))
            frames.extend(resized_chunk)
        
        container.close()
        logger.info(f"✓ GPU decoding completed: {len(frames)} frames")
        return np.array(frames)
        
    except Exception as e:
        logger.debug(f"GPU decoding failed: {e}")
        return None



def setup_logging(log_level):
    """Setup logging configuration for Kaggle compatibility"""
    # Force flush to ensure logs appear immediately on Kaggle
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True,  # Override any existing logging config
        handlers=[
            logging.StreamHandler(sys.stdout)  # Use stdout for Kaggle
        ]
    )
    
    # Ensure immediate flushing
    logger = logging.getLogger(__name__)
    for handler in logger.handlers:
        handler.flush()
    
    return logger

from supernet_flattransf_3_8_8_8_13_12_0_16_60 import TransNetV2Supernet

def load_model(model_path, device):
    """Load the TransNetV2 model with pretrained weights"""
    logger = logging.getLogger(__name__)
    supernet_best_f1 = TransNetV2Supernet().eval()
    
    if os.path.exists(model_path):
        logger.info(f'Loading pretrained model from {model_path}')
        model_dict = supernet_best_f1.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict['net'].items() if k in model_dict}
        logger.info(f"Current model has {len(model_dict)} params, updating {len(pretrained_dict)} params")
        model_dict.update(pretrained_dict)
        supernet_best_f1.load_state_dict(model_dict)
    else:
        raise Exception(f"Error: Cannot find pretrained model at {model_path}")
    
    if device == "cuda":
        supernet_best_f1 = supernet_best_f1.cuda(0)
    supernet_best_f1.eval()
    
    logger.info("Model loaded successfully")
    return supernet_best_f1

class KeyframeExtractor:
    def __init__(self, model, device, logits_start, logits_end, threshold, num_workers=4, use_gpu_decode=False, chunk_size=500):
        self.model = model
        self.device = device
        self.logits_start = logits_start
        self.logits_end = logits_end
        self.threshold = threshold
        self.num_workers = num_workers
        self.use_gpu_decode = use_gpu_decode
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
    
    def predict(self, batch):
        batch = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...]) * 1.0
        batch = batch.to(self.device)
        one_hot = self.model(batch)
        if isinstance(one_hot, tuple):
            one_hot = one_hot[0]
        return torch.sigmoid(one_hot[0])
    
    def get_key_frames(self, video_path):
        frames = get_frames_parallel(video_path, self.num_workers, self.chunk_size, self.use_gpu_decode)
        total_frames = len(frames)
        
        self.logger.info(f"Analyzing {total_frames} frames for shot detection...")
        #print(f"🔍 Analyzing {total_frames} frames for shot detection...", flush=True)
        
        predictions = []
        batches = list(get_batches(frames))
        
        # Process batches
        for batch_idx, batch in enumerate(batches, 1):
            logits = self.predict(batch)
            logits = logits.detach().cpu().numpy()
            predictions.append(logits[self.logits_start:self.logits_end])
            
            if batch_idx % 100 == 0:
                self.logger.info(f"Processed {batch_idx}/{len(batches)} batches")
                #print(f"📊 Processed {batch_idx}/{len(batches)} batches", flush=True)

        self.logger.info(f"✓ Shot detection completed")
        #print(f"✓ Shot detection completed", flush=True)
        
        predictions = np.concatenate(predictions, 0)[:total_frames]
        mask = (predictions > self.threshold).astype(np.uint8)
        indices = np.where(mask)[0]
        
        self.logger.debug(f"Found {len(indices)} initial keyframe candidates")

        # Input the last frame into the indices if not already
        if len(indices) > 0 and indices[-1] != total_frames - 1:
            indices = np.append(indices, total_frames - 1)
        elif len(indices) == 0:
            indices = np.array([total_frames - 1])

        def get_num_keyframes(num_frames):
            """Determine number of keyframes based on shot length"""
            if num_frames <= 30:
                return 1
            elif num_frames <= 100:
                return 2
            elif num_frames <= 300:
                return 3
            return 4

        final_indices = []

        last_index = 0
        for index in indices:
            if index == 0: 
                continue
            else:
                shot_length = index - last_index
                num_keyframes = get_num_keyframes(shot_length)
                sep = shot_length / num_keyframes
                for i in range(num_keyframes):
                    new_index = int(last_index + sep * i)  # rounds down
                    if new_index not in final_indices:
                        final_indices.append(new_index)
                last_index = index

        self.logger.info(f"Extracted {len(final_indices)} keyframes from {total_frames} total frames")
        return final_indices

def get_videos(folder_path: str) -> list:
    """Get list of video files from a folder"""
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist!")
        return []
    
    files_path = []
    for f in os.listdir(folder_path):
        file_path = os.path.join(folder_path, f)
        if os.path.isfile(file_path):
            files_path.append(file_path)
    return files_path

def save_frames(res, folder, video_path, save_path, num_workers=4):
    """Save extracted frames as WebP images using multi-threading"""
    logger = logging.getLogger(__name__)
    target_folder = os.path.join(save_path, folder)
    os.makedirs(target_folder, exist_ok=True)

    logger.info(f"Saving {len(res)} keyframes...")
    
    # Pre-load all frames we need
    container = av.open(video_path)
    res_set = set(res)
    frames_to_save = {}
    
    for frame_idx, frame in enumerate(container.decode(video=0)):
        if frame_idx in res_set:
            frames_to_save[frame_idx] = frame.to_ndarray(format="bgr24")
        if frame_idx > max(res):  # Stop early if we've passed all needed frames
            break
    
    container.close()
    
    # Save frames in parallel
    saved_count = 0
    save_lock = Lock()
    
    def save_single_frame(frame_idx, img):
        filepath = os.path.join(target_folder, f"{frame_idx}.webp")
        cv2.imwrite(filepath, img, [cv2.IMWRITE_WEBP_QUALITY, 90])
        with save_lock:
            nonlocal saved_count
            saved_count += 1
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(save_single_frame, idx, img) for idx, img in frames_to_save.items()]
        
        for future in as_completed(futures):
            future.result()  # Wait for completion and raise any exceptions
    
    logger.info(f"✓ Successfully saved {saved_count} frames to {target_folder}")


def parse_args():
    parser = argparse.ArgumentParser(description='Extract keyframes from videos using TransNetV2')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the pretrained model checkpoint')
    parser.add_argument('--logits-start', type=int, default=25,
                        help='Start index for logits slicing (default: 25)')
    parser.add_argument('--logits-end', type=int, default=75,
                        help='End index for logits slicing (default: 75)')
    parser.add_argument('--video-folder', type=str, required=True,
                        help='Path to the folder containing videos')
    parser.add_argument('--save-path', type=str, required=True,
                        help='Path to save extracted keyframes')
    parser.add_argument('--threshold', type=float, default=0.85,
                        help='Threshold for keyframe detection (default: 0.85)')
    parser.add_argument('--skip-videos', type=str, nargs='*', default=[],
                        help='List of video names to skip (without extension)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for inference (default: auto)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (default: INFO)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker threads for frame operations (default: 4)')
    parser.add_argument('--chunk-size', type=int, default=500,
                    help='Number of frames to process at once (default: 500)')
    parser.add_argument('--use-gpu-decode', action='store_true',
                        help='Use GPU-accelerated decoding if available')
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Determine device
    if args.device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    logger.info(f"Worker threads: {args.num_workers}")
    logger.info(f"Logits range: [{args.logits_start}:{args.logits_end}]")
    logger.info(f"Detection threshold: {args.threshold}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Create keyframe extractor
    extractor = KeyframeExtractor(
        model=model,
        device=device,
        logits_start=args.logits_start,
        logits_end=args.logits_end,
        threshold=args.threshold,
        num_workers=args.num_workers,
        use_gpu_decode=args.use_gpu_decode
    )
    
    # Create output directory
    os.makedirs(args.save_path, exist_ok=True)
    logger.info(f"Output directory: {args.save_path}")
    
    # Get videos
    logger.info(f"Looking for videos in: {args.video_folder}")
    videos = get_videos(args.video_folder)
    
    if not videos:
        logger.warning("No videos found!")
        if os.path.exists(args.video_folder):
            logger.info(f"Available files: {os.listdir(args.video_folder)}")
        else:
            logger.error(f"Video folder {args.video_folder} does not exist!")
        return
    
    logger.info(f"Found {len(videos)} videos to process")
    logger.info("=" * 80)
    
    # Process videos
    for video_idx, video in enumerate(videos, 1):
        video_name = os.path.basename(video).split(".")[0]
        
        if video_name in args.skip_videos:
            logger.info(f"[{video_idx}/{len(videos)}] Skipping video: {video_name}")
            continue
        
        logger.info(f"\n[{video_idx}/{len(videos)}] Processing: {video_name}")
        logger.info("-" * 80)
        start = time.time()
        
        try:
            res = extractor.get_key_frames(video)
            
            extraction_time = time.time() - start
            logger.info(f"✓ Keyframe extraction completed in {extraction_time:.2f} seconds")
            
            save_start = time.time()
            save_frames(res, video_name, video, args.save_path, args.num_workers)
            save_time = time.time() - save_start
            
            logger.info(f"✓ Frames saved in {save_time:.2f} seconds")
            logger.info(f"✓ Total time for {video_name}: {(extraction_time + save_time):.2f} seconds")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"✗ Error processing video {video_name}: {str(e)}", exc_info=True)
            logger.info("=" * 80)
            continue
    
    logger.info(f"\n✓ All videos processed successfully!")

if __name__ == "__main__":
    main()