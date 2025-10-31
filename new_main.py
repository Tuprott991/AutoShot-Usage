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
import subprocess
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock  
from PIL import Image
import imagehash

# Add after other imports
def calculate_image_hash(frame):
    """Calculate perceptual hash of a frame"""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.phash(image)

def hamming_distance(hash1, hash2): 
    """Calculate Hamming distance between two hashes"""
    return hash1 - hash2

def format_csv_file(frame_idx, n, fps):
    """Format keyframe data for CSV"""
    pts_time = round(frame_idx / fps, 2)
    new_dict = {
        'n': n,
        'pts_time': pts_time,
        'fps': fps,
        'frame_idx': frame_idx
    }
    return new_dict

def save_to_csv(csv_file, output_folder, file_name):
    """Save keyframe data to CSV file"""
    # Ensure file_name has .csv extension
    if not file_name.endswith('.csv'):
        file_name += '.csv'
    
    # Ensure output_folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Full path to file
    file_path = os.path.join(output_folder, file_name)
    
    # Open csv file and write content
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['n', 'pts_time', 'fps', 'frame_idx'])
        writer.writeheader()
        writer.writerows(csv_file)
    
    logging.getLogger(__name__).info(f"CSV file saved at: {file_path}")

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

def get_video_codec(video_path):
    """Detect video codec using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        codec = result.stdout.strip().lower()
        return codec
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not detect codec: {e}, defaulting to PyAV")
        return "unknown"

def decode_with_ffmpeg_cuda(video_path, logger, resize_for_detection=True, start_frame=0, end_frame=None):
    """Decode video using FFmpeg CLI with CUDA acceleration - memory efficient"""
    logger.info("Using FFmpeg CLI with CUDA acceleration")
    
    # Get video properties first
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_frames',
        '-of', 'csv=p=0',
        video_path
    ]
    
    try:
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        parts = probe_result.stdout.strip().split(',')
        width, height = int(parts[0]), int(parts[1])
        
        if resize_for_detection:
            # For detection: decode at reduced resolution to save memory
            target_width, target_height = 48, 27
            logger.info(f"Decoding at {target_width}x{target_height} for detection (saves memory)")
        else:
            # For keyframe extraction: decode at original resolution
            target_width, target_height = width, height
            logger.info(f"Decoding at original resolution {width}x{height}")
        
        # FFmpeg command with optional scaling
        cmd = [
            'ffmpeg',
            '-hwaccel', 'cuda',
            '-hwaccel_output_format', 'cuda',
        ]
        
        # Add frame range selection BEFORE input
        if end_frame is not None:
            num_frames = end_frame - start_frame
            cmd.extend(['-vframes', str(num_frames)])
        
        cmd.extend(['-i', video_path])

        # Build filter chain with trim
        filters = []
        
        # Trim to frame range
        if start_frame > 0 or end_frame is not None:
            trim_filter = f'trim=start_frame={start_frame}'
            if end_frame is not None:
                trim_filter += f':end_frame={end_frame}'
            filters.append(trim_filter)
            filters.append('setpts=PTS-STARTPTS')  # Reset timestamps

        if resize_for_detection:
            # Correct filter chain: scale on GPU
            filters.append(f'scale_cuda={target_width}:{target_height}')

        # Download to CPU, then convert format
        filters.extend(['hwdownload', 'format=nv12', 'format=rgb24'])
        
        cmd.extend(['-vf', ','.join(filters)])
        
        cmd.extend([
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-vsync', 'passthrough',
            'pipe:1'
        ])
        
        # Start FFmpeg process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        frames = []
        frame_size = target_width * target_height * 3  # RGB24
        
        while True:
            raw_frame = process.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                break
            
            # Convert raw bytes to numpy array
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((target_height, target_width, 3))
            frames.append(frame)
        
        process.wait()
        
        if process.returncode != 0:
            stderr = process.stderr.read().decode()
            logger.warning(f"FFmpeg returned error: {stderr}")
            return None
        
        return frames
        
    except Exception as e:
        logger.error(f"FFmpeg decoding failed: {e}")
        return None

def extract_specific_frames_ffmpeg_cuda(video_path, frame_indices, logger):
    """Extract specific frames using FFmpeg with CUDA - optimized single-pass extraction"""
    logger.info(f"Extracting {len(frame_indices)} specific frames with FFmpeg CUDA (single pass)")
    
    # Get video properties
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0',
        video_path
    ]
    
    try:
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        parts = probe_result.stdout.strip().split(',')
        width, height = int(parts[0]), int(parts[1])
        
        # Build select filter for all frames at once: "eq(n,0)+eq(n,5)+eq(n,10)"
        frame_indices_sorted = sorted(frame_indices)
        select_expr = '+'.join([f'eq(n\\,{idx})' for idx in frame_indices_sorted])
        
        # Single FFmpeg command to extract all keyframes
        cmd = [
            'ffmpeg',
            '-hwaccel', 'cuda',
            '-i', video_path,
            '-vf', f'select={select_expr}',
            '-vsync', '0',  # Pass through frame timing
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            'pipe:1'
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        frames = []
        frame_size = width * height * 3
        
        for _ in range(len(frame_indices_sorted)):
            raw_frame = process.stdout.read(frame_size)
            if len(raw_frame) == frame_size:
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
                frames.append(frame)
            else:
                logger.warning(f"Failed to read frame, got {len(raw_frame)} bytes instead of {frame_size}")
                frames.append(None)
        
        process.wait()
        
        if process.returncode != 0:
            stderr = process.stderr.read().decode()
            logger.warning(f"FFmpeg returned error: {stderr}")
            return None
        
        # Reorder frames to match original indices order
        frame_map = {idx: frame for idx, frame in zip(frame_indices_sorted, frames)}
        ordered_frames = [frame_map[idx] for idx in frame_indices]
        
        return ordered_frames
        
    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
        return None

def get_video_fps_and_duration(video_path):
    """Get video FPS and total duration using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,duration,nb_frames',
            '-of', 'csv=p=0',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = result.stdout.strip().split(',')
        
        # Parse FPS (format: "30/1" or "30000/1001")
        fps_str = parts[0]
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den
        else:
            fps = float(fps_str)
        
        # Parse duration (seconds)
        duration = float(parts[1]) if len(parts) > 1 else None
        
        # Parse total frames
        total_frames = int(parts[2]) if len(parts) > 2 else None
        
        return fps, duration, total_frames
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not get video info: {e}")
        return None, None, None

def calculate_frame_range(video_path, skip_start_seconds, skip_end_seconds, logger):
    """Calculate start and end frame indices based on time skips"""
    fps, duration, total_frames = get_video_fps_and_duration(video_path)
    
    if fps is None or total_frames is None:
        logger.warning("Could not determine video FPS/frames, processing entire video")
        return 0, None
    
    logger.info(f"Video info: {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s duration")
    
    # Calculate frame indices
    start_frame = int(skip_start_seconds * fps)
    end_frame = total_frames - int(skip_end_seconds * fps)
    
    # Validate
    if start_frame < 0:
        start_frame = 0
    if end_frame > total_frames:
        end_frame = total_frames
    if start_frame >= end_frame:
        logger.error(f"Invalid time range: start_frame={start_frame} >= end_frame={end_frame}")
        return 0, None
    
    logger.info(f"Processing frames [{start_frame} → {end_frame}] (skipping {skip_start_seconds}s start, {skip_end_seconds}s end)")
    
    return start_frame, end_frame

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
    def __init__(self, model, device, logits_start, logits_end, threshold, 
                 num_workers=4, use_gpu_decode=False, chunk_size=500, 
                 hamming_threshold=5, batch_size=50, 
                 skip_start_seconds=0.0, skip_end_seconds=0.0):
        self.model = model
        self.device = device
        self.logits_start = logits_start
        self.logits_end = logits_end
        self.threshold = threshold
        self.num_workers = num_workers
        self.use_gpu_decode = use_gpu_decode
        self.chunk_size = chunk_size
        self.hamming_threshold = hamming_threshold
        self.batch_size = batch_size
        self.skip_start_seconds = skip_start_seconds
        self.skip_end_seconds = skip_end_seconds
        self.logger = logging.getLogger(__name__)
    
    def predict(self, batch):
        batch = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...]) * 1.0
        batch = batch.to(self.device)
        one_hot = self.model(batch)
        if isinstance(one_hot, tuple):
            one_hot = one_hot[0]
        return torch.sigmoid(one_hot[0])
    
    @staticmethod
    def get_num_keyframes(num_frames):
        """Determine number of keyframes based on shot length"""
        if num_frames <= 30:
            return 1
        elif num_frames <= 100:
            return 2
        elif num_frames <= 300:
            return 3
        return 4
    
    @staticmethod
    def get_centered_frame_positions(start_frame, end_frame, keyframe_count):
        """Get evenly distributed frame positions centered within the shot range"""
        num_frames = end_frame - start_frame + 1
        step = num_frames / (keyframe_count + 1)
        positions = [int(start_frame + round(step * (i + 1))) for i in range(keyframe_count)]
        return positions
    
    def filter_duplicates_by_hash(self, keyframe_indices):
        """Filter duplicate frames using perceptual hashing - memory efficient version"""
        # This will be called after we have the actual frames
        return keyframe_indices  # Placeholder, will filter during frame extraction

    def get_keyframes(self, video_path):
        """Memory-efficient keyframe extraction with codec-specific decoding"""
        
        # Calculate frame range based on time skips
        start_frame, end_frame = calculate_frame_range(
            video_path, 
            self.skip_start_seconds, 
            self.skip_end_seconds, 
            self.logger
        )

        # Detect codec
        codec = get_video_codec(video_path)
        self.logger.info(f"Detected codec: {codec}")
        
        # Choose decoding method based on codec
        use_ffmpeg = codec in ['h264', 'h265', 'hevc']
        
        # STEP 1: Decode at low resolution for shot detection (saves memory)
        frames_for_detection = []
        
        if use_ffmpeg and self.use_gpu_decode and torch.cuda.is_available():
            # Use FFmpeg CLI with CUDA for h264/h265 - decode at 48x27
            frames_for_detection = decode_with_ffmpeg_cuda(
                video_path, self.logger, 
                resize_for_detection=True,
                start_frame=start_frame,
                end_frame=end_frame
            )
            
            # Fallback to PyAV if FFmpeg fails
            if frames_for_detection is None:
                self.logger.warning("FFmpeg decoding failed, falling back to PyAV")
                use_ffmpeg = False
        
        # Use PyAV for non-h264/h265 or if FFmpeg failed
        if not use_ffmpeg or frames_for_detection is None:
            self.logger.info("Using PyAV for decoding (low-res for detection)")
            container = av.open(video_path)
            
            frames_for_detection = []
            for frame_idx, frame in enumerate(container.decode(video=0)):
                # Skip frames outside range
                if frame_idx < start_frame:
                    continue
                if end_frame is not None and frame_idx >= end_frame:
                    break

                original = frame.to_ndarray(format="rgb24")
                # Resize immediately to save memory
                resized = cv2.resize(original, (48, 27))
                frames_for_detection.append(resized)
            
            container.close()
        
        total_frames = len(frames_for_detection)
        self.logger.info(f"Decoded {total_frames} frames at 48x27 for detection")
        
        # STEP 2: Detect shot boundaries
        self.logger.info(f"Analyzing {total_frames} frames for shot detection...")
        
        frames_array = np.array(frames_for_detection)
        predictions = []
        batches = list(get_batches(frames_array, batch_size=self.batch_size))
        
        for batch_idx, batch in enumerate(batches, 1):
            logits = self.predict(batch)
            logits = logits.detach().cpu().numpy()
            predictions.append(logits[self.logits_start:self.logits_end])
            
            if batch_idx % 100 == 0:
                self.logger.info(f"Processed {batch_idx}/{len(batches)} batches")

        self.logger.info(f"✓ Shot detection completed")
        
        # Free memory from detection frames
        del frames_for_detection
        del frames_array
        
        predictions = np.concatenate(predictions, 0)[:total_frames]
        mask = (predictions > self.threshold).astype(np.uint8)
        shot_boundaries = np.where(mask)[0].tolist()
        
        # Ensure last frame is a boundary
        if not shot_boundaries or shot_boundaries[-1] != total_frames - 1:
            shot_boundaries.append(total_frames - 1)

        # STEP 3: Calculate keyframe indices (centered within each shot)
        keyframes_indices = []
        last_boundary = 0

        for boundary in shot_boundaries:
            if boundary == 0:
                continue
            else:
                shot_length = boundary - last_boundary
                num_keyframes = self.get_num_keyframes(shot_length)
                
                # Get centered positions within the shot
                positions = self.get_centered_frame_positions(last_boundary, boundary - 1, num_keyframes)
                
                for pos in positions:
                    if pos not in keyframes_indices:
                        keyframes_indices.append(pos)

                last_boundary = boundary

        self.logger.info(f"Identified {len(keyframes_indices)} keyframe candidates")
        
        # STEP 4: Extract only the keyframes at full resolution
        adjusted_indices = [idx + start_frame for idx in keyframes_indices] # Adjust indices back to original video frame numbers
        
        return adjusted_indices, video_path, use_ffmpeg


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

def save_frames(keyframe_indices, video_path, use_ffmpeg, folder, save_path, num_workers=4, hamming_threshold=5, use_gpu_decode=False, save_csv=True):
    """Extract and save only the keyframes - memory efficient"""
    logger = logging.getLogger(__name__)
    target_folder = os.path.join(save_path, folder)
    os.makedirs(target_folder, exist_ok=True)
    
    # Get video FPS for CSV
    fps, _, _ = get_video_fps_and_duration(video_path)
    if fps is None:
        fps = 30.0  # Default fallback
        logger.warning(f"Could not determine FPS, using default: {fps}")
    
    logger.info(f"Extracting and saving {len(keyframe_indices)} keyframes at full resolution...")
    
    # Extract keyframes at full resolution
    keyframes = []
    
    if use_ffmpeg and use_gpu_decode:
        # Use FFmpeg to extract specific frames
        keyframes = extract_specific_frames_ffmpeg_cuda(video_path, keyframe_indices, logger)
        
        if keyframes is None:
            logger.warning("FFmpeg extraction failed, falling back to PyAV")
            use_ffmpeg = False
    
    if not use_ffmpeg or keyframes is None:
        # Use PyAV - seek to specific frames
        logger.info("Extracting keyframes with PyAV")
        container = av.open(video_path)
        
        keyframe_set = set(keyframe_indices)
        keyframes = [None] * len(keyframe_indices)
        idx_map = {idx: i for i, idx in enumerate(keyframe_indices)}
        
        for frame_idx, frame in enumerate(container.decode(video=0)):
            if frame_idx in keyframe_set:
                keyframes[idx_map[frame_idx]] = frame.to_ndarray(format="rgb24")
                
                # Early exit if we got all keyframes
                if all(kf is not None for kf in keyframes):
                    break
        
        container.close()
    
    # Filter duplicates using perceptual hashing
    logger.info("Filtering duplicates with perceptual hashing...")
    filtered_frames = []
    filtered_indices = []
    csv_entries = []
    prev_hash = None
    count = 0
    
    for i, (idx, frame) in enumerate(zip(keyframe_indices, keyframes)):
        if frame is None:
            logger.warning(f"Frame {idx} was not extracted, skipping")
            continue
            
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cur_hash = calculate_image_hash(frame_bgr)
        
        if prev_hash is None or hamming_distance(cur_hash, prev_hash) > hamming_threshold:
            filtered_frames.append(frame_bgr)
            filtered_indices.append(idx)
            
            # Create CSV entry for this keyframe
            if save_csv:
                csv_entry = format_csv_file(frame_idx=idx, n=count, fps=fps)
                csv_entries.append(csv_entry)
                count += 1
            
            prev_hash = cur_hash
    
    logger.info(f"After deduplication: {len(filtered_frames)} unique keyframes")
    
    # Save frames
    saved_count = 0
    save_lock = Lock()
    
    def save_single_frame(idx, frame_bgr):
        filepath = os.path.join(target_folder, f"{idx:03d}.webp")
        cv2.imwrite(filepath, frame_bgr, [cv2.IMWRITE_WEBP_QUALITY, 90])
        with save_lock:
            nonlocal saved_count
            saved_count += 1
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(save_single_frame, idx, frame) 
            for idx, frame in zip(filtered_indices, filtered_frames)
        ]
        for future in as_completed(futures):
            future.result()
    
    logger.info(f"✓ Successfully saved {saved_count} frames to {target_folder}")
    
    # Save CSV file in csv_output folder
    if save_csv and csv_entries:
        csv_output_folder = os.path.join(save_path, "csv_output")
        save_to_csv(csv_entries, csv_output_folder, folder)
        logger.info(f"✓ CSV file saved with {len(csv_entries)} entries")


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
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size for GPU processing (default: 50). Higher values use more VRAM. Recommended: 100-200 for 14GB VRAM')
    parser.add_argument('--use-gpu-decode', action='store_true',
                        help='Use GPU-accelerated decoding if available')
    parser.add_argument('--hamming-threshold', type=int, default=5,
                        help='Hamming distance threshold for duplicate frame filtering (default: 5)')
    parser.add_argument('--skip-start-seconds', type=float, default=0.0,
                        help='Skip X seconds from the start of video (default: 0.0)')
    parser.add_argument('--skip-end-seconds', type=float, default=0.0,
                        help='Skip Y seconds from the end of video (default: 0.0)')
    
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
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Logits range: [{args.logits_start}:{args.logits_end}]")
    logger.info(f"Detection threshold: {args.threshold}")
    logger.info(f"Hamming threshold: {args.hamming_threshold}")
    logger.info(f"Time skip: start={args.skip_start_seconds}s, end={args.skip_end_seconds}s")

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
        use_gpu_decode=args.use_gpu_decode,
        batch_size=args.batch_size,
        hamming_threshold=args.hamming_threshold,
        skip_start_seconds=args.skip_start_seconds,
        skip_end_seconds=args.skip_end_seconds,
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
            keyframe_indices, video_path_result, use_ffmpeg = extractor.get_keyframes(video)
            
            extraction_time = time.time() - start
            logger.info(f"✓ Shot detection completed in {extraction_time:.2f} seconds")
            
            save_start = time.time()

            save_frames(
                keyframe_indices, 
                video_path_result, 
                use_ffmpeg,
                video_name, 
                args.save_path, 
                args.num_workers,
                args.hamming_threshold,
                args.use_gpu_decode,
                save_csv=True
            )
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