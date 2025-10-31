# AutoShot-Usage
This repo is use for extracting keyframes from video using SOTA shot detection model AutoShot with pHash for deduplicate frames that have too similar in perception. 
Also this repo can handle all video format, using CUDA for acceleration (50% faster than multithread CPU). Automatic fallback to pyav if GPU does not support av1  
