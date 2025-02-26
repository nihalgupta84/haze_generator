# Haze Dataset Generator

A Python tool to generate synthetic hazy images from various optical flow datasets using depth estimation.

## Features

- Supports multiple datasets:
  - KITTI 2015
  - Sintel
  - FlyingChairs
  - FlyingThings3D
  - HD1K
  - ChairsSDHom
  - FlyingChairsOcc
  - FlyingThings3D subset
- Uses MonoDepth2 for depth estimation
- Includes checkpoint system to resume interrupted processing
- Provides preview functionality before full dataset processing

## Prerequisites

```bash
# Required packages
pip install torch torchvision opencv-python pillow matplotlib tqdm numpy
```

You'll also need:
1. MonoDepth2 pretrained models
   - Download from: [MonoDepth2 Models](https://drive.google.com/drive/folders/1V8c4nrYP_19HxCX5K7Sz89dVb5Nc1cJy)
   - Place in `./models/mono+stereo_1024x320/`

2. One or more of the supported datasets:
   - [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
   - [Sintel](http://sintel.is.tue.mpg.de/)
   - [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html)
   - [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
   - [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/)
   - [ChairsSDHom](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html)
   - [FlyingChairsOcc](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html)

3. Create a symbolic link to put all optical flow datasets into a single folder named `./datasets`:
   ```bash
   ln -s /path/to/your/datasets ./datasets
   ```

## Usage

1. **Preview Samples**
```bash
python haze.py --dataset kitti15 --src_dir path/to/dataset --dst_dir path/to/output --num_samples 2
```

2. **Generate Full Hazy Dataset**
```bash
python haze.py --dataset kitti15 --src_dir path/to/dataset --dst_dir path/to/output --beta 1 --k 0.5
```

Arguments:
- `--dataset`: Dataset type (required)
- `--src_dir`: Source dataset directory (required)
- `--dst_dir`: Output directory for hazy images (required)
- `--beta`: Haze scattering coefficient (default: 1)
- `--k`: Atmospheric light contribution (default: 0.5)
- `--num_samples`: Number of preview samples (default: 1)

## Project Structure

```
haze-dataset-generator/
├── models/
│   └── mono+stereo_1024x320/
│       ├── encoder.pth
│       └── depth.pth
├── monodepth/
│   └── ... (MonoDepth2 files)
├── haze.py
└── README.md
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [MonoDepth2](https://github.com/nianticlabs/monodepth2) for depth estimation
- Various optical flow dataset providers
