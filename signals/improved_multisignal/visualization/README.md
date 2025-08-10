# PAUT Neural Network Visualization Suite

Professional animations explaining PAUT data processing and neural network architectures using realistic data.

## Files Overview

### Core Files
- **`paut_data_generator.py`** - Generates realistic PAUT data with defects
- **`paut_3d_visualization.py`** - Main PAUT structure and NN pipeline animations  
- **`signal_processing_animation.py`** - Detailed signal processing and position prediction
- **`run_visualizations.py`** - Main runner script for all animations

## Available Animations

### 1. PAUT 3D Data Structure
- **Scene**: `PAUT3DStructure`
- **Shows**: 3D coordinate system, data points, defect highlighting
- **Purpose**: Explain PAUT data organization

### 2. Signal Sequence Extraction  
- **Scene**: `SignalSequenceExtraction`
- **Shows**: How sequences are extracted from 3D data for NN processing
- **Purpose**: Demonstrate data preprocessing for neural networks

### 3. Neural Network Processing Pipeline
- **Scene**: `NeuralNetworkProcessing` 
- **Shows**: Complete workflow from input signals to final predictions
- **Purpose**: End-to-end system overview

### 4. Detailed Architecture
- **Scene**: `DetailedArchitecture`
- **Shows**: Internal structure of both detection and localization models
- **Purpose**: Technical architecture explanation

### 5. Real Signal Analysis
- **Scene**: `RealSignalProcessing`
- **Shows**: CNN feature extraction, transformer attention on real signals
- **Purpose**: Detailed signal processing explanation

### 6. Position Prediction Process
- **Scene**: `PositionPredictionVisualization`
- **Shows**: Dual-head position prediction, IoU calculation
- **Purpose**: Localization model explanation

## Usage

### Prerequisites
```bash
pip install manim numpy scipy matplotlib
```

### Generate All Animations
```bash
cd visualization/
python run_visualizations.py
```

### Generate Specific Animation
```bash
python run_visualizations.py 1  # PAUT 3D Structure
python run_visualizations.py 2  # Sequence Extraction
python run_visualizations.py 3  # NN Processing
python run_visualizations.py 4  # Detailed Architecture
python run_visualizations.py 5  # Signal Analysis
python run_visualizations.py 6  # Position Prediction
```

### Direct Manim Commands
```bash
# High quality (slower)
manim -qh paut_3d_visualization.py PAUT3DStructure

# Medium quality (recommended)
manim -qm paut_3d_visualization.py SignalSequenceExtraction

# Low quality (fast preview)
manim -ql signal_processing_animation.py RealSignalProcessing
```

## Output

- **Location**: `./media/videos/`
- **Format**: MP4 (H.264)
- **Resolution**: 1920x1080 (medium quality)
- **Duration**: 30-60 seconds per animation

## Key Features

### Realistic Data
- **Authentic PAUT signals** with proper physics-based modeling
- **Real defect signatures** (cracks, voids, inclusions)
- **Proper material properties** (steel, sound velocity, sampling)

### Technical Accuracy
- **Actual NN architectures** from your models
- **Real dimensions** and layer configurations
- **Accurate data flow** representation

### Professional Quality
- **Clean animations** suitable for technical presentations
- **Proper labeling** and technical terminology
- **Color coding** for different components and data types

## Customization

### Modify Data Parameters
Edit `paut_data_generator.py`:
```python
generator = PAUTDataGenerator(x_size=50, y_size=30, z_size=320)
generator.add_defect(x_pos=15, y_pos=10, depth_start=8.0, depth_end=12.0)
```

### Adjust Animation Settings
Edit scene files:
```python
# Change colors
signal_curve.set_color(RED)

# Modify timing
self.wait(2)  # Wait 2 seconds
self.play(animation, run_time=3)  # 3-second animation
```

### Export Settings
```python
# In run_visualizations.py
quality_flags = {
    "low": "-ql",      # 480p, fast
    "medium": "-qm",   # 720p, balanced  
    "high": "-qh",     # 1080p, slow
}
```

## Troubleshooting

### Common Issues

**Manim not found**
```bash
pip install manim
# or
conda install -c conda-forge manim
```

**Cairo/pkg-config errors (macOS)**
```bash
brew install cairo pkg-config
```

**Memory issues with large datasets**
- Reduce data size in `paut_data_generator.py`
- Use lower quality settings (`-ql`)

**Slow rendering**
- Use medium quality (`-qm`) for development
- Generate high quality (`-qh`) only for final output

### Performance Tips
- **Preview mode**: Use `-ql` for fast iteration
- **Partial rendering**: Generate specific scenes only
- **Data sampling**: Reduce signal resolution for complex scenes

## Integration with Presentations

### PowerPoint/Keynote
1. Export animations as MP4
2. Insert videos into slides
3. Set to play automatically or on click

### Web Presentations
- Use MP4 files directly in HTML5 video tags
- Convert to GIF for smaller file sizes (lower quality)

### Documentation
- Embed videos in Markdown/HTML documentation
- Use as supplementary material for technical papers
