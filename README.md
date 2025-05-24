# PyTorch Frame - Tabular Deep Learning Examples

This project demonstrates comprehensive PyTorch Frame capabilities for tabular deep learning with diverse data types, multiple tasks, and performance optimization across CPU and GPU devices.

## 🚀 Quick Start with Dev Container

### Prerequisites

1. **Docker**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. **VS Code**: Install [Visual Studio Code](https://code.visualstudio.com/)
3. **Dev Containers Extension**: Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Running the Project

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd pytorch_frame
   ```

2. **Open in VS Code**:
   ```bash
   code .
   ```

3. **Reopen in Container**:
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Dev Containers: Reopen in Container"
   - Select the option and wait for the container to build

4. **Run any example**:
   ```bash
   python pytorch_frame_quick_demo.py
   python pytorch_frame_diverse_example.py
   python pytorch_frame_enhanced_examples.py
   ```

### What the Dev Container Includes

- **Python 3.12** with all dependencies from `requirements.txt`
- **PyTorch with CUDA 12.8 support** for GPU acceleration
- **Pre-installed VS Code extensions**:
  - Python development tools
  - Jupyter notebook support
  - Code formatting (Black)
  - Linting (Pylint)
  - GitHub Copilot
- **Port forwarding** for Jupyter (8888) and TensorBoard (6006)
- **Zsh with Oh My Zsh** for better terminal experience

## 📁 Project Structure

```
pytorch_frame/
├── .devcontainer/
│   └── devcontainer.json              # Dev container configuration
├── pytorch_frame_quick_demo.py        # Quick multi-task demo (3 epochs)
├── pytorch_frame_diverse_example.py   # Comprehensive multi-task example
├── pytorch_frame_enhanced_examples.py # Advanced examples with specialized models
├── requirements.txt                   # Python dependencies with CUDA support
├── data/                              # Auto-downloaded datasets
└── README.md                          # This file
```

## 🎯 Example Scripts Overview

### 1. `pytorch_frame_quick_demo.py`
- **Purpose**: Fast demonstration of multi-task learning
- **Features**: 4 different task types (categorical, binary, regression, multi-class)
- **Training**: 3 epochs for quick testing
- **Model**: Simple shared encoder with task-specific heads

### 2. `pytorch_frame_diverse_example.py`
- **Purpose**: Comprehensive multi-task learning with transformer architecture
- **Features**: Advanced TabTransformer layers with attention mechanisms
- **Training**: 3 epochs with detailed evaluation metrics
- **Model**: Transformer-based encoder with multiple output heads

### 3. `pytorch_frame_enhanced_examples.py`
- **Purpose**: Specialized models for different task types
- **Features**: 
  - Binary classification models (3 tasks)
  - Regression models (3 tasks)
  - Multi-class classification models (3 tasks)
- **Training**: Task-specific architectures with optimized hyperparameters
- **Models**: Specialized architectures for each problem type

## ⚡ Performance Features

### CPU vs GPU Benchmarking
All scripts automatically run on both CPU and GPU (if available) and provide performance comparisons:

```
Performance Summary for pytorch_frame_quick_demo.py:
CPU execution time: 1.86 seconds
GPU execution time: 1.40 seconds
GPU speedup: 1.33x faster than CPU
```

### Execution Timing
Each script displays:
- Program name being executed
- Total execution time for each device
- Speedup ratio when GPU is available

## 🛠 Manual Setup (Alternative)

If you prefer not to use dev containers:

1. **Install Python 3.9+**
2. **Install PyTorch with CUDA support**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
3. **Install other dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run any example**:
   ```bash
   python pytorch_frame_quick_demo.py
   ```

## 📊 What the Examples Demonstrate

### Multi-Task Learning Capabilities
1. **Simultaneous training** on multiple target types:
   - Binary classification (age groups, income levels)
   - Regression (education years, age prediction)
   - Multi-class classification (work sectors, education tiers)

2. **Shared representations** across related tasks for improved efficiency

3. **Task-specific architectures** optimized for different problem types

### Advanced Features
- **Feature encoding** for numerical and categorical data
- **Transformer-based convolutions** for tabular data
- **Attention mechanisms** for improved feature learning
- **Comprehensive evaluation metrics** (Accuracy, AUC, F1, MAE, RMSE)
- **Learning rate scheduling** with ReduceLROnPlateau

## 🐛 Troubleshooting

### Container Issues
- **Container won't start**: Ensure Docker is running
- **Build errors**: Try rebuilding the container: `Ctrl+Shift+P` → "Dev Containers: Rebuild Container"
- **Permission issues**: Check that Docker has proper permissions

### GPU Issues
- **CUDA not available**: Check if NVIDIA drivers are installed and GPU is detected
- **Out of memory**: Reduce batch sizes in the scripts or use smaller models
- **Version conflicts**: Ensure CUDA version matches PyTorch installation

### Python Issues
- **Import errors**: The dev container should automatically install all requirements
- **Module not found**: Try rebuilding the container or manually installing missing packages

### Data Issues
- **Dataset download fails**: Ensure you have internet connection. The Yandex dataset will be downloaded automatically to `./data/adult/`
- **Slow data loading**: First run may be slower due to dataset download and preprocessing

## 🔧 Customization

### Modifying Training Parameters
Each script can be customized by editing:
- `num_epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `lr`: Learning rate
- `sample_size`: Dataset sample size for faster testing

### Adding New Tasks
To add new tasks:
1. Modify the target configuration dictionaries
2. Update the data preprocessing functions
3. Add appropriate loss functions and evaluation metrics

### GPU Memory Optimization
For large models or limited GPU memory:
- Reduce `batch_size` in DataLoader
- Decrease `channels` in model architectures
- Use gradient accumulation for effective larger batch sizes

## 🎯 Performance Optimization

### GPU Acceleration
- **Automatic device detection**: Scripts automatically use GPU when available
- **Memory management**: CUDA cache clearing between runs for fair comparisons
- **Optimized data loading**: Efficient tensor operations and batch processing

### Model Efficiency
- **Shared encoders**: Reduce parameter count through shared representations
- **Task-specific heads**: Lightweight output layers for different tasks
- **Attention mechanisms**: Improved feature learning without excessive parameters

## 📚 Learn More

- [PyTorch Frame Documentation](https://pytorch-frame.readthedocs.io/)
- [PyTorch Frame GitHub](https://github.com/pyg-team/pytorch-frame)
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
- [Dev Containers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- [TabTransformer Paper](https://arxiv.org/abs/2012.06678) 