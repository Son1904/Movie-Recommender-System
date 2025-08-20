# VS CODE SETUP INSTRUCTIONS
## Movie Recommender System - Development Environment Setup

---

## 🚀 QUICK START GUIDE

### 1. **Project Structure Setup**

Ensure your project has the following structure:
```
Final Project/
├── EDA.ipynb                           # Main analysis notebook
├── PROJECT_PROPOSAL.md                 # Project proposal
├── PROJECT_IMPLEMENTATION_STAGES.md    # Implementation roadmap
├── SETUP_INSTRUCTIONS.md              # This file
├── movie/
│   └── ml-latest-small/                # MovieLens dataset
│       ├── ratings.csv
│       ├── movies.csv
│       ├── tags.csv
│       ├── links.csv
│       └── README.txt
└── data/
    └── processed/                      # Generated processed data
        ├── train_data.csv
        ├── val_data.csv
        ├── test_data.csv
        ├── user_features.csv
        ├── movie_features.csv
        └── user_genre_preferences.csv
```

### 2. **Download MovieLens Dataset**

1. **Download the dataset:**
   - Go to: https://grouplens.org/datasets/movielens/
   - Download "ml-latest-small.zip" (size: ~1MB)
   
2. **Extract the dataset:**
   - Extract to `./movie/ml-latest-small/` directory
   - Ensure the CSV files are directly in this folder

3. **Verify files:**
   ```
   ✅ ./movie/ml-latest-small/ratings.csv    (~100K ratings)
   ✅ ./movie/ml-latest-small/movies.csv     (~9K movies)
   ✅ ./movie/ml-latest-small/tags.csv       (~3K tags)
   ✅ ./movie/ml-latest-small/links.csv      (IMDB/TMDB links)
   ```

### 3. **Python Environment Setup**

#### Option A: Using Conda (Recommended)
```bash
# Create new environment
conda create -n movie-recommender python=3.9

# Activate environment
conda activate movie-recommender

# Install packages
conda install pandas numpy matplotlib seaborn scikit-learn jupyter

# Additional packages
pip install plotly
```

#### Option B: Using pip + venv
```bash
# Create virtual environment
python -m venv movie-recommender-env

# Activate environment (Windows)
movie-recommender-env\Scripts\activate

# Activate environment (Mac/Linux)  
source movie-recommender-env/bin/activate

# Install packages
pip install pandas numpy matplotlib seaborn scikit-learn jupyter plotly
```

### 4. **VS Code Setup**

#### Required Extensions:
1. **Python** (Microsoft)
2. **Jupyter** (Microsoft) 
3. **Pylance** (Microsoft)

#### VS Code Settings:
1. **Select Python Interpreter:**
   - Press `Ctrl+Shift+P`
   - Type "Python: Select Interpreter"
   - Choose your environment's Python

2. **Jupyter Kernel:**
   - Open `EDA.ipynb`
   - Click "Select Kernel" in top right
   - Choose your Python environment

### 5. **Run the Notebook**

1. **Open VS Code in project directory**
2. **Open `EDA.ipynb`**
3. **Run cells sequentially:**
   - Use `Shift + Enter` to run each cell
   - Or use "Run All" from toolbar

---

## 🔧 TROUBLESHOOTING

### ❌ **"Data directory not found"**
**Solution:** 
- Check that MovieLens dataset is in `./movie/ml-latest-small/`
- Verify file paths match exactly

### ❌ **"Module not found" errors**
**Solution:**
```bash
# Reinstall packages
pip install pandas numpy matplotlib seaborn scikit-learn

# Check Python environment
which python  # Mac/Linux
where python  # Windows
```

### ❌ **Jupyter kernel issues**
**Solution:**
```bash
# Install ipykernel
pip install ipykernel

# Add environment to Jupyter
python -m ipykernel install --user --name=movie-recommender
```

### ❌ **Plots not showing**
**Solution:**
- Ensure `%matplotlib inline` is in the notebook
- Try `%matplotlib widget` for interactive plots
- Restart VS Code if needed

---

## 📦 REQUIRED PACKAGES

### Core Dependencies:
```
pandas>=1.5.0          # Data manipulation
numpy>=1.21.0           # Numerical computing
matplotlib>=3.5.0       # Basic plotting
seaborn>=0.11.0         # Statistical visualization
scikit-learn>=1.1.0     # Machine learning
jupyter>=1.0.0          # Notebook environment
```

### Optional Dependencies:
```
plotly>=5.0.0          # Interactive plots
ipywidgets>=7.6.0      # Interactive widgets
tqdm>=4.62.0           # Progress bars
```

### Install All at Once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter plotly ipywidgets tqdm
```

---

## 🎯 NEXT STEPS

After successful setup:

1. **✅ Run EDA.ipynb completely**
2. **📊 Review generated visualizations** 
3. **💾 Check processed data files in `./data/processed/`**
4. **📈 Ready for Stage 2: Model Development**

---

## 🆘 GETTING HELP

### Common Issues:
- **Path problems:** Use forward slashes `/` in paths
- **Permission errors:** Run VS Code as administrator if needed
- **Memory issues:** Close other applications if dataset is large

### Resources:
- **Pandas Documentation:** https://pandas.pydata.org/docs/
- **Matplotlib Gallery:** https://matplotlib.org/stable/gallery/
- **VS Code Python:** https://code.visualstudio.com/docs/python/python-tutorial

---

**Last Updated:** August 20, 2025  
**Compatibility:** Windows 10/11, macOS, Linux  
**Python Version:** 3.8+ (Recommended: 3.9)  
**VS Code Version:** 1.80+
