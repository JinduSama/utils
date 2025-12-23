# Configuration

DS Utils is highly configurable to match your organization's branding and preferences.

## Configuration File

The main configuration file is `plot_styles.yaml` located in the `config` module. This file controls:

- Color palettes
- Fonts and sizes
- Figure dimensions
- Grid and axis styling
- Logo settings
- Output settings
- Locale settings

## Default Configuration

```yaml
# Color Palettes
colors:
  categorical:
    - "#1f77b4"  # Blue
    - "#ff7f0e"  # Orange
    - "#2ca02c"  # Green
    - "#d62728"  # Red
    - "#9467bd"  # Purple
    - "#8c564b"  # Brown
    - "#e377c2"  # Pink
    - "#7f7f7f"  # Gray
    - "#bcbd22"  # Olive
    - "#17becf"  # Cyan
  
  sequential:
    - "#f7fbff"
    - "#deebf7"
    - "#c6dbef"
    - "#9ecae1"
    - "#6baed6"
    - "#4292c6"
    - "#2171b5"
    - "#08519c"
    - "#08306b"
  
  diverging:
    - "#d73027"
    - "#f46d43"
    - "#fdae61"
    - "#fee090"
    - "#ffffbf"
    - "#e0f3f8"
    - "#abd9e9"
    - "#74add1"
    - "#4575b4"

# Font Settings
fonts:
  family: "sans-serif"
  title_size: 14
  label_size: 12
  tick_size: 10
  legend_size: 10

# Figure Settings
figure:
  default_size: [10, 6]
  dpi: 100
  facecolor: "white"
  edgecolor: "white"

# Grid Settings
grid:
  show: true
  color: "#E0E0E0"
  linestyle: "-"
  linewidth: 0.5
  alpha: 0.7

# Spine Settings
spines:
  top: false
  right: false
  left: true
  bottom: true

# Logo Settings
logo:
  path: null
  position: "lower right"
  size: 0.1
  alpha: 0.8

# Save Settings
save:
  formats: ["png"]
  dpi: 300
  bbox_inches: "tight"
  transparent: false

# Locale Settings
locale:
  name: "de_DE.UTF-8"
  decimal_separator: ","
  thousands_separator: "."
```

## Customizing Configuration

### Method 1: Modify the YAML File

Edit `src/ds_utils/config/plot_styles.yaml` directly:

```yaml
colors:
  categorical:
    - "#003366"  # Company Blue
    - "#CC0000"  # Company Red
    - "#009933"  # Company Green
```

### Method 2: Runtime Configuration

Override settings at runtime:

```python
from ds_utils.config import set_config, get_config

# Get current configuration
config = get_config()

# Modify settings
config["colors"]["categorical"] = ["#003366", "#CC0000", "#009933"]

# Apply changes
set_config(config)
```

### Method 3: Environment Variables

Set configuration via environment variables:

```bash
export DS_UTILS_LOCALE="en_US.UTF-8"
export DS_UTILS_DPI="300"
```

## Color Palettes

### Categorical Palette

Used for discrete categories (default 10 colors):

```python
from ds_utils.plotting import get_color_palette

colors = get_color_palette("categorical")
```

### Sequential Palette

Used for continuous data ranging from low to high:

```python
colors = get_color_palette("sequential", n_colors=9)
```

### Diverging Palette

Used for data with a meaningful center point:

```python
colors = get_color_palette("diverging", n_colors=9)
```

### Custom Palettes

Add custom palettes to the configuration:

```yaml
colors:
  custom:
    my_palette:
      - "#FF0000"
      - "#00FF00"
      - "#0000FF"
```

```python
colors = get_color_palette("my_palette")
```

## Font Configuration

### System Fonts

Use fonts installed on your system:

```yaml
fonts:
  family: "Arial"
  title_size: 16
  label_size: 14
```

### Custom Fonts

Install and use custom fonts:

```python
import matplotlib.font_manager as fm

# Add custom font
font_path = "/path/to/CustomFont.ttf"
fm.fontManager.addfont(font_path)

# Update configuration
from ds_utils.config import set_font_family
set_font_family("CustomFont")
```

## Locale Configuration

### German Locale (Default)

```yaml
locale:
  name: "de_DE.UTF-8"
  decimal_separator: ","
  thousands_separator: "."
```

Numbers are formatted as: `1.234,56`

### US Locale

```yaml
locale:
  name: "en_US.UTF-8"
  decimal_separator: "."
  thousands_separator: ","
```

Numbers are formatted as: `1,234.56`

### Setting Locale Programmatically

```python
from ds_utils.plotting import setup_locale

# German locale
setup_locale("de_DE.UTF-8")

# US locale
setup_locale("en_US.UTF-8")
```

## Logo Configuration

Add your corporate logo to plots:

```yaml
logo:
  path: "/path/to/logo.png"
  position: "lower right"  # or "lower left", "upper right", "upper left"
  size: 0.1  # Relative size (0-1)
  alpha: 0.8  # Transparency
```

### Programmatic Logo Addition

```python
from ds_utils.plotting import add_corporate_logo

fig, ax = plot_line(x, y)
add_corporate_logo(fig, logo_path="/path/to/logo.png", position="lower right")
```

## Save Configuration

Configure default save settings:

```yaml
save:
  formats: ["png", "pdf", "svg"]
  dpi: 300
  bbox_inches: "tight"
  transparent: false
  output_dir: "./figures"
```

### Programmatic Save

```python
from ds_utils.plotting import save_plot

save_plot(
    fig, 
    "my_figure",
    formats=["png", "pdf"],
    dpi=300,
    output_dir="./reports/figures",
)
```

## Logging Configuration

Configure logging behavior:

```python
from ds_utils.config import setup_logging

# Console only
setup_logging(level="INFO", console=True, file=False)

# File only
setup_logging(level="DEBUG", console=False, file=True, log_file="ds_utils.log")

# Both
setup_logging(level="INFO", console=True, file=True)
```

## Configuration Presets

### Research Preset

High-quality settings for publication:

```python
from ds_utils.config import apply_preset

apply_preset("research")
# - DPI: 300
# - Formats: ["png", "pdf", "svg"]
# - Tight bounding box
# - Transparent background
```

### Presentation Preset

Settings optimized for presentations:

```python
apply_preset("presentation")
# - Larger fonts
# - Higher contrast colors
# - Larger figure size
```

### Report Preset

Settings for business reports:

```python
apply_preset("report")
# - Corporate colors
# - Logo enabled
# - German locale
```

## Environment-Specific Configuration

Load different configurations based on environment:

```python
import os
from ds_utils.config import load_config

env = os.getenv("DS_UTILS_ENV", "development")

if env == "production":
    load_config("config/production.yaml")
elif env == "testing":
    load_config("config/testing.yaml")
else:
    load_config("config/development.yaml")
```
