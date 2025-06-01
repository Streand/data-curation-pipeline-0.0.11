# Data Curation Pipeline

## Overview

The Data Curation Pipeline is a modular framework designed for processing and analyzing images and videos. It integrates various analysis tools and techniques, allowing users to perform tasks such as face detection, clothing analysis, pose estimation, and NSFW content detection. The project utilizes Gradio for a user-friendly interface, enabling rapid prototyping and easy extension of functionalities.

## Features

- **Modular Design**: Each analysis component is encapsulated in its own module, making it easy to add or modify features.
- **Gradio Integration**: A web-based interface for uploading data, configuring analysis options, and visualizing results.
- **Batch and Interactive Workflows**: Supports both batch processing of files and interactive analysis through the Gradio UI.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Git (optional, for cloning the repository)

### Installation

1. Clone the repository or download the project files.
2. Open a terminal and navigate to the project directory.
3. Run the following command to create a virtual environment and install dependencies:

   ```
   install.bat
   ```

### Running the Application

To start the Gradio application, run:

```
run.bat
```

This will activate the virtual environment and launch the Gradio interface in your default web browser.

## Directory Structure

```
data-curation-pipeline/
├── install.bat
├── run.bat
├── requirements.txt
├── gradio_app.py
├── config.yaml
├── app/
│   ├── gradio_tabs/
│   └── utils/
└── pipelines/
    ├── controller.py
    ├── input.py
    ├── face/
    │   ├── detection.py
    │   ├── emotion.py
    │   ├── pose.py
    │   ├── features.py
    │   └── tattoos.py
    ├── clothing.py
    ├── pose.py
    ├── camera.py
    ├── nsfw.py
    ├── accessories.py
    ├── aggregator.py
    └── export.py
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.