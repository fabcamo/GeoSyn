# GeoSyn: Synthetic Database Generation for Geotechnical Cross-Sections

## Overview
GeoSyn is a tool designed to generate a synthetic database of geotechnical cross-sections. This database aids in training a conditional Generative Adversarial Network (cGAN) capable of capturing complex relationships in subsurface schematisations, such as geometry in layer boundaries and anisotropy within each layer. The synthetic datasets are populated with Soil Behaviour Type index (Ic) values, derived from Cone Penetration Test (CPT) data, to facilitate geotechnical investigations.

For a detailed explanation of the synthetic data generation process, please refer to the work of Campos-Montero et al. (2023).

**Keywords:** Deep learning, machine learning, generative adversarial network, geotechnical engineering, cross-sections, synthetic data, cone penetration test, soil behaviour type.

![Alt text](https://github.com/fabcamo/GeoSchemaGen/blob/main/tests/schemas.png?raw=true)



## Installation

To set up your environment to run GeoSchemaGen, please follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the cloned directory.
3. Install the required dependencies using the following command:

    ```bash
    pip install -r requirements.txt
    ```  

## Usage

To generate the synthetic images, run the `main.py` script:

```bash
python main.py
``` 

The script will generate a synthetic database of geotechnical cross-sections in the `output` directory.


### Configuration

Before running the script, ensure you configure the following key input parameters:

- **Number of Realizations**: The total number of synthetic images to generate.
- **Size in X**: The length of the model in pixels.
- **Size in Y**: The depth of the model in pixels.

These parameters can be adjusted within the `main.py` file.

## Output

GeoSyn outputs geotechnical cross-sections in both PNG and CSV formats, formatted to a size of 32x512 pixels to maintain a 1:16 ratio for length over depth. This ensures computational efficiency and visual fidelity.

## Research and Contributions

This project is part of broader research into the application of generative adversarial networks in geotechnical engineering. For more information and to access the full thesis, please visit [official thesis](https://repository.tudelft.nl/islandora/object/uuid:c18cb6cf-3574-484d-aacc-dabd882341de?collection=education).

---

