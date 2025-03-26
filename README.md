# GeoSyn

**GeoSyn** is an open-source Python tool for generating synthetic geotechnical 2D cross-sections. It was developed to support data-driven applications in geotechnical engineering by creating realistic and diverse training datasets for machine learning (ML) models.

This tool allows users to define key subsurface parameters, including:

- Number of layers  
- Geometry of layer boundaries (via sine/cosine functions)  
- Material categories and properties  
- Spatial heterogeneity using anisotropic Gaussian random fields

Each synthetic cross-section can represent any scalar geotechnical property such as:

- Soil type (e.g., sand, clay)  
- Soil Behaviour Type Index (Ic)  
- Permeability  
- Undrained shear strength  
- Any other scalar property expressible in 2D

---

## 🔍 Example Use Case

GeoSyn was used in the following peer-reviewed study to train a conditional Generative Adversarial Network (cGAN) that reconstructs geotechnical stratigraphy from sparse CPT data:

**Campos Montero, F.A., Zuada Coelho, B., Smyrniou, E., Taormina, R., & Vardon, P.J. (2025)**  
*SchemaGAN: A conditional Generative Adversarial Network for geotechnical subsurface schematisation*  
Computers and Geotechnics, 183, 107177  
📄 [https://doi.org/10.1016/j.compgeo.2025.107177](https://doi.org/10.1016/j.compgeo.2025.107177)

---

## 📦 Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/fabcamo/GeoSyn.git
cd GeoSyn
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

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute this software under the terms of that license.

## Third-party licenses
GeoSyn uses the following third-party library:

GSTools
Repository: https://github.com/GeoStat-Framework/GSTools

License: MIT License

## Research and Contributions

This project is part of broader research into the application of generative adversarial networks in geotechnical engineering. For more information and to access the full thesis, please visit [official thesis](https://repository.tudelft.nl/islandora/object/uuid:c18cb6cf-3574-484d-aacc-dabd882341de?collection=education).

---

