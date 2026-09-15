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

These parameters can be adjusted directly within the `main.py` file.

For more advanced use, users can also customize geotechnical properties, their distributions, and boundary geometry settings. This requires deeper modifications to the codebase, particularly in the modules responsible for random field generation and stratigraphy modeling.


## Output

GeoSyn outputs geotechnical cross-sections in both PNG and CSV formats, formatted to a size of 32x512 pixels to maintain a 1:16 ratio for length over depth. This ensures computational efficiency and visual fidelity.

---

## 📄 Related Publication

GeoSyn is used in the following publication as part of a deep generative modeling pipeline for subsurface stratigraphy:

Campos Montero, F.A., Zuada Coelho, B., Smyrniou, E., Taormina, R., & Vardon, P.J. (2025)  
*SchemaGAN: A conditional Generative Adversarial Network for geotechnical subsurface schematisation*  
Computers and Geotechnics, 183, 107177  
[https://doi.org/10.1016/j.compgeo.2025.107177](https://doi.org/10.1016/j.compgeo.2025.107177)

---

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 📚 Third-Party Licenses

GeoSyn depends on several third-party Python packages, including:

- [GSTools](https://github.com/GeoStat-Framework/GSTools) (MIT License)

These packages are subject to their own licenses, including MIT, BSD, PSF, and Apache 2.0 variants. While GeoSyn is released under the MIT License, you must ensure compliance with the licenses of any dependencies you use.

For a full list of dependencies, see [`requirements.txt`](./requirements.txt).

To review license information for installed packages, you can use:

```bash
pip install pip-licenses
pip-licenses
```

---

