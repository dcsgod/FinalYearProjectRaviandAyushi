
```markdown
# Smart Crop Disease Detection and Yield Prediction System 🌾🧠

A full-stack, AI-powered system designed to assist farmers by detecting crop diseases from images, predicting crop yields based on environmental data, and classifying agricultural land using satellite imagery. This project integrates Java frontend, Python-based machine learning models, and geospatial analytics to deliver intelligent agricultural decision support.

---

## 🔍 Features

- **🌿 Disease Detection**: Upload a crop leaf image and detect diseases using a CNN model.
- **📊 Yield Prediction**: Predict crop yield using XGBoost regression based on parameters like rainfall, temperature, humidity, and soil pH.
- **🛰️ Satellite Image Classification**: Segment and classify irrigated and fallow lands using Mask R-CNN on multispectral satellite data.
- **📈 Dashboard**: View prediction results, NDVI graphs, historical records, and more.
- **📡 Real-Time Data**: Integrated with OpenWeatherMap and SoilGrids APIs for live weather and soil data.

---

## 🛠️ Tech Stack

| Component         | Technology/Framework                    |
|-------------------|------------------------------------------|
| Frontend          | JavaFX / Spring Boot                     |
| Backend           | Python (Flask / FastAPI)                 |
| ML Models         | TensorFlow, Keras, Scikit-learn, PyTorch |
| Database          | MySQL (structured), MongoDB (images)     |
| APIs              | OpenWeatherMap, SoilGrids, Google Earth Engine |
| Satellite Tools   | Rasterio, GDAL, NDVI Calculators         |
| Deployment        | Docker, AWS EC2, Terraform               |

---

## 📁 Project Structure

```

├── frontend/              # Java UI or Spring Boot web interface
├── backend/
│   ├── app.py             # Main Flask/FastAPI app
│   ├── models/            # Trained ML models
│   ├── routes/            # REST API endpoints
├── ml/
│   ├── disease\_detection/ # CNN model code
│   ├── yield\_prediction/  # XGBoost regression code
│   ├── satellite\_segmentation/ # Mask R-CNN setup
├── database/
│   ├── mysql\_schema.sql
│   ├── mongodb\_collections.json
├── docs/                  # Reports, papers, presentation
├── requirements.txt       # Python dependencies
├── Dockerfile
├── README.md

````

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Java 17+ (for JavaFX/Spring Boot)
- Node.js (for frontend graphs)
- MySQL & MongoDB installed
- Docker (optional for deployment)

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/smart-agriculture-system.git
cd smart-agriculture-system
````

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Setup the database:

```sql
Run `mysql_schema.sql` on your MySQL server
```

4. Start the backend API:

```bash
cd backend/
python app.py
```

5. Run the frontend (JavaFX):
   Open in IntelliJ/Eclipse and run `Main.java`
   *or*
   Run Spring Boot application if using web version

---

## 📷 Sample Use Cases

| Task                   | Input                     | Output                                    |
| ---------------------- | ------------------------- | ----------------------------------------- |
| Disease Detection      | Tomato leaf image         | Early blight with 94.3% confidence        |
| Yield Prediction       | Soil & weather parameters | Predicted Yield: 4.5 tons/hectare         |
| Satellite Segmentation | Coordinates / GeoTIFF     | NDVI map with 85% irrigated zone coverage |

---

## 📈 Results

* **Disease Detection Accuracy**: 96.4%
* **Yield Prediction R² Score**: 0.87
* **Satellite Classification mAP**: 0.29
* Real-time performance: <2s for disease/yield, \~5s for satellite analysis

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙏 Acknowledgments

* PlantVillage & Kaggle for datasets
* OpenWeatherMap & SoilGrids API
* Matterport Mask R-CNN (GitHub)
* Faculty mentors and peer reviewers

---

## 📬 Contact

* Ayushi Kumari – Java & UI Developer – `21EAYCS030`
* Ravi Kumar – Python & ML Developer – `21EAYCS117`

Department of Computer Science & Engineering
\[Your College Name], \[City, Country]

```

---

Let me know if you'd like this converted into a `README.md` file or published as a GitHub-ready repo setup!
```
