## PLANTMED – AI-Powered Medicinal Plant Identification System ##

## Included featuresb ##
- AI medicinal plant identification using the supplied InceptionV3 weights
- Image upload and camera capture
- Top-5 predictions and confidence scores
- Low-confidence / verification warning
- Medicinal plant metadata from the supplied Excel database
- Safety guidance layer and medical-use disclaimer
- English + Kannada-ready interface
- Observation/GPS capture with SQLite storage and map
- Expert verification workflow
- Research/evidence search starting points
- Retrieval-based Plant Assistant
- Plant Explorer and searchable database
- Analytics dashboard
- FastAPI REST API (`/health`, `/predict`)
- Responsive Streamlit UI
- Docker and Windows launcher files

## VS Code / Windows ##
1. Extract this folder.
2. Open the `PlantMed_Final` folder in VS Code.
3. Open Terminal.
4. Run:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run streamlit_app.py
   ```
5. Open the URL shown by Streamlit, normally `http://localhost:8501`.

Or double-click `run_streamlit.bat`.

## REST API
With the virtual environment activated:
```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```
## API docs: `http://localhost:8000/docs`

## Important
The model and supplied plant database are retained from the original project. Safety text is deliberately conservative. PLANTMED is an educational identification and information system, not a medical diagnosis, prescription, or guarantee of plant safety. Expert and clinical verification is required before medicinal use.

## Production roadmap
For a worldwide deployment, add authenticated experts, curated literature/DOI records, multilingual translations, browser/mobile geolocation permissions, model monitoring, audit logs, privacy controls, community moderation, and a provenance/licensing system for traditional knowledge.
