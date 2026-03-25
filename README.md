PROJECT TITLE: 
FADING LIGHT: REVOLUTIONIZING GLAUCOMA DIAGNOSTICS WITH AI 

DATASET: HILLEL YAFFE GLAUCOMA DATASET (HYGD)

VISION K: GLAUCOMA AI RISK SCORING SYSTEM 



HOW TO RUN & REPRODUCE? 

STEP 1: DATA CONFIGURATION

Download all files in the Github repo and put it under one file 

STEP 2: ENVIRONMENT SETUP

Install the dependencies using the provided requirements.txt

pip install -r requirements.txt

STEP 3: EXECUTION CONTROL FLAGS 

In Section 1, you can toggle the following flags to skip or force complicated computations:

RETRAIN_MODEL = False: Set to True only if you want to overwrite the existing weights in models/

RUN_KFOLD = True: Run this to generate the kfold_results.csv needed for the performance report. You may set to False if you want to use our kfold_results.csv. 

STEP 4: PREPROCESSING WORKFLOW 

For absolute reproducibility, the following steps are documented in Section 3 of the code:

Adaptive CLAHE: Enhances optic disc boundaries based on image resolution.

Green Channel Sharpening: Uses an unsharp mask to highlight structural RNFL information.

Quality Filter: Automatically excludes images with an HYGD quality score < 3.0.

STEP 5: PERFORMANCE & DUAL XAI FRAMEWORK

Metrics: Achieved a test AUC of 0.9992 and a 5-fold mean AUC of 0.9846 +- 0.0064.

Interpretability: Outputs Grad-CAM heatmaps and SHAP waterfalls to verify the model focuses on clinical markers (Optic Disc/RNFL) rather than artifacts.

You may refer to this link to deploy our app Vision K on Streamlit: 

https://visionkvitalkidsc2026-areyougon2026.streamlit.app/

