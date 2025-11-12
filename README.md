# PATN-Predictive-Adversarial-Transformation-Network
A real-time, privacy-preserving framework that proactively generates future adversarial perturbations from historical signals. Ensuring continuous protection without disrupting the functionality of benign tasks.
## Appendix.pdf：
The Appendix serves as supplementary material for the AAAI26 oral paper (Privacy on the Fly: A Predictive Adversarial Transformation Network for Mobile Sensor Data), including interpretability and visualization analyses of PATN, illustrations of gyroscope IMU, and an evaluation of privacy‑preserving performance in sequence models.
## Dataset:
The full dataset is available for download from GitHub (MotionSense:https://github.com/mmalekzadeh/motion-sense).
## Code:
Code/train\_our\_model.py: The structure of the model we employ.
Code/train\_PATN.py: Training methods for PATN networks.
Code/test\_PATN.ipynb: Testing PATN's privacy protection capabilities compared to raw data.
Code/test\_HAR.ipynb: Testing the availability of PATN benign tasks.

## Model:
We have already trained models.
Model/Gender_model.pth: Gender Privacy Inference Model
Model/HAR_model.pth: Human Activity Recognition Model
Model/PATN_model.pth: Predictive Adversarial Transformation Model

