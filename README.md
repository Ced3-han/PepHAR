# PepHAR
[ICLR 2025] Hotspot-Driven Peptide Design via Multi-Fragment Autoregressive Extension

You can use the same enviroment as PepFlow (https://github.com/Ced3-han/PepFlowww) for running this method.

After cloning the repo, plese download the **data** and **ckpts** from google drive (https://drive.google.com/drive/folders/1jJFPZbczI7Nxai-9X5UcNsv5U8rcBUEY?usp=sharing) and place them in this folder.

Please see ```train_ddp.py``` for training and ```sample.py``` for sampling. 

During training, as we finally find that the two models heavily overfitted the training set, we select early training steps for evaluation.

During sampling, you can alternate different sample methods and please adjust the root dirs.

Please do not hesitate to reach out to me at ced3ljhypc@gmail.com (Jiahan Li) with any inquiries regarding our methodology or implementations.
