# HDNN_MFDL
Final project submission for MACHINE LEARNING AND PROGRAMMING (MEU5053.01-00) (Project 14)

## How To Use
* Environment: Tensorflow 2.7 / Python 3.9
* Required library
  - pip install tensorflow_addons
  - pip install pynrrd
  - pip install seg_metrics

* Due to data size and security reasons, the file was not uploaded to Github, and the upload was replaced with a Google Drive link
  - https://drive.google.com/file/d/1-8TWSfcniGsfHAYrhqLqP9phthq_viqO/view?usp=sharing
  - Please download the .pkl file through the link, and place it at '/pkl'

* Execute the code after executing the PointNet_SyntheticVessel3D_V2.ipynb file

## Problem Statement
CAD(Coronary artery disease) is one of the most dangerous diseases for humans. Therefore, it is important to diagnose accurately to avoid severe situation. For diagnosing this disease, an invasive diagnostic process or CFD(computational fluid dynamics) simulation is used to predict the disease. 
Fractional flow reserve (FFR) is used as a representative diagnostic factor for heart disease. To measure this, an invasive method is used.
However, they have burdens such as cost and complications, and because it takes a lot of time to derive results. Therefore, we tried to support diagnosis by suggesting additional non-invasive diagnostic factor for heart disease.

## Data Generation
We utilized Lattice boltzman method to perform hemodynamics simulation. Compared to conventional grid, LBM overcomes the existing limitations because it simplifies and analyzes the grid, and it can predict values that change over time. In addition, LBM is highly used to analyze meso-scale flow dynamics. Therefore, it is proper to analyze coronary artery lesion. We introduced the simulation domain to describe coronary artery lesion.

## Method
### 'PointNet' on 3D synthetic stenotic vessel
1. Model point cloud: spatial coordinates (x, y, z) for the outermost points of the vessel
2. Query point cloud: the remaining points inside the vessel, which includes spatial coordinates (x, y, z) and its corresponding hemodynamics (pressure, Ux, Uy, Uz)

## Quantitative Performance
![image](https://user-images.githubusercontent.com/56405223/175470712-2a7e1068-0306-4b00-a459-beac82e23791.png)
![image](https://user-images.githubusercontent.com/56405223/175470739-589e0c95-659e-49e0-b67e-4c92d170c4a2.png)
NMAE can characterize the error of the deep learning prediction result relative to the true value of the overall flow field (CFD result). MRE can characterize the error of the deep learning prediction value relative to the true value at all query points of the model. The definition of the error function draws on previous studies.

## Qualitative Performance
![image](https://user-images.githubusercontent.com/56405223/175470842-838b6ea8-c8fb-42c9-965f-13a59e7c80e8.png)
![image](https://user-images.githubusercontent.com/56405223/175470863-3e4e7b41-0704-42e1-9501-d124c5de6b47.png)
![image](https://user-images.githubusercontent.com/56405223/175470885-cafb1162-8b49-4642-829f-9191e3ae625b.png)
![image](https://user-images.githubusercontent.com/56405223/175470905-833c8796-af5d-46c8-83ef-e045752c888f.png)
![image](https://user-images.githubusercontent.com/56405223/175470934-4f6b1c35-8414-4385-b4ce-17f7349ee5a8.png)
