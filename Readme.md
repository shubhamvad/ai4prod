# BUILD

CPU
ONNXRUNTIME
Comando utilizzato per s
./build.sh --config RelWithDebInfo --build_shared_lib --parallel

#TORCHVISION

Per il momento è possibile compilare ma non lo aggiungiamo perchè non ci sono ancora le trasformazioni.

Per includerlo devo cambiare le path dei file torchvision/share/TorchVisionTargets-noconfig.cmake dove c'è scritto path da cambiare


La cartella torchvision è stata creata copiando i file in usr/local/include/torchvision /usr/local/lib/libtorchvision.so /usr/local/share/cmake/torchvision 
una volta completato il build della libreria e eseguendo sudo make install

# BUILD OPENCV 

cmake -DCMAKE_BUILD_TYPE= RELEASE -D CMAKE_INSTALL_PREFIX=/home/eric/Scrivania/2020/opencv4.1.0/install -D WITH_TTB=ON -D OPENCV_GENERATE_PKGCONFIG=YES -DBUILD_SHARED_LIBS=ON -DOPENCV_EXTRA_MODULES_PATH=./opencv_contrib/modules ..

make package -> crea un file zip contenente tutti i file per includere le opencv in un progetto esterno






## PROB 1 CPU
Non ho controllo su quante Cpu posso utilizzare

session_options.SetIntraOpNumThreads(1);

Thread Related
https://github.com/microsoft/onnxruntime/issues/3099#issuecomment-610668766


**nella versione corrente sembra risolto**

## TO DO

- Fare il training su imagenet in grayscale aumenta la detection in grayscale
-  Creare una distribuzione linux avviabile con le cuda già installate http://www.linuxandubuntu.com/home/make-your-very-own-customized-linux-distro-from-your-current-installation





# REPO DA TESTARE

#Video Recognition

https://github.com/facebookresearch/SlowFast

# Object Tracking

https://github.com/Stephenfang51/tracklite

https://github.com/LeonLok/Multi-Camera-Live-Object-Tracking

# Facial Recognition

https://medium.com/better-programming/add-facial-recognition-to-your-app-easily-with-face-api-js-58df65921e7

https://medium.com/@andreas.schallwig/do-not-laugh-a-simple-ai-powered-game-3e22ad0f8166

https://github.com/justadudewhohacks/face-api.js/

# Object Detection 

DETR https://www.youtube.com/watch?v=T35ba_VXkMY

# Search Similarity

https://github.com/facebookresearch/faiss

# multi modal learning

https://github.com/facebookresearch/mmf

# Human Pose Estimation

https://github.com/leoxiaobin/deep-high-resolution-net.pytorch

Tensorrt thread https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/issues/9

https://github.com/NVIDIA-AI-IOT/trt_pose

# latent autoencoder

https://github.com/podgorskiy/ALAE

# Detectron

https://github.com/facebookresearch/detectron2


# Face recognition

https://github.com/timesler/facenet-pytorch

https://towardsdatascience.com/face-detection-recognition-and-emotion-detection-in-8-lines-of-code-b2ce32d4d5de


# Action Detection

https://github.com/zhoubolei/TRN-pytorch
https://github.com/open-mmlab/mmaction

#Background Matting

https://github.com/senguptaumd/Background-Matting


# NLP

# Link Article

https://github.com/facebookresearch/BLINK

# Sentence Classification

https://github.com/facebookresearch/fastText


#Sport

https://github.com/chonyy/AI-basketball-analysis
https://medium.com/@osai.ai/osai-empowered-russian-table-tennis-championship-with-cv-and-ai-analytics-e7d52a6d8a5c

# Camera Calibration

https://github.com/puzzlepaint/camera_calibration

# Voice Cloning

https://unilight.github.io/Publication-Demos/publications/transformer-vc/

https://github.com/soobinseo/Transformer-TTS

https://nv-adlr.github.io/WaveGlow

# Repository Tensorrt Model

https://github.com/wang-xinyu/tensorrtx




# ONNXRUNTIME LIBTORCH THREAD

https://discuss.pytorch.org/t/onnx-deploying-a-trained-model-in-a-c-project/9593/13


# TORCHVISION CPP

https://discuss.pytorch.org/t/about-torchvision-for-c-frontend/49822/8

Scivere a Sharirar quando il framework sarà pronto













