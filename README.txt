DESCRIPTION
	Folder of the Ifrit application, an AI application that implement a CNN model built in Python for the Computational Intelligence and Deep Learning project at the University of Pisa.
	The task to be performed by the network and addressed in the project is forest fire detection.

DATASET
	The dataset used for the application can be found on kaggle at the following link: 
	https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images

The folder contains:
 - 3 python files: Fire_cls_GUI (main file, GUI), AlexNet_class, GoogLeNet class, IfritNet class (my own CNN)
 - Dataset: folder containing the dataset images used for training and testing networks in the project. This folder is divided into:
	– old ds: contains the original dataset except for the corrupted images that have been removed;
	– new ds: contains the dataset after the image analysis and selection phase, it does not contain the images that I found not congruent with the problem addressed.
 - Model: folder containing the saved model of the CNN trained during the project. This folder is divided into:
	– train hdf5: Contains checkpoint saves of models made during network training phases and other test saves;
	– best model: contains the saves of the best models.
 - result test CNN: folder containing a record of all network training done for the project. It is divided into subfolders, one for each type of CNN seen in the project. 
   There is data related to workouts done with GTX 1080, RTX 3060, and Google Colab. The training reports are both in notepad form (text lines) and saved images (accuracy and loss trends and confusion matrix).
 - Others folder: for future implementations, utilitys or CNN experiments in Python
 - CIDL_Documentation: documentation regarding the application and the analysis carried out for the choice of model and parameters.

Developer's notes
	The work related to the university examination has been done and the project is completed. 
	There may be updates or improvements to the project in the future, but nothing is planned for now.

Developers:
	- Alessandro Diana