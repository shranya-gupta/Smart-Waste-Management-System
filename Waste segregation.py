import os
import cvzone
from cvzone.ClassificationModule import Classifier
import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load the trained model and labels
classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')

# Initialize bin images
imgBinsList = []
pathFolderBins = "Resources/Bins"
pathList = os.listdir(pathFolderBins)
pathList.sort()  # Ensure consistent ordering of bins
for path in pathList:
    imgBinsList.append(cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED))

# Map class IDs to corresponding bin indices
classDic = {
    0: 0,  # Default bin (e.g., unknown or no waste detected)
    1: 0,  # Recyclable waste
    2: 0,  # Recyclable waste
    3: 1,  # Non-Recyclable waste
    4: 1,  # Non-Recyclable waste
    5: 1,  # Non-Recyclable waste
    6: 0,  # Recyclable waste
    7: 1,  # Non-Recyclable waste
    8: 0   # Recyclable waste
}

while True:
    # Capture frame from webcam
    _, img = cap.read()
    imgResize = cv2.resize(img, (454, 340))  # Resize for overlay

    # Predict the waste type
    prediction = classifier.getPrediction(img)
    classID = prediction[1]
    print(f"Predicted Class: {classID}")

    # Determine the corresponding bin
    classIDBin = classDic.get(classID, 0)

    # Load the background and overlay the bin
    imgBackground = cv2.imread('Resources/background.png')
    imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[classIDBin], (895, 374))

    # Overlay the resized webcam feed on the background
    imgBackground[148:148 + 340, 159:159 + 454] = imgResize

    # Display output with background and bins in another window
    cv2.imshow("Output", imgBackground)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
