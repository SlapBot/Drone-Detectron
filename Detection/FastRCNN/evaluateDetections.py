from fastRCNN.test import test_net
from imdb_data import imdb_data
from cntk_helpers import makeDirectory, parseCntkOutput, DummyNet

image_set = 'test'
classifier = "nn"
cntkFilesDir = "/home/slapbot/my_side_projects/drone-detection/Detection/FastRCNN/proc/Drones_500/cntkFiles/"
print("Parsing CNTK output for image set: " + image_set)
cntkImgsListPath = cntkFilesDir + image_set + ".txt"
outParsedDir = cntkFilesDir + image_set + "_parsed/"
cntkOutputPath = cntkFilesDir + image_set + ".z"
cntk_nrRois = 500
cntk_featureDimensions = {
    "nn": 3
}

makeDirectory(outParsedDir)
parseCntkOutput(cntkImgsListPath, cntkOutputPath, outParsedDir, cntk_nrRois, cntk_featureDimensions[classifier],
                saveCompressed=True, skipCheck=True)

classes = ('__background__',  # always index 0
           'drone', 'dummy')
datasetName = "Drones"
imgDir = "/home/slapbot/my_side_projects/drone-detection/DataSets/Drones/"
roiDir = "/home/slapbot/my_side_projects/drone-detection/Detection/FastRCNN/proc/Drones_500/rois/"
imdbs = dict()  # database provider of images and image annotations
for image_set in ["train", "test"]:
    imdbs[image_set] = imdb_data(image_set, classes, cntk_nrRois, imgDir, roiDir, cntkFilesDir,
                                 boAddGroundTruthRois=(image_set != 'test'))

imdb = imdbs[image_set]
net = DummyNet(4096, imdb.num_classes, outParsedDir)

evalTempDir = None
classifier = "nn"
nmsThreshold = 0.01

test_net(net, imdb, evalTempDir, None, classifier, nmsThreshold, boUsePythonImpl=True)

print("DONE.")
