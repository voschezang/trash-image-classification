# Classification of trash images

This code expands upon the research paper "_Classification of Trash for Recyclability Status._" by Yang, M., and Thung, G. (2016). Available at: [link](https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf)
Download the dataset [here](https://drive.google.com/drive/folders/0B3P9oO5A3RvSUW9qTG11Ul83TEE)

## Setup

Install dependencies 
```
git clone https://github.com/voschezang/dog-breed-identification.git
cd dog-breed-identification
make deps start
```

Or, if you do not have `pip3` installed
```
make deps2 start
```

Then navigate to the folder _nn_ and the file "_transfer learning_final.ipynb_"


###

(The project should have the following structure)

```
image-classification/
  src/
    (jupyter notebooks)
    (some python scripts)
  datasets/
    labels.csv
    train/
      image-0.jpg
    test/
      image-0.jpg
```


Run from the command line with
```
make
```
