# Dog Breed Identification

Different approaches to solve this [kaggle](https://www.kaggle.com/c/dog-breed-identification) competition. Train and test data can be downloaded [here](http://vision.stanford.edu/aditya86/ImageNetDogs/) as well.

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


This repo does not include the dataset itself. Download the dataset from an external source [here](https://www.kaggle.com/c/dog-breed-identification/data) or [here](http://vision.stanford.edu/aditya86/ImageNetDogs/). Unzip all downloaded files and put them in a folder in the project root `datasets/`.

(The project should have the following structure)

```
dog-breed-identification/
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
