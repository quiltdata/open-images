# open images

This repository contains the code, in Python scripts and Jupyter notebooks, for building a convolutional neural network machine learning classifier based on a custom subset of the Google Open Images dataset.

For an overview of this project be sure to also read the complimentary article: "Building custom datasets and models using Google Open Images" (link forthcoming).

To get everything you need to retrain the model locally:

```bash
git clone https://github.com/quiltdata/open-images.git
cd open-images/
conda env create -f environment.yml
source activate quilt-open-images-dev
python -c "import t4; t4.Package.install('s3://quilt-example', 'quilt/open_images', './')"
```

### The data

The [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html) is an enormous image dataset intended for use in machine learning projects. A Google project, V1 of this dataset was [initially released](https://ai.googleblog.com/2016/09/introducing-open-images-dataset.html) in late 2016. This repository and project is based on V4 of the data.

The Open Images Dataset is an attractive target for building image recognition algorithms because it is one of the largest, most accurate, and most easily accessible image recognition datasets. For image recognition tasks, Open Images contains 15 million bounding boxes for 600 categories of objects on 1.75 million images. Image labeling tasks meanwhile enjoy 30 million labels across almost 20,000 categories.

The images are of highly variable quality, as would be realistic in an applied machine learning setting. They come from Flickr.

To learn more about the Open Images Dataset check out their [homepage](https://storage.googleapis.com/openimages/web/index.html).

### Downloading the data

Downloading the entire Google Open Images corpus is possible and potentially necessary if you want to build a general purpose image classifier or bounding box alglorithm. However downloading *everything* is a waste if you just want a small categorical subset of the data in the corpus.

The `src/openimager` subfolder contains a small module that handles downloading a categorical subset of the Open Images corpus: just the images corresponding with a user-selected group of labels, and just from the set of images with bounding box information attached. Instead of using the zipped blob files it does so by downloading the source images from flickr direcly.

The `openimager` mainfile is well-annotated in-code. To run it as a module run `import openimager; openimager.download([...list of category names...])`. To run it from the command line run `python openimager "some_category" "some_other_category" ...`.

Note that `notebooks/initial-exploration.ipynb` may also be helpful reading, to help contextualize how the dataset is distributed.

The result of running this script is the core of the `quilt/open_images` data package. If you install that data package you do not need to also re-source from flickr yourself.

### Building a custom dataset

The `notebooks/build-dataset.ipynb` notebook walk through processing and bundling the images downloaded in the previous step into a well-structured directory of properly cropped, tag-identified images on disk.

### Building a model

The `notebooks/build-model.ipynb` notebook walks through building a simple CNN (convolutional neural network) based on this dataset. `notebooks/evaluate-model.ipynb` is a separate notebook with an evaluation of that model's performance.

### Distributing the model

The `notebooks/distribute-model.ipynb` notebook walks through distributing the data and model artifacts externally. It follows the principle of performing version control over `{data, code, environment}`, using good tools and best practices for each component. For more on this idea see the article ["Reproducing a machine learning model build in four lines of code"](https://blog.quiltdata.com/reproduce-a-machine-learning-model-build-in-four-lines-of-code-b4f0a5c5f8c8).