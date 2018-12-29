"""Helper library for downloading open images categorically."""
import t4

import pandas as pd
import requests
import os

from tqdm import tqdm
import ratelim
from checkpoints import checkpoints
checkpoints.enable()


# TODO: refactor downloads to use a data package you define
def download(categories, packagename, registry,
             class_names_fp=None, train_boxed_fp=None, image_ids_fp=None, prior_results=None):
    """Download images in categories from flickr"""

    # Download or load the class names pandas DataFrame
    kwargs = {'header': None, 'names': ['LabelID', 'LabelName']}
    orig_url = "https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv"
    class_names = pd.read_csv(class_names_fp, **kwargs) if class_names_fp else pd.read_csv(orig_url, **kwargs)

    # TODO: setting index_col should not be necessary in this and the next section, update the save-to-disk code
    # Download or load the boxed image metadata pandas DataFrame
    orig_url = "https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv"
    train_boxed = pd.read_csv(train_boxed_fp, index_col=0) if train_boxed_fp else pd.read_csv(orig_url)

    # Download or load the image ids metadata pandas DataFrame
    orig_url = "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"
    image_ids = pd.read_csv(image_ids_fp, index_col=0) if image_ids_fp else pd.read_csv(orig_url)

    # Get category IDs for the given categories and sub-select train_boxed with them.
    label_map = dict(class_names.set_index('LabelName').loc[categories, 'LabelID']
                     .to_frame().reset_index().set_index('LabelID')['LabelName'])
    label_values = set(label_map.keys())
    relevant_training_images = train_boxed[train_boxed.LabelName.isin(label_values)]

    # Start from prior results if they exist and are specified, otherwise start from scratch.
    relevant_flickr_urls = (relevant_training_images.set_index('ImageID')
                            .join(image_ids.set_index('ImageID'))
                            .loc[:, 'OriginalURL'])
    relevant_flickr_img_metadata = (relevant_training_images.set_index('ImageID').loc[relevant_flickr_urls.index]
                                    .pipe(lambda df: df.assign(LabelValue=df.LabelName.map(lambda v: label_map[v]))))
    remaining_todo = len(relevant_flickr_urls) if checkpoints.results is None else\
        len(relevant_flickr_urls) - len(checkpoints.results)
    print(f"Parsing {remaining_todo} images (of which "
          f"{len(relevant_flickr_urls) - remaining_todo} have already been downloaded)")

    # Download the images
    with tqdm(total=remaining_todo) as progress_bar:
        relevant_image_requests = relevant_flickr_urls.safe_map(lambda url: _download_image(url, progress_bar))
        progress_bar.close()

    # Initialize a new data package or update an existing one
    p = t4.Package.browse(packagename, registry) if packagename in t4.list_packages(registry) else t4.Package()

    # Write the images to files, adding them to the package as we go along.
    if not os.path.isdir("temp/"):
        os.mkdir("temp/")
    for ((_, r), (_, url), (_, meta)) in zip(relevant_image_requests.iteritems(), relevant_flickr_urls.iteritems(),
                                             relevant_flickr_img_metadata.iterrows()):
        image_name = url.split("/")[-1]
        image_label = meta['LabelValue']

        _write_image_file(r, image_name)

        p.set(f"{image_label}/{image_name}", f"temp/{image_name}", meta=dict(meta))

    # Push the updated package
    tophash = p.push(packagename, registry)

    # TODO: delete the temporary folder
    return tophash


@ratelim.patient(5, 5)
def _download_image(url, pbar):
    """Download a single image from a URL, rate-limited to once per second"""
    r = requests.get(url)
    r.raise_for_status()
    pbar.update(1)
    return r


def _write_image_file(r, image_name):
    """Write an image to a file"""
    filename = f"temp/{image_name}"
    with open(filename, "wb") as f:
        f.write(r.content)
