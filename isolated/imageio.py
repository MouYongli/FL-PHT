#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from abc import ABC, abstractmethod
from typing import Tuple, Union, List
import numpy as np

import nibabel
from nibabel import io_orientation


class BaseReaderWriter(ABC):
    @staticmethod
    def _check_all_same(input_list):
        # compare all entries to the first
        for i in input_list[1:]:
            if i != input_list[0]:
                return False
        return True

    @staticmethod
    def _check_all_same_array(input_list):
        # compare all entries to the first
        for i in input_list[1:]:
            if i.shape != input_list[0].shape or not np.allclose(i, input_list[0]):
                return False
        return True

    @abstractmethod
    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        """
        Reads a sequence of images and returns a 4d (!) np.ndarray along with a dictionary. The 4d array must have the
        modalities (or color channels, or however you would like to call them) in its first axis, followed by the
        spatial dimensions (so shape must be c,x,y,z where c is the number of modalities (can be 1)).
        Use the dictionary to store necessary meta information that is lost when converting to numpy arrays, for
        example the Spacing, Orientation and Direction of the image. This dictionary will be handed over to write_seg
        for exporting the predicted segmentations, so make sure you have everything you need in there!

        IMPORTANT: dict MUST have a 'spacing' key with a tuple/list of length 3 with the voxel spacing of the np.ndarray.
        Example: my_dict = {'spacing': (3, 0.5, 0.5), ...}. This is needed for planning and
        preprocessing. The ordering of the numbers must correspond to the axis ordering in the returned numpy array. So
        if the array has shape c,x,y,z and the spacing is (a,b,c) then a must be the spacing of x, b the spacing of y
        and c the spacing of z.

        In the case of 2D images, the returned array should have shape (c, 1, x, y) and the spacing should be
        (999, sp_x, sp_y). Make sure 999 is larger than sp_x and sp_y! Example: shape=(3, 1, 224, 224),
        spacing=(999, 1, 1)

        For images that don't have a spacing, set the spacing to 1 (2d exception with 999 for the first axis still applies!)

        :param image_fnames:
        :return:
            1) a np.ndarray of shape (c, x, y, z) where c is the number of image channels (can be 1) and x, y, z are
            the spatial dimensions (set x=1 for 2D! Example: (3, 1, 224, 224) for RGB image).
            2) a dictionary with metadata. This can be anything. BUT it HAS to include a {'spacing': (a, b, c)} where a
            is the spacing of x, b of y and c of z! If an image doesn't have spacing, just set this to 1. For 2D, set
            a=999 (largest spacing value! Make it larger than b and c)

        """
        pass

    @abstractmethod
    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        """
        Same requirements as BaseReaderWriter.read_image. Returned segmentations must have shape 1,x,y,z. Multiple
        segmentations are not (yet?) allowed

        If images and segmentations can be read the same way you can just `return self.read_image((image_fname,))`
        :param seg_fname:
        :return:
            1) a np.ndarray of shape (1, x, y, z) where x, y, z are
            the spatial dimensions (set x=1 for 2D! Example: (1, 1, 224, 224) for 2D segmentation).
            2) a dictionary with metadata. This can be anything. BUT it HAS to include a {'spacing': (a, b, c)} where a
            is the spacing of x, b of y and c of z! If an image doesn't have spacing, just set this to 1. For 2D, set
            a=999 (largest spacing value! Make it larger than b and c)
        """
        pass

    @abstractmethod
    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        """
        Export the predicted segmentation to the desired file format. The given seg array will have the same shape and
        orientation as the corresponding image data, so you don't need to do any resampling or whatever. Just save :-)

        properties is the same dictionary you created during read_images/read_seg so you can use the information here
        to restore metadata

        IMPORTANT: Segmentations are always 3D! If your input images were 2d then the segmentation will have shape
        1,x,y. You need to catch that and export accordingly (for 2d images you need to convert the 3d segmentation
        to 2d via seg = seg[0])!

        :param seg: A segmentation (np.ndarray, integer) of shape (x, y, z). For 2D segmentations this will be (1, y, z)!
        :param output_fname:
        :param properties: the dictionary that you created in read_images (the ones this segmentation is based on).
        Use this to restore metadata
        :return:
        """
        pass

class NibabelIO(BaseReaderWriter):
    """
    Nibabel loads the images in a different order than sitk. We convert the axes to the sitk order to be
    consistent. This is of course considered properly in segmentation export as well.

    IMPORTANT: Run nnUNet_plot_dataset_pngs to verify that this did not destroy the alignment of data and seg!
    """
    supported_file_endings = [
        '.nii.gz',
        '.nrrd',
        '.mha'
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images = []
        original_affines = []

        spacings_for_nnunet = []
        for f in image_fnames:
            nib_image = nibabel.load(f)
            assert nib_image.ndim == 3, 'only 3d images are supported by NibabelIO'
            original_affine = nib_image.affine

            original_affines.append(original_affine)

            # spacing is taken in reverse order to be consistent with SimpleITK axis ordering (confusing, I know...)
            spacings_for_nnunet.append(
                    [float(i) for i in nib_image.header.get_zooms()[::-1]]
            )

            # transpose image to be consistent with the way SimpleITk reads images. Yeah. Annoying.
            images.append(nib_image.get_fdata().transpose((2, 1, 0))[None])

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same_array(original_affines):
            print('WARNING! Not all input images have the same original_affines!')
            print('Affines:')
            print(original_affines)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNet_plot_dataset_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(spacings_for_nnunet):
            print('ERROR! Not all input images have the same spacing_for_nnunet! This might be caused by them not '
                  'having the same affine')
            print('spacings_for_nnunet:')
            print(spacings_for_nnunet)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        stacked_images = np.vstack(images)
        dict = {
            'nibabel_stuff': {
                'original_affine': original_affines[0],
            },
            'spacing': spacings_for_nnunet[0]
        }
        return stacked_images.astype(np.float32), dict

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname, ))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        # revert transpose
        seg = seg.transpose((2, 1, 0)).astype(np.uint8)
        seg_nib = nibabel.Nifti1Image(seg, affine=properties['nibabel_stuff']['original_affine'])
        nibabel.save(seg_nib, output_fname)


class NibabelIOWithReorient(BaseReaderWriter):
    """
    Reorients images to RAS

    Nibabel loads the images in a different order than sitk. We convert the axes to the sitk order to be
    consistent. This is of course considered properly in segmentation export as well.

    IMPORTANT: Run nnUNet_plot_dataset_pngs to verify that this did not destroy the alignment of data and seg!
    """
    supported_file_endings = [
        '.nii.gz',
        '.nrrd',
        '.mha'
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images = []
        original_affines = []
        reoriented_affines = []

        spacings_for_nnunet = []
        for f in image_fnames:
            nib_image = nibabel.load(f)
            assert nib_image.ndim == 3, 'only 3d images are supported by NibabelIO'
            original_affine = nib_image.affine
            reoriented_image = nib_image.as_reoriented(io_orientation(original_affine))
            reoriented_affine = reoriented_image.affine

            original_affines.append(original_affine)
            reoriented_affines.append(reoriented_affine)

            # spacing is taken in reverse order to be consistent with SimpleITK axis ordering (confusing, I know...)
            spacings_for_nnunet.append(
                    [float(i) for i in reoriented_image.header.get_zooms()[::-1]]
            )

            # transpose image to be consistent with the way SimpleITk reads images. Yeah. Annoying.
            images.append(reoriented_image.get_fdata().transpose((2, 1, 0))[None])

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same_array(reoriented_affines):
            print('WARNING! Not all input images have the same reoriented_affines!')
            print('Affines:')
            print(reoriented_affines)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNet_plot_dataset_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(spacings_for_nnunet):
            print('ERROR! Not all input images have the same spacing_for_nnunet! This might be caused by them not '
                  'having the same affine')
            print('spacings_for_nnunet:')
            print(spacings_for_nnunet)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        stacked_images = np.vstack(images)
        dict = {
            'nibabel_stuff': {
                'original_affine': original_affines[0],
                'reoriented_affine': reoriented_affines[0],
            },
            'spacing': spacings_for_nnunet[0]
        }
        return stacked_images.astype(np.float32), dict

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname, ))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        # revert transpose
        seg = seg.transpose((2, 1, 0)).astype(np.uint8)

        seg_nib = nibabel.Nifti1Image(seg, affine=properties['nibabel_stuff']['reoriented_affine'])
        seg_nib_reoriented = seg_nib.as_reoriented(io_orientation(properties['nibabel_stuff']['original_affine']))
        assert np.allclose(properties['nibabel_stuff']['original_affine'], seg_nib_reoriented.affine), \
            'restored affine does not match original affine'
        nibabel.save(seg_nib_reoriented, output_fname)


if __name__ == '__main__':

    import os.path as osp
    MRI_CASE_PATH = "../data/miccai_fets2022_trainingdata/MICCAI_FeTS2022_TrainingData/FeTS2022_00000"
    img_file = osp.join(MRI_CASE_PATH, 'FeTS2022_00000_t1.nii.gz')
    seg_file = osp.join(MRI_CASE_PATH, 'FeTS2022_00000_seg.nii.gz')

    # nibio = NibabelIO()
    # images, dct = nibio.read_images([img_file])
    # seg, dctseg = nibio.read_seg(seg_file)

    # nibio_r = NibabelIOWithReorient()
    # images_r, dct_r = nibio_r.read_images([img_file])
    # seg_r, dctseg_r = nibio_r.read_seg(seg_file)
    #
    # nibio.write_seg(seg[0], '/home/isensee/seg_nibio.nii.gz', dctseg)
    # nibio_r.write_seg(seg_r[0], '/home/isensee/seg_nibio_r.nii.gz', dctseg_r)
    #
    # s_orig = nibabel.load(seg_file).get_fdata()
    # s_nibio = nibabel.load('/home/isensee/seg_nibio.nii.gz').get_fdata()
    # s_nibio_r = nibabel.load('/home/isensee/seg_nibio_r.nii.gz').get_fdata()

    nibio = NibabelIO()
    img, dctimg = nibio.read_images([img_file])
    seg, dctseg = nibio.read_seg(seg_file)
    print(img.shape, seg.shape)
    print(dctimg)
    print(dctseg)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img[0][img.shape[1] // 2])
    ax1.set_title('Image')
    ax2.imshow(seg[0][seg.shape[1] // 2])
    ax2.set_title('Mask')
    plt.show()

    from skimage.util import montage
    fig, ax1 = plt.subplots(1, 1, figsize=(20, 20))
    ax1.imshow(montage(img[0]), cmap='bone')
    plt.show()



