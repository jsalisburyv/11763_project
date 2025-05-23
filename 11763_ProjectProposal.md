## **Project proposal**

In this project, students are asked to perform **two of the three objectives** proposed:

- 1) DICOM loading and visualization
	- a) Download the sample RadCTTACEomics DDDD [\(click here\)](https://drive.google.com/drive/folders/1WtV5apHm2-6Ze9YM7YgljnXkO8ofMqIA?usp=sharing), where DDDD is the sample associated to you (see assignment in Aula Digital).
	- b) Visualize it with the help of a third party DICOM visualizer (3D-Slicer is recommended).
	- c) Load the reference CT image (pydicom) and the associated segmentations (highdicom). Rearrange the image and segmentation 'pixel array' given by PyDicom based on the headers. Some relevant headers include:
		- 'Acquisition Number'.
		- 'Slice Index'.
		- 'Per-frame Functional Groups Sequence' Ñ 'Image Position Patient'.
		- 'Segment Identification Sequence' Ñ 'Referenced Segment Number'.
	- d) Create an animation (e.g. gif file) with a rotating Maximum Intensity Projection on the coronal-sagittal planes, visualizing the tumoral mask.
- 2) 3D Image Segmentation
	- a) Consider the Tumor mask associated to the reference image, and extract its bounding box and centroid.
	- b) Create a semi-automatic tumor segmentation algorithm that only uses the CT image, and either the bounding box or the centroid of the tumor.
	- c) Visualize both the provided Tumor mask and the segmented Tumor mask on the image. Assess the correctness of the algorithm, numerically and visually.
- 3) 3D Rigid Coregistration
	- a) Coregister the input to the reference image, implementing all steps of the image coregistration yourself (i.e. without libraries such as PyElastix).
	- b) Visualize the Liver region on the input image space. Assess the correctness of the algorithm, numerically and visually.

## **Submission and grading**

Students are asked to submit:

## – **Intermediate submission**

- The self-evaluation form, completed according to the progress made.
- A brief document showing the progress made (max 2 pages, including a github link to your repository) and a list of questions on how to proceed/overcome difficulties found during the development (without page limit).

## – **Final submission**

- A 4-page summary, which should focus on the technical part of the project: the algorithms, their implementation, and their performance. Figures, code, title page and index are excluded from the page limit. Please include a github link to your repository.
- Slides and multimedia material to be used during the 10-minute oral presentation, which should include a demonstration of the software developed and the results obtained.

To calculate the final course grade, the following weights will used:

| Activity                | Weight                                           |
|-------------------------|--------------------------------------------------|
| Intermediate submission | –                                                |
| Final submission        | 30 % (technical quality)<br>20 % (documentation) |
| Oral presentation       | 20 %                                             |
| Project total           | 70 %                                             |

No questions about the project development/results will be answered to students that do not take part in the intermediate submission.