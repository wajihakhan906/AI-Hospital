# Datasets

No patient data is stored in this repository. Public datasets suitable for each modality:

| Modality | Dataset | Annotations |
|---|---|---|
| Chest X-ray | [VinDr-CXR](https://physionet.org/content/vindr-cxr/), [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) | boxes (VinDr), labels |
| Dental CBCT | [ToothFairy](https://toothfairy.grand-challenge.org/) | voxel masks |
| CT | [DeepLesion](https://nihcc.app.box.com/v/DeepLesion), [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) | boxes, nodule contours |
| MRI | [BraTS](https://www.synapse.org/brats) | tumor masks |
| Ultrasound | [BUSI](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) | lesion masks |
| PET-CT | [AutoPET](https://autopet.grand-challenge.org/) | lesion masks |
| Mammography | [CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/), [VinDr-Mammo](https://physionet.org/content/vindr-mammo/) | ROI masks, boxes |
| Angiography | [ARCADE](https://zenodo.org/records/10390295) | stenosis / vessel masks |

Convert box annotations to YOLO format (`class cx cy w h`, normalised) per modality:
```
Dataset/<modality>/images/{train,val}/*.png
Dataset/<modality>/labels/{train,val}/*.txt
Dataset/<modality>/<modality>.yaml
```
Access to several of these datasets requires a data-use agreement (PhysioNet credentialing, TCIA terms).
