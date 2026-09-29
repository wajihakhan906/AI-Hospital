# Model weights

Place trained weights here (not tracked in git). Expected file names are configured in
`ai_hospital/modalities.py`:

| Modality | Detector (Ultralytics YOLO) | Segmenter (TorchScript U-Net, optional) |
|---|---|---|
| X-ray | `xray_yolo.pt` | `xray_unet.pt` |
| CBCT | `cbct_yolo.pt` | `cbct_unet.pt` |
| CT | `ct_yolo.pt` | `ct_unet.pt` |
| MRI | `mri_yolo.pt` | `mri_unet.pt` |
| Ultrasound | `us_yolo.pt` | `us_unet.pt` |
| PET-CT | `petct_yolo.pt` | – |
| Mammography | `mammo_yolo.pt` | `mammo_unet.pt` |
| Angiography | `angio_yolo.pt` | `angio_unet.pt` |

Train a detector with Ultralytics, e.g. `yolo detect train data=xray.yaml model=yolov8s.pt imgsz=1024`,
and export a U-Net with `torch.jit.script(model).save("xray_unet.pt")`.
