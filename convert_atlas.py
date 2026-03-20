import SimpleITK as sitk

for src, dst in [
    ("/SSD2/mice_mri/mouse_atlas/allen_ccf25/average_template_25.nrrd", "/SSD2/mice_mri/mouse_atlas/allen_ccf25/average_template_25.nii.gz"),
    ("/SSD2/mice_mri/mouse_atlas/allen_ccf25/annotation_25.nrrd", "/SSD2/mice_mri/mouse_atlas/allen_ccf25/annotation_25.nii.gz"),
]:
    img = sitk.ReadImage(src)
    sitk.WriteImage(img, dst)