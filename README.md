# DeepOCSORT module for Detectron 2

This is an implementation of DeepOCSORT for Detectron 2 that is heavily based on [mikel-brostorm's DeepOCSORT implementation for YOLO](https://github.com/mikel-brostrom/yolo_tracking), but modified somewhat to adapt to the tracking system in Detectron 2.

To use, after cloning this repo using

```git
git clone git@github.com:remineneko/DeepOCSORT-detectron2.git
```

follow facebook's [instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) on how to install Detectron 2,

then this repo can be used normally for video inference, similar to how [video inference is normally done on Detectron 2](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html).

Current plans:

- [ ] Identify remaining bugs and issues while updating Instances.

- [ ] Write a test suite for this tracker. 


