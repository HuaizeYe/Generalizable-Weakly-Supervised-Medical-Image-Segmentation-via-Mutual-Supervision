# Generalizable-Weakly-Supervised-Medical-Image-Segmentation-via-Mutual-Supervision

### Although this work has been temporarily terminated, it still has medical application value, so the code and article are retained here(New work of single domain generalization with SAM will be uploaded in Dec. 2023)

Domain generalization, which aims to reduce domain shift between domains to achieve promising performance on the unseen target domain, has been widely practiced in medical image segmentation. Although this technique has made considerable progress, existing methods rely on fully-supervised methods with pixel-wise labels, which consume significant manpower for annotation. In this paper, we introduce the concept of generalizable weakly-supervised segmentation and train the model with bounding-box annotations of source domains only. To address this task, we propose a model with a dual-branch augmentation and mutual supervision structure. The source images are augmented in two different strategies and sent to two segmentation networks, respectively, which enrich the source domains and force the network to learn different views of augmented features. The prediction from the network is used to teach the other one through mutual supervision for knowledge sharing, promoting the performance of generalization. We evaluate our model on several medical image segmentation tasks and achieve competitive results compared to its upper-bound, i.e., fully-supervised domain generalization methods.

Paper: https://github.com/HuaizeYe/Generalizable-Weakly-Supervised-Medical-Image-Segmentation-via-Mutual-Supervision/blob/main/HuaizeyeWeakly_Supervised_Domain_Generalization_for_Medical__Image_Segmentation_with_Cross_Learning.pdf

## Train
### Fundus Dataset
```Linux
bash fundus_train.sh 
```

### Prostate Dataset
```Linux
bash prostate_train.sh 
```
This base code is provided by Ziqi Zhou Nanjing University. https://github.com/zzzqzhou/RAM-DSIR.
