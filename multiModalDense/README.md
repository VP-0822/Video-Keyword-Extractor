Sample Model (Best performing as per cited paper).

Implementation relies https://github.com/v-iashin/MDVC and http://nlp.seas.harvard.edu/2018/04/03/attention.html.
 
```
@InProceedings{MDVC_Iashin_2020,
  author = {Iashin, Vladimir and Rahtu, Esa},
  title = {Multi-modal Dense Video Captioning},
  booktitle = {Workshop on Multimodal Learning (CVPR Workshop)},
  year = {2020}
}
```

Packages
- pip install torchtext
- pip install spacy
- python -m spacy download en
- pip install tensorboard
- pip install torchsummary
- pip install torchvision
- pip install pandas
- pip install h5py
- conda install pytorch torchvision -c soumith (For CPU only pytorch)

Git Submodules:
https://github.com/salaniz/pycocoevalcap/tree/ca1b05fa0e99f86de2259f87d43c72b322523731

After cloning use command: git submodule update --init