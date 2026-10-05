#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Write the thebibliography block of main.tex in order of first citation.

The SAI/CVC template numbers references sequentially "by order of citation or by
alphabetical order"; we use citation order. Entries are in Springer LNCS style,
transcribed from the references of paper/draft/MedMamba-SS-TRM_manuscript_v18.md.
Author lists longer than six are cut to six plus "et al." (template rule); where the
v18 source itself gives only "First et al.", that form is kept rather than guessed.

    python build_bibliography.py      # rewrites the block between %BIB-BEGIN and %BIB-END
"""
import re
import sys
from pathlib import Path

TEX = Path(__file__).with_name("main.tex")


def doi(d):
    return rf"\url{{https://doi.org/{d}}}"


REFS = {
    "medmamba": r"Yue, Y., Li, Z.: MedMamba: Vision Mamba for medical image classification. arXiv:2403.03849 (2024)",
    "trm": r"Jolicoeur-Martineau, A.: Less is more: recursive reasoning with tiny networks. arXiv:2510.04871 (2025)",
    "lu2014": r"Lu, G., Fei, B.: Medical hyperspectral imaging: a review. J. Biomed. Opt. \textbf{19}(1), 010901 (2014). " + doi("10.1117/1.JBO.19.1.010901"),
    "pandey2026": r"Pandey, A., Ahmed, S.B.: Hyperspectral melanoma segmentation using SLIC-derived pseudo-labels. In: Proceedings of the Computer Vision Conference (CVC) 2026, Vol.~3. LNNS, pp. 171--183. Springer, Cham (2026). " + doi("10.1007/978-3-032-26217-2_14"),
    "fatima2025": r"Fatima, S., Akram, M.U., Mohammad, S., Ahmed, S.B.: Deep learning in dermatopathology: applications for skin disease diagnosis and classification. Discover Appl. Sci. \textbf{7}, 1006 (2025). " + doi("10.1007/s42452-025-07138-3"),
    "channelvit": r"Bao, Y., Sivanandan, S., Karaletsos, T.: Channel vision transformers: an image is worth 1\,$\times$\,16\,$\times$\,16 words. In: International Conference on Learning Representations (ICLR) (2024). \url{https://openreview.net/forum?id=CK5Hfb5hBG}",
    "dofa": r"Xiong, Z., et al.: Neural plasticity-inspired multimodal foundation model for Earth observation. arXiv:2403.15356 (2024)",
    "batformer": r"Shahi, N.J., Ahmed, S.B.: Band-aware transformer (BAT-Former) for general image understanding. In: Proceedings of the Computer Vision Conference (CVC) 2026, Vol.~3. LNNS, pp. 184--200. Springer, Cham (2026). " + doi("10.1007/978-3-032-26217-2_15"),
    "carl": r"Baumann, A., Ayala, L., Seidlitz, S., Sellner, J., Studier-Fischer, A., \"{O}zdemir, B., et al.: CARL: camera-agnostic representation learning for spectral image analysis. arXiv:2504.19223 (2025)",
    "film": r"Perez, E., Strub, F., de~Vries, H., Dumoulin, V., Courville, A.: FiLM: visual reasoning with a general conditioning layer. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol.~32 (2018). " + doi("10.1609/aaai.v32i1.11671"),
    "resnet": r"He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770--778 (2016). " + doi("10.1109/CVPR.2016.90"),
    "convnext": r"Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., Xie, S.: A ConvNet for the 2020s. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11966--11976 (2022). " + doi("10.1109/CVPR52688.2022.01167"),
    "vit": r"Dosovitskiy, A., et al.: An image is worth 16\,$\times$\,16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (ICLR) (2021). \url{https://openreview.net/forum?id=YicbFdNTTy}",
    "swin": r"Liu, Z., et al.: Swin Transformer: hierarchical vision transformer using shifted windows. In: IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9992--10002 (2021). " + doi("10.1109/ICCV48922.2021.00986"),
    "mamba": r"Gu, A., Dao, T.: Mamba: linear-time sequence modeling with selective state spaces. In: Conference on Language Modeling (COLM) (2024). \url{https://openreview.net/forum?id=tEYskw1VY2}",
    "vim": r"Zhu, L., et al.: Vision Mamba: efficient visual representation learning with bidirectional state space model. In: International Conference on Machine Learning (ICML). PMLR, vol.~235, pp. 62429--62442 (2024)",
    "vmamba": r"Liu, Y., et al.: VMamba: visual state space model. In: Advances in Neural Information Processing Systems (NeurIPS), vol.~37, pp. 103031--103063 (2024)",
    "khan2026": r"Khan, M.S., Atoofian, E., Ahmed, S.B.: Quantum enchanced [sic] multi-scale CNN with bi-directional Mamba for crop field analysis. arXiv:2606.17222 (2026)",
    "melanospec": r"Zhao, Q., Lei, S., Tian, C., Li, W.: Patient-level hyperspectral state-space learning for melanoma pathology diagnosis. Photodiagnosis Photodyn. Ther. \textbf{61}, 105642 (2026). " + doi("10.1016/j.pdpdt.2026.105642"),
    "ut": r"Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., Kaiser, \L.: Universal Transformers. In: International Conference on Learning Representations (ICLR) (2019). \url{https://openreview.net/forum?id=HyzdRiR9Y7}",
    "albert": r"Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., Soricut, R.: ALBERT: a lite BERT for self-supervised learning of language representations. In: International Conference on Learning Representations (ICLR) (2020). \url{https://openreview.net/forum?id=H1eA7AEtvS}",
    "deq": r"Bai, S., Kolter, J.Z., Koltun, V.: Deep equilibrium models. In: Advances in Neural Information Processing Systems (NeurIPS), vol.~32 (2019)",
    "act": r"Graves, A.: Adaptive computation time for recurrent neural networks. arXiv:1603.08983 (2016)",
    "hrm": r"Wang, G., et al.: Hierarchical reasoning model. arXiv:2506.21734 (2025)",
    "hybridsn": r"Roy, S.K., Krishna, G., Dubey, S.R., Chaudhuri, B.B.: HybridSN: exploring 3-D--2-D CNN feature hierarchy for hyperspectral image classification. IEEE Geosci. Remote Sens. Lett. \textbf{17}(2), 277--281 (2020). " + doi("10.1109/LGRS.2019.2918719"),
    "spectralformer": r"Hong, D., et al.: SpectralFormer: rethinking hyperspectral image classification with Transformers. IEEE Trans. Geosci. Remote Sens. \textbf{60}, 5518615 (2022). " + doi("10.1109/TGRS.2021.3130716"),
    "quantformer": r"Ahmed, S.B., Lunia, J.: QuantFormer: a hybrid quantum classical transformer for hyperspectral image classification. In: Proceedings of the 39th Canadian Conference on Artificial Intelligence. PMLR, vol.~318, pp. 103--114 (2026)",
    "fatima2026": r"Fatima, S., Salam, A.A., Akram, M.U., Hameed, I.A., Ahmed, S.B.: Incremental learning approach for semantic segmentation of skin histology images. Sci. Rep. \textbf{16}, 9593 (2026). " + doi("10.1038/s41598-025-31553-6"),
    "ortega2020gbm": r"Ortega, S., Halicek, M., Fabelo, H., Camacho, R., de~la Luz Plaza, M., Godtliebsen, F., et al.: Hyperspectral imaging for the detection of glioblastoma tumor cells in H\&E slides using convolutional neural networks. Sensors \textbf{20}(7), 1911 (2020). " + doi("10.3390/s20071911"),
    "ortega2020breast": r"Ortega, S., Halicek, M., Fabelo, H., Guerra, R., Lopez, C., Lejaune, M., et al.: Hyperspectral imaging and deep learning for the detection of breast cancer cells in digitized histological images. In: Medical Imaging 2020: Digital Pathology. Proc. SPIE, vol.~11320, 113200V (2020). " + doi("10.1117/12.2548609"),
    "spectralgpt": r"Hong, D., et al.: SpectralGPT: spectral remote sensing foundation model. IEEE Trans. Pattern Anal. Mach. Intell. \textbf{46}(8), 5227--5244 (2024). " + doi("10.1109/TPAMI.2024.3362475"),
    "hypersigma": r"Wang, D., et al.: HyperSIGMA: hyperspectral intelligence comprehension foundation model. IEEE Trans. Pattern Anal. Mach. Intell. \textbf{47}(8), 6427--6444 (2025). " + doi("10.1109/TPAMI.2025.3557581"),
    "hsibc_data": r"Quintana Quintana, L., et al.: Recurrent breast cancer: histopathological and hyperspectral images database (HistologyHSI-BC-Recurrence), version~1 [data set]. The Cancer Imaging Archive (2025). " + doi("10.7937/6KPY-YT49"),
    "hsibc_desc": r"Quintana-Quintana, L., et al.: Histological hyperspectral breast cancer recurrence database (HistologyHSI-BC Recurrence). Sci. Data \textbf{12}, 1886 (2025). " + doi("10.1038/s41597-025-06157-4"),
    "sklearn": r"Pedregosa, F., et al.: Scikit-learn: machine learning in Python. J. Mach. Learn. Res. \textbf{12}, 2825--2830 (2011)",
    "pad_desc": r"Pacheco, A.G.C., et al.: PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones. Data Brief \textbf{32}, 106221 (2020). " + doi("10.1016/j.dib.2020.106221"),
    "pad_data": r"Pacheco, A.G.C., et al.: PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones, version~1 [data set]. Mendeley Data (2020). " + doi("10.17632/zr7vgbcyr2.1"),
    "glu": r"Shazeer, N.: GLU variants improve Transformer. arXiv:2002.05202 (2020)",
    "rmsnorm": r"Zhang, B., Sennrich, R.: Root mean square layer normalization. In: Advances in Neural Information Processing Systems (NeurIPS), vol.~32 (2019)",
    "senet": r"Hu, J., Shen, L., Sun, G.: Squeeze-and-excitation networks. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7132--7141 (2018). " + doi("10.1109/CVPR.2018.00745"),
    "layerscale": r"Touvron, H., Cord, M., Sablayrolles, A., Synnaeve, G., J\'{e}gou, H.: Going deeper with image transformers. In: IEEE/CVF International Conference on Computer Vision (ICCV), pp. 32--42 (2021). " + doi("10.1109/ICCV48922.2021.00010"),
    "checkpointing": r"Chen, T., Xu, B., Zhang, C., Guestrin, C.: Training deep nets with sublinear memory cost. arXiv:1604.06174 (2016)",
    "focal": r"Lin, T.-Y., Goyal, P., Girshick, R., He, K., Doll\'{a}r, P.: Focal loss for dense object detection. In: IEEE International Conference on Computer Vision (ICCV), pp. 2999--3007 (2017). " + doi("10.1109/ICCV.2017.324"),
    "naeini": r"Naeini, M.P., Cooper, G., Hauskrecht, M.: Obtaining well calibrated probabilities using Bayesian binning. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol.~29, pp. 2901--2907 (2015). " + doi("10.1609/aaai.v29i1.9602"),
    "guo2017": r"Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On calibration of modern neural networks. In: International Conference on Machine Learning (ICML). PMLR, vol.~70, pp. 1321--1330 (2017)",
    "brier": r"Brier, G.W.: Verification of forecasts expressed in terms of probability. Mon. Weather Rev. \textbf{78}(1), 1--3 (1950). " + doi("10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2"),
    "pytorch": r"Paszke, A., et al.: PyTorch: an imperative style, high-performance deep learning library. In: Advances in Neural Information Processing Systems (NeurIPS), vol.~32, pp. 8024--8035 (2019)",
    "adamw": r"Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (ICLR) (2019). \url{https://openreview.net/forum?id=Bkg6RiCqY7}",
    "geifman2017": r"Geifman, Y., El-Yaniv, R.: Selective classification for deep neural networks. In: Advances in Neural Information Processing Systems (NeurIPS), vol.~30, pp. 4878--4887 (2017)",
    "medformer": r"Sibhai, M.M., Alkhateeb, A., Ahmed, S.B.: MedFormer-UR: uncertainty-routed transformer for medical image classification. arXiv:2604.08868 (2026)",
    "aurc": r"Geifman, Y., Uziel, G., El-Yaniv, R.: Bias-reduced uncertainty estimation for deep neural classifiers. In: International Conference on Learning Representations (ICLR) (2019). \url{https://openreview.net/forum?id=SJfb5jCqKm}",
    "ilse": r"Ilse, M., Tomczak, J.M., Welling, M.: Attention-based deep multiple instance learning. In: International Conference on Machine Learning (ICML). PMLR, vol.~80, pp. 2127--2136 (2018)",
    "medmamba_repo": r"Yue, Y., Li, Z.: MedMamba. GitHub repository (2024). \url{https://github.com/YubiaoYue/MedMamba}",
    "elmore": r"Elmore, J.G., et al.: Diagnostic concordance among pathologists interpreting breast biopsy specimens. JAMA \textbf{313}(11), 1122--1132 (2015). " + doi("10.1001/jama.2015.1405"),
    "hmilusc": r"Yan, Z., Huang, H., Guo, Y., Shi, J., Geng, R., Zhang, J., et al.: HMI-LUSC: a histological hyperspectral imaging dataset for lung squamous cell carcinoma. Sci. Data \textbf{13}, 415 (2026). " + doi("10.1038/s41597-026-06766-7"),
    "trm_repo": r"Jolicoeur-Martineau, A.: TinyRecursiveModels. GitHub repository (2025). \url{https://github.com/SamsungSAILMontreal/TinyRecursiveModels}",
    "tsne": r"van~der Maaten, L., Hinton, G.: Visualizing data using t-SNE. J. Mach. Learn. Res. \textbf{9}, 2579--2605 (2008)",
}


def main():
    tex = TEX.read_text()
    head, rest = tex.split("%BIB-BEGIN", 1)
    _, tail = rest.split("%BIB-END", 1)
    order = []
    for grp in re.findall(r"\\cite\{([^}]*)\}", head):
        for k in (x.strip() for x in grp.split(",")):
            if k not in order:
                order.append(k)
    missing = [k for k in order if k not in REFS]
    unused = [k for k in REFS if k not in order]
    if missing:
        sys.exit(f"cited but not defined: {missing}")
    items = "\n\n".join(rf"\bibitem{{{k}}}" + "\n" + REFS[k] for k in order)
    block = ("%BIB-BEGIN\n\\begin{thebibliography}{" + str(len(order)) + "}\n\n"
             + items + "\n\n\\end{thebibliography}\n%BIB-END")
    TEX.write_text(head + block + tail)
    print(f"{len(order)} references written in citation order")
    if unused:
        print(f"defined but not cited (omitted): {unused}")


if __name__ == "__main__":
    main()
