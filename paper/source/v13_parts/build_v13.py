"""Build the v13 manuscript from its parts.

    python paper/source/v13_parts/build_v13.py [output.md]

Concatenates p*.md in name order, numbers citations ([@key]) by first
appearance (IEEE), tables ({{T:key}}) in Roman and figures ({{F:key}}) in
Arabic by first appearance, and appends the reference list. Edit the p*.md
parts, never the output. Default output:
paper/draft/MedMamba-SS-TRM_manuscript_v13.md
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper/draft/MedMamba-SS-TRM_manuscript_v13.md"

REFS = {
    "medmamba": 'Y. Yue and Z. Li, "MedMamba: Vision Mamba for medical image classification," arXiv:2403.03849, 2024. [Online]. Available: https://arxiv.org/abs/2403.03849',
    "trm": 'A. Jolicoeur-Martineau, "Less is more: Recursive reasoning with tiny networks," arXiv:2510.04871, Oct. 2025. [Online]. Available: https://arxiv.org/abs/2510.04871',
    "resnet": 'K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.',
    "vit": 'A. Dosovitskiy et al., "An image is worth 16×16 words: Transformers for image recognition at scale," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021. [Online]. Available: https://openreview.net/forum?id=YicbFdNTTy',
    "hsi_review": 'G. Lu and B. Fei, "Medical hyperspectral imaging: A review," *J. Biomed. Opt.*, vol. 19, no. 1, Art. no. 010901, Jan. 2014, doi: 10.1117/1.JBO.19.1.010901.',
    "mamba": 'A. Gu and T. Dao, "Mamba: Linear-time sequence modeling with selective state spaces," in *Proc. Conf. Lang. Model. (COLM)*, 2024. [Online]. Available: https://openreview.net/forum?id=tEYskw1VY2',
    "vim": 'L. Zhu et al., "Vision Mamba: Efficient visual representation learning with bidirectional state space model," in *Proc. Int. Conf. Mach. Learn. (ICML)*, ser. Proc. Mach. Learn. Res., vol. 235, 2024, pp. 62429–62442. [Online]. Available: https://proceedings.mlr.press/v235/zhu24f.html',
    "vmamba": 'Y. Liu et al., "VMamba: Visual state space model," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 37, 2024, pp. 103031–103063. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2024/hash/baa2da9ae4bfed26520bb61d259a3653-Abstract-Conference.html',
    "film": 'E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, "FiLM: Visual reasoning with a general conditioning layer," in *Proc. AAAI Conf. Artif. Intell.*, vol. 32, no. 1, 2018, doi: 10.1609/aaai.v32i1.11671.',
    "convnext": 'Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 11966–11976, doi: 10.1109/CVPR52688.2022.01167.',
    "swin": 'Z. Liu et al., "Swin Transformer: Hierarchical vision transformer using shifted windows," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2021, pp. 9992–10002, doi: 10.1109/ICCV48922.2021.00986.',
    "medmamba_code": 'Y. Yue and Z. Li, "MedMamba," GitHub repository, 2024. [Online]. Available: https://github.com/YubiaoYue/MedMamba',
    "hrm": 'G. Wang et al., "Hierarchical reasoning model," arXiv:2506.21734, 2025. [Online]. Available: https://arxiv.org/abs/2506.21734',
    "ut": 'M. Dehghani, S. Gouws, O. Vinyals, J. Uszkoreit, and Ł. Kaiser, "Universal Transformers," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019. [Online]. Available: https://openreview.net/forum?id=HyzdRiR9Y7',
    "albert": 'Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, and R. Soricut, "ALBERT: A lite BERT for self-supervised learning of language representations," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2020. [Online]. Available: https://openreview.net/forum?id=H1eA7AEtvS',
    "deq": 'S. Bai, J. Z. Kolter, and V. Koltun, "Deep equilibrium models," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 32, 2019. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html',
    "act": 'A. Graves, "Adaptive computation time for recurrent neural networks," arXiv:1603.08983, 2016. [Online]. Available: https://arxiv.org/abs/1603.08983',
    "hybridsn": 'S. K. Roy, G. Krishna, S. R. Dubey, and B. B. Chaudhuri, "HybridSN: Exploring 3-D–2-D CNN feature hierarchy for hyperspectral image classification," *IEEE Geosci. Remote Sens. Lett.*, vol. 17, no. 2, pp. 277–281, Feb. 2020, doi: 10.1109/LGRS.2019.2918719.',
    "spectralformer": 'D. Hong et al., "SpectralFormer: Rethinking hyperspectral image classification with Transformers," *IEEE Trans. Geosci. Remote Sens.*, vol. 60, Art. no. 5518615, 2022, doi: 10.1109/TGRS.2021.3130716.',
    "se": 'J. Hu, L. Shen, and G. Sun, "Squeeze-and-excitation networks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 7132–7141, doi: 10.1109/CVPR.2018.00745.',
    "glu": 'N. Shazeer, "GLU variants improve Transformer," arXiv:2002.05202, 2020. [Online]. Available: https://arxiv.org/abs/2002.05202',
    "rmsnorm": 'B. Zhang and R. Sennrich, "Root mean square layer normalization," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 32, 2019. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html',
    "focal": 'T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 2999–3007, doi: 10.1109/ICCV.2017.324.',
    "checkpoint": 'T. Chen, B. Xu, C. Zhang, and C. Guestrin, "Training deep nets with sublinear memory cost," arXiv:1604.06174, 2016. [Online]. Available: https://arxiv.org/abs/1604.06174',
    "hsibc_data": 'L. Quintana Quintana et al., "Recurrent breast cancer: Histopathological and hyperspectral images database (HistologyHSI-BC-Recurrence)," Version 1 [Data set], The Cancer Imaging Archive, 2025. [Online]. Available: https://doi.org/10.7937/6KPY-YT49',
    "hsibc_paper": 'L. Quintana-Quintana et al., "Histological hyperspectral breast cancer recurrence database (HistologyHSI-BC Recurrence)," *Sci. Data*, vol. 12, Art. no. 1886, 2025, doi: 10.1038/s41597-025-06157-4.',
    "pad": 'A. G. C. Pacheco et al., "PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones," *Data Brief*, vol. 32, Art. no. 106221, Oct. 2020, doi: 10.1016/j.dib.2020.106221.',
    "pad_data": 'A. G. C. Pacheco et al., "PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones," Version 1 [Data set], Mendeley Data, 2020. [Online]. Available: https://doi.org/10.17632/zr7vgbcyr2.1',
    "pytorch": 'A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 32, 2019, pp. 8024–8035. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html',
    "adamw": 'I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019. [Online]. Available: https://openreview.net/forum?id=Bkg6RiCqY7',
    "sklearn": 'F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011. [Online]. Available: https://jmlr.org/papers/v12/pedregosa11a.html',
    "mil": 'M. Ilse, J. M. Tomczak, and M. Welling, "Attention-based deep multiple instance learning," in *Proc. Int. Conf. Mach. Learn. (ICML)*, ser. Proc. Mach. Learn. Res., vol. 80, 2018, pp. 2127–2136. [Online]. Available: https://proceedings.mlr.press/v80/ilse18a.html',
    "trm_code": 'A. Jolicoeur-Martineau, "TinyRecursiveModels," GitHub repository, 2025. [Online]. Available: https://github.com/SamsungSAILMontreal/TinyRecursiveModels',
    "tsne": 'L. van der Maaten and G. Hinton, "Visualizing data using t-SNE," *J. Mach. Learn. Res.*, vol. 9, pp. 2579–2605, 2008. [Online]. Available: https://jmlr.org/papers/v9/vandermaaten08a.html',
    "layerscale": 'H. Touvron, M. Cord, A. Sablayrolles, G. Synnaeve, and H. Jégou, "Going deeper with image transformers," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2021, pp. 32–42, doi: 10.1109/ICCV48922.2021.00010.',
    'channelvit': 'Y. Bao, S. Sivanandan, and T. Karaletsos, "Channel vision transformers: An image is worth 1 × 16 × 16 words," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2024. [Online]. Available: https://openreview.net/forum?id=CK5Hfb5hBG',
    'dofa': 'Z. Xiong et al., "Neural plasticity-inspired multimodal foundation model for Earth observation," arXiv:2403.15356, 2024. [Online]. Available: https://arxiv.org/abs/2403.15356',
    'batformer': 'N. J. Shahi and S. B. Ahmed, "Band-aware transformer (BAT-Former) for general image understanding," in *Proc. Comput. Vis. Conf. (CVC) 2026, Vol. 3*, ser. Lecture Notes in Networks and Systems. Springer, 2026, pp. 184–200, doi: 10.1007/978-3-032-26217-2_15.',
    'hsi_melanoma': 'A. Pandey and S. B. Ahmed, "Hyperspectral melanoma segmentation using SLIC-derived pseudo-labels," in *Proc. Comput. Vis. Conf. (CVC) 2026, Vol. 3*, ser. Lecture Notes in Networks and Systems. Springer, 2026, pp. 171–183, doi: 10.1007/978-3-032-26217-2_14.',
    'quantformer': 'J. V. Lunia and S. B. Ahmed, "QuantFormer: A hybrid quantum classical transformer for hyperspectral image classification," in *Proc. 39th Can. Conf. Artif. Intell.*, ser. Proc. Mach. Learn. Res., vol. 318, 2026, pp. 103–114. [Online]. Available: https://proceedings.mlr.press/v318/ahmed26a.html',
    'patchgraph': 'J. Lunia and S. B. Ahmed, "PatchGraph-MTFormer: A multitask patch-graph transformer for hyperspectral image analysis," TechRxiv, 2025, doi: 10.36227/techrxiv.176617104.42612822/v1.',
    'medformer_ur': 'M. M. Sibhai, A. Alkhateeb, and S. B. Ahmed, "MedFormer-UR: Uncertainty-routed transformer for medical image classification," arXiv:2604.08868, 2026. [Online]. Available: https://arxiv.org/abs/2604.08868',
    'crop_mamba': 'M. S. Khan, E. Atoofian, and S. B. Ahmed, "Quantum enchanced multi-scale CNN with bi-directional Mamba for crop field analysis," arXiv:2606.17222, 2026. [Online]. Available: https://arxiv.org/abs/2606.17222',
    'melanospec': 'Q. Zhao, S. Lei, C. Tian, and W. Li, "Patient-level hyperspectral state-space learning for melanoma pathology diagnosis," *Photodiagnosis Photodyn. Ther.*, vol. 61, Art. no. 105642, 2026, doi: 10.1016/j.pdpdt.2026.105642.',
}

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV",
         "XVI", "XVII", "XVIII", "XIX", "XX"]

text = "".join(p.read_text() for p in sorted(HERE.glob("p*.md")))

CITE = re.compile(r"\[@(\w+)((?:, [^\]]*)?)\]")
order = {}
for m in CITE.finditer(text):
    order.setdefault(m.group(1), len(order) + 1)
missing = [k for k in order if k not in REFS]
if missing:
    sys.exit(f"undefined citation keys: {missing}")
unused = [k for k in REFS if k not in order]
text = CITE.sub(lambda m: f"[{order[m.group(1)]}{m.group(2)}]", text)

for tag, fmt in (("T", lambda i: ROMAN[i]), ("F", lambda i: str(i + 1))):
    pat = re.compile(r"\{\{" + tag + r":(\w+)\}\}")
    seen = {}
    for m in pat.finditer(text):
        seen.setdefault(m.group(1), len(seen))
    text = pat.sub(lambda m: fmt(seen[m.group(1)]), text)
    print(f"{tag}: {len(seen)} ->", ", ".join(seen))

leftover = re.findall(r"\{\{[^}]*\}\}|\[@[^\]]*\]", text)
if leftover:
    sys.exit(f"unresolved placeholders: {leftover}")
refs = "\n\n".join(f"[{n}] {REFS[k]}" for k, n in sorted(order.items(), key=lambda kv: kv[1]))
OUT.write_text(text.rstrip() + "\n\n" + refs + "\n")
body = re.sub(r"```.*?```", "", text, flags=re.S)
print(f"citations: {len(order)} (unused keys: {unused or 'none'}); prose words: {len(re.findall(r'[A-Za-z][A-Za-z-]*', body))}")
print(f"[built] {OUT}")
