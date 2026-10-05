"""Build the v12 manuscript from its parts.

    python paper/source/v12_parts/build_v12.py [output.md]

Concatenates p*.md in name order, numbers citations ([@key]) by first
appearance (IEEE), tables ({{T:key}}) in Roman and figures ({{F:key}}) in
Arabic by first appearance, and appends the reference list. Edit the p*.md
parts, never the output. Default output:
paper/draft/MedMamba-SS-TRM_manuscript_v12.md
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper/draft/MedMamba-SS-TRM_manuscript_v12.md"

REFS = {
    "medmamba": 'Y. Yue and Z. Li, "MedMamba: Vision Mamba for medical image classification," arXiv:2403.03849, 2024. [Online]. Available: https://arxiv.org/pdf/2403.03849',
    "trm": 'A. Jolicoeur-Martineau, "Less is more: Recursive reasoning with tiny networks," arXiv:2510.04871, Oct. 2025. [Online]. Available: https://arxiv.org/pdf/2510.04871',
    "resnet": 'K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778.',
    "vit": 'A. Dosovitskiy et al., "An image is worth 16×16 words: Transformers for image recognition at scale," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.',
    "hsi_review": 'G. Lu and B. Fei, "Medical hyperspectral imaging: A review," *J. Biomed. Opt.*, vol. 19, no. 1, Art. no. 010901, Jan. 2014, doi: 10.1117/1.JBO.19.1.010901.',
    "mamba": 'A. Gu and T. Dao, "Mamba: Linear-time sequence modeling with selective state spaces," arXiv:2312.00752, 2023.',
    "vim": 'L. Zhu et al., "Vision Mamba: Efficient visual representation learning with bidirectional state space model," arXiv:2401.09417, 2024.',
    "vmamba": 'Y. Liu et al., "VMamba: Visual state space model," arXiv:2401.10166, 2024.',
    "film": 'E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, "FiLM: Visual reasoning with a general conditioning layer," in *Proc. AAAI Conf. Artif. Intell.*, 2018.',
    "convnext": 'Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 11976–11986.',
    "swin": 'Z. Liu et al., "Swin Transformer: Hierarchical vision transformer using shifted windows," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2021, pp. 10012–10022.',
    "medmamba_code": 'Y. Yue and Z. Li, "MedMamba," GitHub repository, 2024. [Online]. Available: https://github.com/YubiaoYue/MedMamba',
    "hrm": 'G. Wang et al., "Hierarchical reasoning model," arXiv:2506.21734, 2025.',
    "ut": 'M. Dehghani, S. Gouws, O. Vinyals, J. Uszkoreit, and Ł. Kaiser, "Universal Transformers," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019.',
    "albert": 'Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, and R. Soricut, "ALBERT: A lite BERT for self-supervised learning of language representations," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2020.',
    "deq": 'S. Bai, J. Z. Kolter, and V. Koltun, "Deep equilibrium models," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2019.',
    "act": 'A. Graves, "Adaptive computation time for recurrent neural networks," arXiv:1603.08983, 2016.',
    "hybridsn": 'S. K. Roy, G. Krishna, S. R. Dubey, and B. B. Chaudhuri, "HybridSN: Exploring 3-D–2-D CNN feature hierarchy for hyperspectral image classification," *IEEE Geosci. Remote Sens. Lett.*, vol. 17, no. 2, pp. 277–281, Feb. 2020.',
    "spectralformer": 'D. Hong et al., "SpectralFormer: Rethinking hyperspectral image classification with Transformers," *IEEE Trans. Geosci. Remote Sens.*, vol. 60, Art. no. 5518615, 2022.',
    "se": 'J. Hu, L. Shen, and G. Sun, "Squeeze-and-excitation networks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 7132–7141.',
    "glu": 'N. Shazeer, "GLU variants improve Transformer," arXiv:2002.05202, 2020.',
    "rmsnorm": 'B. Zhang and R. Sennrich, "Root mean square layer normalization," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2019.',
    "focal": 'T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 2980–2988.',
    "checkpoint": 'T. Chen, B. Xu, C. Zhang, and C. Guestrin, "Training deep nets with sublinear memory cost," arXiv:1604.06174, 2016.',
    "hsibc_data": 'L. Quintana-Quintana et al., "Recurrent breast cancer: Histopathological and hyperspectral images database (HistologyHSI-BC-Recurrence)," version 1, dataset, The Cancer Imaging Archive, 2025, doi: 10.7937/6KPY-YT49.',
    "hsibc_paper": 'L. Quintana-Quintana et al., "Histological hyperspectral breast cancer recurrence database (HistologyHSI-BC Recurrence)," *Sci. Data*, vol. 12, Art. no. 1886, 2025, doi: 10.1038/s41597-025-06157-4.',
    "pad": 'A. G. C. Pacheco et al., "PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones," *Data Brief*, vol. 32, Art. no. 106221, Oct. 2020, doi: 10.1016/j.dib.2020.106221.',
    "pytorch": 'A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2019, pp. 8024–8035.',
    "adamw": 'I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019.',
    "sklearn": 'F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.',
    "mil": 'M. Ilse, J. M. Tomczak, and M. Welling, "Attention-based deep multiple instance learning," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2018, pp. 2127–2136.',
    "trm_code": 'A. Jolicoeur-Martineau, "TinyRecursiveModels," GitHub repository, 2025. [Online]. Available: https://github.com/SamsungSAILMontreal/TinyRecursiveModels',
    "tsne": 'L. van der Maaten and G. Hinton, "Visualizing data using t-SNE," *J. Mach. Learn. Res.*, vol. 9, pp. 2579–2605, 2008.',
    "layerscale": 'H. Touvron, M. Cord, A. Sablayrolles, G. Synnaeve, and H. Jégou, "Going deeper with image transformers," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2021, pp. 32–42.',
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
