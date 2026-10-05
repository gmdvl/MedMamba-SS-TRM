### I. Explainability: What the Spectral Pathway Attends To

The band gate (10) weights every band of every patch before pooling, so its values show which wavelengths the pathway emphasizes. Fig. {{F:xai}}(a) averages them over 9,000 class-stratified test patches for the seed-42 hyperspectral model. The gate is not uniform. It rises from about 0.55 below 500 nm to a maximum at 577 nm and stays high to 633 nm, then falls to about 0.54 in the near-infrared block. The maximum is highest for DCIS (0.67 at 577 nm), then IDC (0.65) and healthy tissue (0.61), and its between-class spread is largest in the same 535–633 nm region where the input spectra differ most (Fig. {{F:xai}}(b)): at 535 nm healthy tissue reflects 0.17 above the overall mean and DCIS 0.15 below it, in the range where the haematoxylin and eosin stains absorb. The model has therefore learned, without supervision on bands, to weight the part of the spectrum that separates the classes, and to weight it most for the minority class whose separation from healthy tissue drives the modality effect. The gate is a learned weighting rather than a causal attribution, and the model still receives every band.

![Fig. xai](../figures/results/fig_xai_bands.png)

*Fig. {{F:xai}}. (a) Band-gate weights $a_c$ of (10), averaged within each patch, for 9,000 class-stratified test patches (3,000 per class). (b) Difference between each class's mean spectrum and the overall mean for the same patches.*

Fig. {{F:tsne}} embeds the pooled answer state, the vector the classifier reads, for the same patches with t-SNE [@tsne]. With 32 bands the three classes form three separate groups, healthy and DCIS adjacent but distinct. With 3 bands IDC stays separate, while healthy and DCIS patches intermix along a shared boundary, which is the confusion that Fig. {{F:error}} counts (27.5 % of healthy patches predicted DCIS, against 14.3 % with 32 bands).

![Fig. tsne](../figures/results/fig_tsne.png)

*Fig. {{F:tsne}}. t-SNE of the pooled answer state for 9,000 class-stratified test patches (seed 42). (a) 32-band model. (b) 3-band model.*
