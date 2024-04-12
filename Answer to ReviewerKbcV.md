# Rebuttal to Reviewer KbcV

## Cons3:
While FRNet indeed draws upon established techniques such as FFT, mixer blocks, and patching, it innovatively introduces the "Frequency Rotation Module" – a novel component that uniquely integrates these methodologies into a unified lightweight framework, which is unprecedented in existing approaches. But simple stacking cannot achieve excellent performance

And our methodology diverges from standard practice in several technical aspects. 
1. FFT: Standard methods apply FFT once for static spectral info. We use FFT twice: first to find significant periods, then on each period individually, reveal their dynamic frequency evolution beyond static analysis. 
2. Mixer: We adapt mixers to complex values for frequency component interaction and prediction. Thus, our work transcends mere application of the individual techniques and contributes a distinct methodological advancement in the realm of time series analysis. 
3. Period-dependent patch strides size, rather than arbitrary size. 

## Cons4:
We've compared FRNet with PatchTST in Table 1. Since Contiformer specializes in irregular time series prediction (not our focus), we exclude it from our comparisons. To further evaluate FRNet, we now include experiments comparing it with MICN and Crossformer, in addition to the existing results: 

![MainResults](MainResults.png "MainResults")


## Cons5 & Cons6:
To validate that FRNet's superiority stems from its PFR module, we compare FRNet with three variants. 
#### V1: Predicting periodic components in the time domain without using FFT
#### V2: Static frequency-domain prediction, like a frequency-domain Linear 
#### V3: Ignoring both period dynamics and frequency-domain prediction. Results are below:

|  | length | FRNet | V1 | V2 | V3 |
| --- | --- | --- | --- | --- | --- |
| ETTH2 | 96 | **0.263** | 0.269 | 0.289 | 0.282 |
|| 192 | **0.325** | 0.331 | 0.344 | 0.350 | 
|| 336 | **0.320** | 0.332 | 0.343 | 0.410 | 
|| 720 | **0.377** | 0.378 | 0.414 | 0.587 | 
| ETTM1 | 96 | **0.287** | 0.310 | 0.327 | 0.299 |
|| 192 | **0.327** | 0.337 | 0.352 | 0.335 | 
|| 336 | **0.362** | 0.368 | 0.395 | 0.369 | 
|| 720 | **0.408** | 0.422 | 0.482 | 0.425 | 

It can be seen that removing the dynamic representation of the period in the frequency domain or not reshaping the time series for dynamic prediction will significantly reduce prediction performance.

## Q5:
Top-K amplitude periods selection is widely used and proven effective[1] [2] [3]. Higher amplitudes signify stronger periodic effects and form the time series' backbone. It neglects low-amplitude noise. Top-amplitude period selection is backed by energy concentration and maximum entropy concepts [4].

While there are some similarities between our method and existing techniques like FEDformer, such as operating in the frequency domain and subsequently projecting back into the time domain, several distinctive features set our approach apart:

(1). Component Selection Strategy: Unlike FEDformer, which selects a random subset of Fourier bases from the input sequence and encodes them jointly in a single branch, our method identifies and learns from the Q most salient periods across the entire dataset, each treated as a separate branch. This targeted selection process enhances the model's focus on the truly dominant periodic elements.

(2). Decomposition and Prediction Approach: FEDformer employs an incremental decomposition strategy, which has been shown to be suboptimal [5]. In contrast, our method opts for a more direct and efficient two-step process: first, decomposing the data into its constituent periodic components, followed by independent prediction for each component. This streamlined procedure allows for a more precise and focused treatment of individual periodic patterns.

(3). Architectural Design: FEDformer's core architecture relies on pointwise attention mechanisms, which have been demonstrated to hinder the performance of transformers. In contrast, our model adopts a complex-valued Mixer architecture. Not only is this design more effective in capturing the intricate relationships within time series data, but it also boasts a lighter computational footprint, contributing to the overall efficiency and scalability of our method.
[1] ICLR 2023, TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
[2] Periodicity Detection in Time Series Data: A Comprehensive Review. Journal of Systems Engineering and Electronics, 2019.
[3] ICLR 2024, Periodicity Decoupling Framework for Long-term Series Forecasting
[4] Principles of Digital Signal Processing: Theory, Algorithms, and Hardware Design. Oxford University Press, 2015.
[5] First De-Trend then Attend: Rethinking Attention for Time-Series Forecasting

