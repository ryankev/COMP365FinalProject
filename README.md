# Object Image Classifier: Transfer Learning vs. Training From Scratch
COMP 365 Individual Project, Final Submission. Ryan Kevena.

**Question.** On 32×32 CIFAR-10 images, how much does fine-tuning an ImageNet-pretrained ResNet-18 improve test
accuracy over the same network trained from scratch, how does the gap change from 500 to 5,000 training images,
and does fine-tuning beat simpler baselines (a frozen-feature linear probe and pixel logistic regression)?

All results in the paper come from `Kevena_COMP365_Checkpoint2_executed.ipynb` (executed, with all outputs shown)
and the result files listed below.

## Files in this repository
| File | What it is |
|---|---|
| `Kevena_COMP365_Checkpoint2_executed.ipynb` | Executed notebook with all outputs: data, 16 network runs (2 methods × 4 sizes × 2 seeds), 16 linear-baseline runs, bootstrap CIs, McNemar tests, failure analysis, web-app test |
| `app.py` | Gradio web app: upload a photo, get the top-3 predicted classes |
| `requirements.txt` | Dependencies (torch/torchvision versions as recorded in the run) |
| `cp2_summary_table.csv` | Mean ± std test accuracy (Table 2 in the paper) |
| `cp2_all_runs.csv`, `cp2_runs.csv` | Every run: accuracy, macro-F1, training time |
| `cp2_bootstrap_ci_n5000.csv`, `cp2_mcnemar_n5000.csv` | 95% bootstrap intervals and paired McNemar tests at 5,000 training images |
| `cp2_per_class_n5000.csv`, `cp2_top_confusions.csv` | Per-class accuracy and most common confusions |
| `cp2_webapp_test.csv` | Web-app API test on 50 test images |
| `cp2_results_figure.png`, `cp2_confident_mistakes.png` | Figures 1 and 2 in the paper |
| `cp2_example_input.png` | Example test image (ship) for trying the app |
| `cp2_meta.json` | Data source and example-output record from the run |
| `improved_pipeline_not_run/` | Improved experiment code written after feedback. **Not executed; no reported results come from it.** |

**Model weights** are too large for the repository and are attached to the release
**[1.0 – Model Weights](https://github.com/ryankev/COMP365FinalProject/releases/tag/1.0)**:
- `resnet18_finetune_n5000_seed0_cp2.pt`: Checkpoint 2 fine-tuned model (5,000 images, seed 0)
- `resnet18_finetune_n5000.pt`: Checkpoint 1 fine-tuned model (used in the notebook's data check and web-app test)

## Settings used for the reported results
* Data: CIFAR-10 from the public JPEG mirror https://github.com/YoongiKim/CIFAR-10-images (the official host was
  unreachable during the run). Same images and split as the official files, but JPEG-compressed.
* Balanced training subsets of 500 / 1,000 / 2,500 / 5,000 images, seeds 0 and 1; full 10,000-image test set.
* Both networks: ResNet-18, 200 SGD steps (fixed-step budget), batch 64, momentum 0.9 (Nesterov), one-cycle schedule
  (15% warm-up), weight decay 5e-4, label smoothing 0.1, inputs upsampled to 64×64, random crop + flip.
  **Learning rates: 0.05 (scratch) and 0.01 (fine-tuned), same rate for all layers.**
* Fine-tuned model starts from timm ImageNet-1k weights `resnet18_a1_0`.
* Linear probe: logistic regression (C = 0.1) on frozen ImageNet ResNet-18 features. Pixel baseline: logistic regression (C = 0.01) on raw pixels.
* Hardware: one CPU core (PyTorch 2.5.1). Each network run took about 140 s.

## How to reproduce the results
1. Download or clone this repository and `pip install -r requirements.txt` (Python 3.10–3.12; PyTorch does not yet support newer versions).
2. Create a folder named `outputs/` in the repository folder and put **both `.pt` files from the release** in it
   (the notebook loads `outputs/resnet18_finetune_n5000.pt`).
3. Open and run `Kevena_COMP365_Checkpoint2_executed.ipynb` (**Run all**). All 16 network runs are retrained from scratch
   (about 45 minutes on one CPU core, a few minutes on a GPU) and every result file is written to `outputs/`.
   Note: if the official CIFAR-10 download works on your machine, the notebook uses it instead of the JPEG mirror,
   which changes the absolute numbers; the reported results used the JPEG mirror.
4. Compare the new `outputs/cp2_*.csv` files with the ones in this repository.

## Run the web app
1. Download `resnet18_finetune_n5000_seed0_cp2.pt` from the release into the repository folder.
2. `pip install torch torchvision gradio pillow numpy`
3. `python app.py resnet18_finetune_n5000_seed0_cp2.pt` and open http://127.0.0.1:7860
   (add `--share` for a temporary public link, e.g. in Google Colab).

## Known limitations
* No separate validation split: training settings were fixed in advance, not tuned.
* Only the fixed 200-step budget was tested, so larger subsets see fewer passes over the data.
* JPEG mirror instead of the official files (the Checkpoint 1 model scored 59.7% on the official test set vs. 54.4% on the mirror).
* Only 2 seeds; at most 5,000 training images; the web-app test used the Checkpoint 1 weights; no real phone-photo test.
* The notebook's run caching is keyed only by method, size and seed. Cached runs are not included in this repository,
  so a fresh run always retrains.

## External data, code, models and AI assistance
- **Data:** CIFAR-10 (Krizhevsky, 2009), https://www.cs.toronto.edu/~kriz/cifar.html; JPEG mirror https://github.com/YoongiKim/CIFAR-10-images
- **Pretrained model:** ResNet-18 ImageNet-1k weights `resnet18_a1_0-d63eafa0.pth` from timm (Wightman, Touvron & Jégou, 2021), https://github.com/huggingface/pytorch-image-models (Apache-2.0)
- **Libraries:** PyTorch, torchvision, NumPy, pandas, Matplotlib, scikit-learn, SciPy, Pillow, Gradio, gradio_client
- **AI assistance:** Claude (Anthropic) was used to help write the notebook code, the web app, the tests and this README.
