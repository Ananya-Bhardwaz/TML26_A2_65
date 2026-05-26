<h1>Stolen Model Detection</h1>

<p>
This project focuses on detecting whether image classification models were stolen, copied,
fine-tuned, distilled, or otherwise derived from a given target model.
</p>

<p>
We implement a <strong>Stolen Model Detection</strong> pipeline for a <strong>CIFAR-style ResNet-18</strong>
classifier using multiple forensic similarity signals:
</p>

<ul>
  <li>Weight-based fingerprinting</li>
  <li>BatchNorm statistical fingerprinting</li>
  <li>Behavioral similarity scoring</li>
  <li>Multi-layer feature similarity</li>
  <li>Rank fusion ensemble scoring</li>
</ul>

<h2>Methods used </h2>

<ul>
  <li>
    <strong>Weight Similarity</strong> — All trainable parameters of the target and suspect models are
    flattened into one vector. Cosine similarity and L2-distance-based similarity are used to
    detect direct copies, fine-tuned checkpoints, and weight-derived models.
  </li>

  <li>
    <strong>BatchNorm Similarity</strong> — Running mean and running variance values from BatchNorm
    layers are extracted and compared. These statistics can preserve useful fingerprints even
    after moderate fine-tuning.
  </li>

  <li>
    <strong>Prediction Agreement</strong> — The target and suspect models are tested on the same
    CIFAR-100 probe images. We measure how often both models predict the same class.
  </li>

  <li>
    <strong>Logit Similarity</strong> — Raw output logits from both models are compared using cosine
    similarity. This helps detect models with similar decision behaviour.
  </li>

  <li>
    <strong>Probability Similarity</strong> — Softmax probability distributions are compared using
    cosine similarity. This helps detect extracted or distilled models that imitate output
    probabilities.
  </li>

  <li>
    <strong>KL-Divergence Similarity</strong> — KL-divergence is used to compare the probability
    distributions of the target and suspect models. This captures distribution-level similarity.
  </li>

 <li>
  <strong>Feature Similarity</strong> — A forward hook is used to extract intermediate activations
  from layer4 of ResNet-18. These feature representations are flattened and compared using
  cosine similarity.
</li>

  <li>
    <strong>Rank Fusion</strong> — Instead of relying on raw weighted scores, each metric produces its
    own suspiciousness ranking. These rankings are then fused into a final stealing confidence
    score. Rank fusion reduces score saturation and improves calibration.
  </li>
</ul>

<h2>Final Feature Score</h2>

<p>
The feature similarity score is computed using layer4 representations.
</p>
<pre><code>
feature_score = layer4_feature_score
</code></pre>

<p>
Layer4 was used for feature similarity because it captures high-level task-specific
representations from the ResNet-18 model. These layer4 activations were extracted using
a forward hook, flattened, and compared between the target model and each suspect model
using cosine similarity.
</p>

<h2>Final Rank Fusion Weights</h2>

<p>
The final detector uses rank fusion over the following signals:
</p>

<pre><code>
rank_weights = {
    "weight_score_rank": 0.10,
    "bn_score_rank": 0.15,
    "logit_cosine_rank": 0.15,
    "prob_cosine_rank": 0.10,
    "kl_score_rank": 0.10,
    "agreement_rank": 0.10,
    "feature_score_rank": 0.30,
}
</code></pre>

<p>
Feature similarity gets the highest weight because it gives the strongest separation between
suspicious models and the genuine models.
</p>

<h2>Setup</h2>

<ul>
  <li>Target model is a CIFAR-style ResNet-18 classifier for 100 classes.</li>
  <li>The suspect set contains 360 ResNet-18 models.</li>
  <li>CIFAR-100 test data is used as the probe dataset.</li>
  <li>The full probe set contains 10,000 images.</li>
  <li>All models are evaluated in inference mode.</li>
</ul>

<h2>Testing</h2>

<p>
The evaluation metric is <strong>TPR @ 5% FPR</strong>. It measures how many stolen models are
detected correctly when we are keeping the false positives very low.
</p>

<p>
Our final public leaderboard result:
</p>

<ul>
  <li><strong>Initial weight-only score:</strong> 0.166667</li>
  <li><strong>Behavior + BatchNorm score:</strong> 0.611111</li>
  <li><strong>Final feature + rank fusion score:</strong> 0.629630</li>
</ul>

<h2>Final Observations</h2>

<ul>
  <li>Weight similarity alone was not enough for strong stolen model detection.</li>
  <li>Behavioral signals improved detection of distilled and extracted models.</li>
  <li>BatchNorm statistics helped capture hidden training fingerprints.</li>
  <li>Multi-layer feature similarity improved representation-level detection.</li>
  <li>Rank fusion reduced score saturation and improved leaderboard performance.</li>
</ul>

<h2>Structure</h2>

<pre><code>
Stolen Model Detection.ipynb
report.pdf
README.md
</code></pre>

<h2>How to Reproduce</h2>

<ol>
  <li>Open <code>Stolen Model Detection.ipynb</code> in Google Colab.</li>
  <li>Enable GPU from Runtime settings.</li>
  <li>Run the dependency installation cell.</li>
  <li>Download/load the target model and suspect models.</li>
  <li>Run the cells for weight similarity, BatchNorm similarity, behavioral analysis, Multi layer feature similarity and rank fusion.</li>
  <li>Generate <code>submission.csv</code>.</li>
  
</ol>

<h2>Takeaway</h2>

<p>
Model stealing detection is not about comparing weights. A stolen model will conserve
behaviour, output distributions, and internal representations even when the
parameters are all changed. Combining all these different signals gives a strong detector.
</p>

<h2>Authors</h2>

<p>
Aryan Aryan — arar00002@stud.uni-saarland.de<br>
Ananya Bhardwaz — anbh00002@stud.uni-saarland.de
</p>
