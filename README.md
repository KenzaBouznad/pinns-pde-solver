<div align="center">
  <h1>🚀 Generalized PDE Solver Using PINNs</h1>
  <p>
    <strong>Physics-Informed Neural Networks vs. Finite Difference Methods</strong>
  </p>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white" />
    <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  </p>
</div>

<hr />

<h2>📖 Overview</h2>
<p>
  This study implements a <strong>Physics-Informed Neural Network (PINN)</strong> to solve Partial Differential Equations (PDEs). By embedding physical laws into the loss function of the neural network, we evaluate its performance against the traditional <strong>Finite Difference Method (FDM)</strong>.
</p>



<h3>Key Objectives</h3>
<ul>
  <li><strong>Generalization:</strong> Development of a PINN framework capable of solving PDEs of varying orders.</li>
  <li><strong>Comparative Analysis:</strong> Benchmarking accuracy, computational complexity, and efficiency against numerical FDM solvers.</li>
</ul>

<h2>🛠️ Core Features</h2>
<ul>
  <li><strong>Automatic Differentiation:</strong> High-order gradient, Jacobian, and Hessian computations using specialized classes.</li>
  <li><strong>Multi-Language Benchmarking:</strong> High-performance FDM implementation in <strong>Rust</strong> for baseline comparisons.</li>
  <li><strong>Hybrid Training:</strong> Dedicated data classes for training and validation dataset synthesis.</li>
</ul>



<h2>📂 Repository Structure</h2>

<table width="100%">
  <thead>
    <tr>
      <th>Module / File</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>fdm_burgers_1d/</code></td>
      <td>High-performance <strong>Rust</strong> implementation of FDM for 1D Burgers.</td>
    </tr>
    <tr>
      <td><code>model.py</code></td>
      <td>Core class to construct and train the PINN architecture.</td>
    </tr>
    <tr>
      <td><code>gradient_function.py</code></td>
      <td>Custom logic for computing physical gradients and Hessians.</td>
    </tr>
    <tr>
      <td><code>burgers_eq_1_degree.py</code></td>
      <td>Main solver script for the 1D Burgers equation.</td>
    </tr>
    <tr>
      <td><code>data.py</code></td>
      <td>Dataset generator for training and validation points.</td>
    </tr>
    <tr>
      <td><code>metrics.py</code></td>
      <td>Visualization suite for plots and error metrics analysis.</td>
    </tr>
    <tr>
      <td><code>results/</code></td>
      <td>Storage for model outputs and benchmark <code>.csv</code> data.</td>
    </tr>
  </tbody>
</table>

<h2>📊 Results & Evaluation</h2>
<p>
  The study systematically compares the disparities between traditional numerical methods and machine learning-based solvers. Results are exported to the <code>/results</code> directory, where <code>results_1d_burgers.csv</code> and <code>results_fdm.csv</code> provide the raw data for accuracy evaluation.
</p>

<hr />

<div align="center">
  <p><i>Developed as a comparative study for the Scientific Computing Module at TU Berlin.</i></p>
</div>

