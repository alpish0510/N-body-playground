Below is a clean, well-structured **`README.md`** you can drop directly into your repository. It explains what the script does, how it works, dependencies, usage, and examples, without going too deep into implementation details.

You can copy-paste this as-is.

---

# N-Body Simulation in Python

This project implements a **2D gravitational N-body simulation** using **symbolic mechanics (SymPy)** to derive the equations of motion and **numerical integration (SciPy)** to evolve the system in time. The result is an animated visualization of interacting bodies under Newtonian gravity.

The simulation supports configurable numbers of bodies, masses, initial conditions, and gravitational constant via command-line arguments.

---

## Features

* Symbolic derivation of equations of motion using **Lagrangian mechanics**
* Automatic generation of acceleration functions via **SymPy → NumPy lambdification**
* High-precision numerical integration using `scipy.integrate.solve_ivp`
* Animated trajectories using **Matplotlib**
* Optional export of the animation as a GIF
* Parallelized symbolic simplification using **multiprocessing**

---

## Requirements

Python **3.8+** is recommended.

Required packages:

```bash
numpy
scipy
sympy
matplotlib
```

Optional but recommended:

```bash
tqdm        # progress bars
ipython     # animation display in notebooks
pillow      # saving GIF animations
```

Install everything with:

```bash
pip install numpy scipy sympy matplotlib tqdm pillow ipython
```

---

## How It Works (Overview)

1. **Symbolic Setup**

   * Defines symbolic positions (x_i(t), y_i(t)) for each body
   * Constructs kinetic and potential energy
   * Forms the Lagrangian (L = T - V)

2. **Equations of Motion**

   * Applies Euler–Lagrange equations
   * Solves symbolically for accelerations
   * Simplifies expressions in parallel

3. **Numerical Integration**

   * Converts symbolic accelerations to fast NumPy functions
   * Integrates using `solve_ivp`

4. **Visualization**

   * Plots trajectories and current positions
   * Animates motion over time
   * Saves animation as a GIF (optional)

---

## Usage

Run the script from the command line:

```bash
python n_body.py [options]
```

### Command-Line Arguments

| Argument            | Description                                     | Default               |
| ------------------- | ----------------------------------------------- | --------------------- |
| `--num_bodies`      | Number of bodies                                | `3`                   |
| `--G`               | Gravitational constant                          | `1.0`                 |
| `--masses`          | Masses of bodies                                | `5.0 3.0 4.0`         |
| `--init_positions`  | Initial positions (x₁…xₙ y₁…yₙ)                 | `-10 1 0 5 2 12.5`    |
| `--init_velocities` | Initial velocities (vx₁…vxₙ vy₁…vyₙ)            | `0.2 -0.2 0 0 0 -0.1` |
| `--orbit`           | Initialize an orbital system (future extension) | `False`               |
| `save_fig`          | Save animation as GIF                           | `True`                |

---

## Example

Run a basic 3-body simulation and save the animation:

```bash
python n_body.py --num_bodies 3 --G 1.0 \
--masses 5 3 4 \
--init_positions -10 1 0 5 2 12.5 \
--init_velocities 0.2 -0.2 0 0 0 -0.1
```

This will generate:

```text
n_body_simulation.gif
```

in the current directory.

---

## Output

* **Animated trajectories** of each body
* **Scatter points** showing current positions
* **Equal-aspect ratio** for physically accurate motion


