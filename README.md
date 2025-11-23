# Evolving a Neural-Network Agent for Planet Wars
`John Asaro - Claire Carroll - Derin Gezgin`

This repository includes our code for the final project of COM407: Computational Intelligence course. In this final project, we evolved a simple neural network agent via CMA-ES for the Planet Wars game. 

## Setup

### Without Docker

Requirements: Python 3.10+, Java JDK 21, Git, Bash
```bash
git clone https://github.com/deringezgin/COM407-FinalProject
cd ci_final
./setup.sh
```

The setup script will create a Python virtual environment, clone the [Planet Wars source code](https://github.com/SimonLucas/planet-wars-rts), apply our patch for the GUI support, install the Python dependencies and build the Planet Wars app. 

If you already have your own virtual environment or wish not to have one, you can run the setup script with the `noenv` flag. 
```bash
./setup.sh noenv
```

### With Docker

You can also run our project in a Docker container.
```bash
git clone https://github.com/deringezgin/COM407-FinalProject
cd ci_final
docker build -t planet-wars-ci .
```

To run it with a GUI, start the container with display forwarding.
```bash
docker run -it -e DISPLAY=host.docker.internal:0 planet-wars-ci /bin/bash
```

If the display forwarding is not supported in your device, you can run our agent in headless mode described in the `Running the Trained Agent` section. 

## Training an Agent

To train our neural network agent, 

## Running the Trained Agent

### Running with GUI

### Running without a GUI

## Running Baseline Benchmarks

## Visualizing Results
