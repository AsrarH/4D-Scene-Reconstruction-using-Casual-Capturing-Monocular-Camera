# 4D Scene Reconstruction using Casual Capturing Monocular Camera

## Overview
This Capstone Project, led by Asrar Alruwayqi under the guidance of Prof. Shubham Tulsiani, explores innovative methods for 4D scene reconstruction using a casual capturing monocular camera. The goal is to develop a system capable of reconstructing dynamic 3D scenes over time (4D) using simple, everyday equipment.

## Background
Traditional methods for 4D scene reconstruction often require specialized equipment, controlled environments, or multiple cameras, which can be costly and complex. This project seeks to simplify the process, enabling dynamic scene capture in natural, uncontrolled settings with a single camera.

## Project Objectives
- **Hybrid Scene Representation:** Utilize a tri-plane and deformation field hybrid approach to represent dynamic scenes.
- **Coarse-to-Fine Reconstruction:** Implement a coarse-to-fine strategy to ensure high-quality reconstruction.
- **Novel Dynamic View Prediction:** Employ video diffusion models to predict novel dynamic views from captured data.

## Methodology


## Installation

## References and Acknowledgements
This project builds upon the D-NeRF

      @article{pumarola2020d,
        title={D-NeRF: Neural Radiance Fields for Dynamic Scenes},
        author={Pumarola, Albert and Corona, Enric and Pons-Moll, Gerard and Moreno-Noguer, Francesc},
        journal={arXiv preprint arXiv:2011.13961},
        year={2020}
      }

    @article{Cao2023HexPlane,
    author    = {Cao, Ang and Johnson, Justin},
    title     = {HexPlane: A Fast Representation for Dynamic Scenes},
    journal   = {CVPR},
    year      = {2023},
    }
