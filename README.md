# Adaptive Control for Skywalker X8
###### Alyssa Hargis and Sam Wu // MIT 2.152 Nonlinear Control Spring 2026

Code in development for adaptive and sliding mode controller for the Skywalker X8.

Presents two levels using Mavlink through ArduPlane SITL with Gazebo integration for distubances.

### Level 1:

- Presents a sliding mode controller on pitch and roll control while using the ArduPlane inner control loop as the stabilizer within GUIDED mode (ie. ArduPlane software) manages the low-level control. 

- Implemented using parameters from [1] and [2].

### Level 2: 

- Presents an adaptive controller that provides direct PWM control over MANUAL mode passthrough.

- Implemented with --rc flag on run command

- Still in development.


[1] K. Gryte, R. Hann, M. Alam, J. Roháč, T. A. Johansen and T. I. Fossen, "Aerodynamic modeling of the Skywalker X8 Fixed-Wing Unmanned Aerial Vehicle," 2018 International Conference on Unmanned Aircraft Systems (ICUAS), Dallas, TX, USA, 2018, pp. 826-835, doi: 10.1109/ICUAS.2018.8453370.
keywords: {Atmospheric modeling;Aerodynamics;Wind tunnels;Mathematical model;Numerical models;Aircraft},

[2] https://github.com/ArduPilot/SITL_Models/blob/master/Gazebo/docs/SkywalkerX8.md
