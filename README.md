# Adaptive & Sliding Mode Control for Skywalker X8
##### Alyssa Hargis and Sam Wu
###### MIT 2.152 Nonlinear Control Spring 2026

Code in development for adaptive and sliding mode controller for the Skywalker X8. This work presents nonlinear control strategies for the underactuated Skywalker X8 (a fixed wing UAV with elevon control surfaces) within a ArduPlane SITL + Gazebo simulation environmnets. We focus on robust attitude and pitch control under disturbances, using sliding mode control (SMC) and adaptive control layers over the ArduPilot PID controller.

<p align="center">
  <img  width="50%" height="1512" alt="IMG_9160" src="https://github.com/user-attachments/assets/d125cce9-1475-4231-a172-0f982bfd6465" />
</p>

## System Architecture

### Level 1 - Sliding Mode Outer Loop (GUIDED):

- **Control Type** - Sliding Mode
- **States** -  Roll, Pitch
- **Inner Loop** - ArduPlane attitude controller (PID)

**Description**
The SMC presents an outer loop that computes desired attitude setpoints (pitch / roll) and are sent to the ArduPlane via Mavlink. This allows the sophisticated ArduPlane processes to handle elevon mixing, rate stabilization, and actuator dynamics. This also relies on existing Skywalker parameters from [1] and [2].

### Level 2 - Adaptive + Sliding Mode Inner and Outer Loop (MANUAL): 

- **Control Type** - Adaptive Control & Sliding Mode
- **States** -  Elevon Deflection
- **Inner Loop** - Direct Passthrough

**Description**
The controller outputs actuator commands for the elevons directly over PWM and has full authority over throttle and the control surfaces. ArduPlane only acts as a passthrough for the inputs. This problem is significantly more difficult as the controller must handle stability, elevon mixing, and model uncertainty but allows for the adaptive controller to converge on parameters.

### Disturbance Modeling

Implemented using Gazebo for controlled wind disturbances. Used to evaluate the robustness of both methods and compared with standard ArduPlane PID results. Our results are shown in the **Results** folder.

[1] K. Gryte, R. Hann, M. Alam, J. Roháč, T. A. Johansen and T. I. Fossen, "Aerodynamic modeling of the Skywalker X8 Fixed-Wing Unmanned Aerial Vehicle," 2018 International Conference on Unmanned Aircraft Systems (ICUAS), Dallas, TX, USA, 2018, pp. 826-835, doi: 10.1109/ICUAS.2018.8453370.
keywords: {Atmospheric modeling;Aerodynamics;Wind tunnels;Mathematical model;Numerical models;Aircraft},

[2] https://github.com/ArduPilot/SITL_Models/blob/master/Gazebo/docs/SkywalkerX8.md
