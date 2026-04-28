#### Usage:

##### Ensure SITL is placed is launched, flying at a reasonable velocity (~17mph), at altitude, and placed in GUIDED mode, before running controller.

Base: 
```python3 x8_run.py```

RC Overrride: 
```python3 x8_run.py --rc```

Test Case Selection: 
```python3 x8_run.py --test chirp```
  (see X8_Sequencer.py for options)

Particular Connection: 
```python3 x8_run.py --connect tcp:127.0.0.1:5762```

#### Prerequisites (for SITL/Gazebo/Controller):

Terminal 1 (within ~/ardupilot/Tools/autotest):  
```./sim_vehicle.py -v ArduPlane --model JSON --add-param-file=$HOME/SITL_Models/Gazebo/config/skywalker_x8.param --console --map```

Terminal 2:         
```gz sim -v4 -r skywalker_x8_runway.sdf```

Terminal 3:  
```python3 x8_run.py```

Ensure Skywalker model is installed with 

```git clone https://github.com/ArduPilot/SITL_Models.git```

#### View Results:

```python3 x8_plot.py sitl_run_1234567890.npz```

```python3 x8_plot.py sitl_run_1234567890.npz --save ```

```python3 x8_plot.py sitl_run_1234567890.npz --panel 1 ```

#### Sliding Mode Control Prototype
Run SITL and Gazebo instance as above, then set plane on takeoff to cruise altitude ('arm throttle', 'mode TAKEOFF' in MAVLink terminal)

```python3 x8_run_smc.py --alt 100 --leg 400 --radius 100 --legs 10 --no-alt```

This should attempt to the SMC controller on a lawnmower trajectory pattern at altitude 100m, straight legs 400m, turns at 100m radius, and 10 legs total. 

SW: DOES NOT WORK WELL - DON'T THINK SPEED COMMAND IS CORRECT. Also it is putting it in GUIDED MODE and think we should do FWBA.. 