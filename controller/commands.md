#### Usage:
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
```gz sim -r skywalker_x8.sdf ```         

Terminal 3:  
```python3 x8_run.py```

Ensure Skywalker model is installed with 

```git clone https://github.com/ArduPilot/SITL_Models.git```

#### View Results:

```python3 x8_plot.py sitl_run_1234567890.npz```

```python3 x8_plot.py sitl_run_1234567890.npz --save ```

```python3 x8_plot.py sitl_run_1234567890.npz --panel 1 ```
