#### Usage:
Base: 
python3 x8_run.py
RC Overrride: 
python3 x8_run.py --rc
Test Case Selection: 
python3 x8_run.py --test chirp
  (see X8_Sequencer.py for options)
Particular Connection: 
python3 x8_run.py --connect tcp:127.0.0.1:5762

### Prerequisites (for SITL/Gazebo/Controller):
    Terminal 1:  sim_vehicle.py -v ArduPlane -f gazebo-zephyr --map --console
    Terminal 2:  gz sim -r skywalker_x8.sdf          (if not launched by sim_vehicle)
    Terminal 3:  python3 x8_run.py
