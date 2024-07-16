from psychopy import visual, core, event
from pylsl import StreamInfo, StreamOutlet
import time
from datetime import datetime

import json
import socket
import struct
from time import time

# Set up window
win = visual.Window(size=(2560, 1440), fullscr=True, monitor='display_stimuli', screen=1, units='pix')
win.color = "black"

# one time connection
host = 'magical-mcclintock'
port = 6767
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))

# then send events
def send_event(sock, id, event_name, value):
    timestamp = time()
    data_to_send = {
        "id": id,
        "timestamp": int(timestamp * 1e6),
        "event": "start_trial",
        "value": "5",
    }
    event = json.dumps(data_to_send).encode("utf-8")
    msg = struct.pack("!I", len(event)) + event
    sock.sendall(msg)
    
event_id = 1
send_event(sock, event_id, "start experiment", "0")
event_id += 1
send_event(sock, event_id, "start_trial", "5")

# Set up LSL
info = StreamInfo(name="VEP", type="Markers", channel_count=1, nominal_srate=1, channel_format="int32", source_id="VEP")
outlet = StreamOutlet(info)

#change parameters to 0.5 hz, 1 hz, and bigger square size

# Parameters
check_size = 20  # Size of each square in pixels #100 #changed size of check from 10 to 20
check_freq = 1    # Frequency of checkerboard flash (Hz)  # changed from 0.5 to 1
duration = 2     # Duration of the experiment (seconds) #increased from 1 to 2 for 1 hz; 
fixation_size = 30 #changed size of fixation from 15 to 30
lineWidth = 15

# Send first event
core.wait(5)  # wait before paradigm starts
outlet.push_sample([5])
outlet.push_sample([5])
outlet.push_sample([5])

# Create checkerboard stimulus
num_squares = 260 #15
colors = [[-1, 1] if (x + y) % 2 == 0 else [1, -1]
          for x in range(num_squares) for y in range(num_squares)]
checkerboard = visual.ElementArrayStim(win, nElements=num_squares**2, sizes=(check_size, check_size),
                                       xys=[((x - num_squares // 2) * check_size, (y - num_squares // 2) * check_size)
                                            for x in range(num_squares) for y in range(num_squares)],
                                       elementTex=None, elementMask=None, colors=colors, colorSpace='rgb')

# Create fixation point stimulus
fixation_vertical = visual.Line(win, start=(0, -fixation_size), end=(0, fixation_size), lineColor="red", lineWidth=lineWidth)
fixation_horizontal = visual.Line(win, start=(-fixation_size, 0), end=(fixation_size, 0), lineColor="red", lineWidth=lineWidth)

# Main loop
win.flip()
print("start")
print(core.getTime()) 

for n in range(10):
    for i in range(10):
        fixation_vertical.draw()
        fixation_horizontal.draw()
        win.flip()
        outlet.push_sample([1])
        core.wait(1)
    for i in range(10):
        start_time = core.getTime()
        while core.getTime() - start_time < duration:
            t = core.getTime() - start_time
            phase = int(t * check_freq) % 4 #changed from 1 to 2 for frequency 1; changed from 2 to 4 for frequency 0.5
            colors = [[-1, -1, -1] if (x + y + phase) % 2 == 0 else [1, 1, 1]
                      for x in range(num_squares) for y in range(num_squares)]
            checkerboard.colors = colors
            checkerboard.draw()
            fixation_vertical.draw()
            fixation_horizontal.draw()
            win.flip()
        outlet.push_sample([2])
        for keys in event.getKeys():
            if 'escape' in keys:
                win.close()
                core.quit()    

for i in range(5):
    fixation_vertical.draw()
    fixation_horizontal.draw()
    win.flip()
    outlet.push_sample([1])
    core.wait(1)

# Clean up
outlet.push_sample([5])
print("end")
print(core.getTime())
win.close()
core.quit()

sock.close()




