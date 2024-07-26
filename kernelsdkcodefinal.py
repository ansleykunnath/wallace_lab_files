from psychopy import visual, core, event
import json
import socket
import struct
from time import time

# Set up window
win = visual.Window(size=(2560, 1440), fullscr=True, monitor='display_stimuli', screen=1, units='pix')
win.color = "black"

# One-time connection
host = 'magical-mcclintock'
port = 6767
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))

# Function to send events
def send_event(sock, id, event_name, value):
    timestamp = time()
    data_to_send = {
        "id": id,
        "timestamp": int(timestamp * 1e6),
        "event": event_name,
        "value": value,
    }
    event = json.dumps(data_to_send).encode("utf-8")
    msg = struct.pack("!I", len(event)) + event
    sock.sendall(msg)

# Initialize event ID
event_id = 1

# Send start_experiment event
send_event(sock, event_id, "start_experiment", "0")
event_id += 1

# Parameters
check_size = 20  # Size of each square in pixels
check_freq = 1   # Frequency of checkerboard flash (Hz)
duration = 2     # Duration of the experiment (seconds)
fixation_size = 30
lineWidth = 15

# Send initial events and wait before paradigm starts
core.wait(5)
send_event(sock, event_id, "initial_event", "5")
event_id += 1

# Create checkerboard stimulus
num_squares = 260
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
        send_event(sock, event_id, "fixation_shown", "1")
        event_id += 1
        core.wait(1)
    for i in range(10):
        start_time = core.getTime()
        send_event(sock, event_id, "checkerboard_start", str(i))
        event_id += 1
        while core.getTime() - start_time < duration:
            t = core.getTime() - start_time
            phase = int(t * check_freq) % 4
            colors = [[-1, -1, -1] if (x + y + phase) % 2 == 0 else [1, 1, 1]
                      for x in range(num_squares) for y in range(num_squares)]
            checkerboard.colors = colors
            checkerboard.draw()
            fixation_vertical.draw()
            fixation_horizontal.draw()
            win.flip()
        send_event(sock, event_id, "checkerboard_end", str(i))
        event_id += 1
        for keys in event.getKeys():
            if 'escape' in keys:
                send_event(sock, event_id, "experiment_abort", "0")
                win.close()
                core.quit()

for i in range(5):
    fixation_vertical.draw()
    fixation_horizontal.draw()
    win.flip()
    send_event(sock, event_id, "final_fixation", "1")
    event_id += 1
    core.wait(1)

# Send end_experiment event
send_event(sock, event_id, "end_experiment", "0")

# Clean up
print("end")
print(core.getTime())
win.close()
core.quit()
sock.close()