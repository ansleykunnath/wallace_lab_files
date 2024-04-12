from psychopy import visual, core, event
from pylsl import StreamInfo, StreamOutlet
from numpy.random import random, randint, normal, shuffle, choice as randchoice
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout

# Set up LSL
info = StreamInfo(name="VEP", type="Markers", channel_count=1, nominal_srate=1, channel_format="int32", source_id="VEP")
outlet = StreamOutlet(info)

psychopyVersion = '2023.2.3'
expName = 'untitled.py'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo

# call all functions in order
expInfo = showExpInfoDlg(expInfo=expInfo)

# Set up window
win = visual.Window(size=(1536, 864), fullscr=True, monitor='display_stimuli', screen=1, units='pix')
win.color = "black"

# Parameters
check_size = 10  # Size of each square in pixels #100
check_freq = 2    # Frequency of checkerboard flash (Hz)
duration = 1     # Duration of the experiment (seconds)

# Send first event
core.wait(2)  # wait before paradigm starts
outlet.push_sample([5])
outlet.push_sample([5])
outlet.push_sample([5])


# Create checkerboard stimulus
num_squares = 160 #15
colors = [[-1, 1] if (x + y) % 2 == 0 else [1, -1]
          for x in range(num_squares) for y in range(num_squares)]
checkerboard = visual.ElementArrayStim(win, nElements=num_squares**2, sizes=(check_size, check_size),
                                       xys=[((x - num_squares // 2) * check_size, (y - num_squares // 2) * check_size)
                                            for x in range(num_squares) for y in range(num_squares)],
                                       elementTex=None, elementMask=None, colors=colors, colorSpace='rgb')

# Create fixation point stimulus
fixation_size = 15
fixation_vertical = visual.Line(win, start=(0, -fixation_size), end=(0, fixation_size), lineColor="red", lineWidth=10)
fixation_horizontal = visual.Line(win, start=(-fixation_size, 0), end=(fixation_size, 0), lineColor="red", lineWidth=10)

# Main loop
win.flip()

for n in range(5):
    for i in range(10):
        fixation_vertical.draw()
        fixation_horizontal.draw()
        win.flip()
        core.wait(1)
        outlet.push_sample([2])

    for i in range(20):
        start_time = core.getTime()
        outlet.push_sample([3])
        while core.getTime() - start_time < duration:
            t = core.getTime() - start_time
            phase = int(t * check_freq) % 2
            colors = [[-1, -1, -1] if (x + y + phase) % 2 == 0 else [1, 1, 1]
                      for x in range(num_squares) for y in range(num_squares)]
            checkerboard.colors = colors
            checkerboard.draw()
            fixation_vertical.draw()
            fixation_horizontal.draw()
            win.flip()
        outlet.push_sample([1])
        for keys in event.getKeys():
            if 'escape' in keys:
                win.close()
                core.quit()    

for i in range(5):
    fixation_vertical.draw()
    fixation_horizontal.draw()
    win.flip()
    core.wait(1)
    outlet.push_sample([2])


# Clean up
outlet.push_sample([5])
win.close()
core.quit()
