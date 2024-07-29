![image](https://github.com/user-attachments/assets/2eee6c44-8218-4c9a-b81a-7285948d0bd4)
# UR3 Robot Playing Chess
The results and a full explanation from this project can be seen here: 
https://www.youtube.com/watch?v=HJsMiX6PEWk&t=0s&ab_channel=EitanCoronel

For any question feel free to reach me on linkedin: https://www.linkedin.com/in/eitan-coronel/

## Overview

This project name CHESS MATE sets up an OSC server to receive coordinates and commands for controlling a UR3 robot using the RoboDK robotics toolbox. The robot can move to specified positions based on received OSC messages, making it ideal for applications like automating chess piece movements on a physical board.

### Main File Explanation

The main file, `server.py`, initializes the robot, sets up an OSC server, and defines handlers for incoming OSC messages. The server listens for coordinate data and commands, and moves the robot to the specified positions. It leverages the RoboDK API for robot control and the `python-osc` library for OSC communication.

## Features

- **OSC Server**: Listens for incoming OSC messages and processes coordinate data.
- **RoboDK Integration**: Utilizes RoboDK for precise robot control.
- **Python OSC**: Leverages `python-osc` for handling OSC messages.
- **UR3 Robot Support**: Specifically designed for the UR3 robot, but can be adapted for other robots.

## Getting Started

### Prerequisites

- [RoboDK](https://robodk.com/download) installed and configured.
- Python 3.x installed.
- Required Python packages: `robodk`, `python-osc`, `numpy`, `opencv-python`.

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/robodk-chess-robot-control.git](https://github.com/eitancoronel/Chess-Mate
   cd robodk-chess-robot-control
   ```

2. **Install the Required Packages**
   pip install robodk python-osc numpy opencv-python
   
### Usage

1. **Configure the Robot**
   - Ensure your UR3 robot is set up in RoboDK with the name "UR3".
   - Update the `robot_name` variable if your robot has a different name.

2. **Run the Server**
   python server.py (you can also make it "run on robot" via robotdk)
   This will start the OSC server, which will listen for incoming messages on the specified IP and port.
   
4. **Send OSC Messages**
   - Use an OSC client to send messages to the server. For example, you can use software like https://hexler.net/touchosc.


### Example OSC Message
/coords 10 20 30 40
This message will be processed by the "coords_handler" function, which will move the robot to the specified coordinates.

After that you configured all this and made sure everything works you can run the main function.
Enjoy !


## Author
Eitan Coronel
