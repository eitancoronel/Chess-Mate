"""
OSC Server for RoboDK Chess Robot Control

Author: Eitan Coronel
Date: [Insert Date]
Description: This script sets up an OSC server to receive coordinates and commands for controlling a UR3 robot using
the RoboDK robotics toolbox. The robot moves to specified positions based on received OSC messages.
"""

from robodk.robolink import Robolink
from robodk import *  # RoboDK robotics toolbox
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server


class RoboDKChessRobot:
    def __init__(self, robot_name, ip="127.0.0.1", port=5005):
        """
        Initialize the RoboDK Chess Robot control server.
        Args:
            robot_name (str): Name of the robot to control.
            ip (str): IP address for the OSC server.
            port (int): Port for the OSC server.
        """
        self.square_size = 41
        self.RDK = Robolink()
        self.robot = self.RDK.Item(robot_name, robolink.ITEM_TYPE_ROBOT)
        if not self.robot.Valid():
            raise Exception(f'Robot "{robot_name}" not found or available')
        self.ip = ip
        self.port = port
        self.dispatcher = Dispatcher()
        self.server = None

    def coords_handler(self, unused_addr, val1, val2, val3, val4):
        """
        Handle incoming OSC messages with coordinates.
        Args:
            unused_addr: Address pattern of the OSC message.
            val1: First coordinate value.
            val2: Second coordinate value.
            val3: Third coordinate value.
            val4: Fourth coordinate value.
        """
        try:
            print(f'val1: {val1} , val2: {val2}, val3: {val3}, val4: {val4}')
            p0 = TxyzRxyz_2_Pose([363, 24, 150, pi, 0, -pi])
            self.robot.MoveL(p0)
        except ValueError:
            pass

    def start_server(self):
        """
        Start the OSC server to listen for incoming messages.
        """
        self.dispatcher.map("/coords", self.coords_handler)
        self.server = osc_server.ThreadingOSCUDPServer(
            (self.ip, self.port), self.dispatcher)
        print("Serving on {}".format(self.server.server_address))
        self.server.serve_forever()


if __name__ == "__main__":
    robot_name = "UR3"
    ip = "127.0.0.1"
    port = 5005

    robot_control_server = RoboDKChessRobot(robot_name, ip, port)
    robot_control_server.start_server()
