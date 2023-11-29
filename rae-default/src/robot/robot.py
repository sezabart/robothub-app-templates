from typing import Optional, List, Tuple

from src.api.ros.ros2_manager import ROS2Manager
from src.robot.movement import MovementController
from sensor_msgs.msg import BatteryState


class Robot:
    """
    A class representing a robot, integrating various controllers for movement, display, and LED management,
    and interfacing with ROS2 for communication and control.

    Attributes:
        logger: An instance used for logging messages and errors.
        ros_manager (ROS2Manager): An object for managing ROS2 communications and functionalities.
        battery_state (BatteryState): Stores the current state of the robot's battery.
        movement_controller (MovementController): Handles the robot's movement.

    Methods:
        battery_state_cb(data): Callback method for updating battery state.
        start(): Initializes the robot's components and starts ROS2 communications.
        stop(): Stops the ROS2 communications and shuts down the robot's components.
        move(velocity): Commands the robot to move at a specified velocity.
        get_battery(): Retrieves the current battery state.
    """

    def __init__(self, logger):
        """
        Initializes the Robot instance.

        Args:
            logger: An instance used for logging messages and errors.
        """

        self.logger = logger
        self.ros_manager = None
        self.battery_state = None
        self.movement_controller = None

    def battery_state_cb(self, data):
        self.battery_state = data

    def start(self):
        """
        Initializes and starts the robot's components and ROS2 communications.
        Sets up necessary controllers and subscribers for the robot's functionalities.
        """
        self.ros_manager = ROS2Manager("base_container")
        self.ros_manager.start()
        self.movement_controller = MovementController(self.ros_manager)
        self.ros_manager.create_subscriber(
            "/battery_status", BatteryState, self.battery_state_cb)

    def stop(self):
        """
        Stops the ROS2 communications and deactivates the robot's controllers.
        Ensures a clean shutdown of all components.
        """
        self.display_controller.stop()
        self.ros_manager.stop()

    def move(self, linear, angular):
        """
        Commands the robot to move at the specified velocity.

        Args:
            linear: Linear velocity.
            angular: Angular velocity.
        """
        self.movement_controller.move(linear, angular)



    def get_battery(self):
        """
        Retrieves the current state of the robot's battery.

        Returns:
            BatteryState: The current state of the battery.
        """
        return self.battery_state
