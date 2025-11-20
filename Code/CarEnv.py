# CarEnv.py
import glob
import os
import sys
import carla
import random
import numpy as np
import time
import math
import cv2

SECONDS_PER_EPISODE = 10
IM_WIDTH = 230
IM_HEIGHT = 240

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

class CarEnv:
    STEER_AMT = 1.0
    spawn_fail_count = 0  # To track the number of spawn failures

    def __init__(self, mount_camera):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.mount_camera = mount_camera
        self.state_dimension = [
            "pos_x", "pos_y", "v_x", "v_y",  # Position and velocity
            "goal_x", "goal_y",  # Goal
            "obstacle_distance",  # Obstacle distance
            "nearest_distance1", "nearest_distances2", "nearest_distances3", "nearest_distances4", "nearest_distances5",  # Nearest vehicle distances
            "dist_to_center",  # Lane info
            "left_lane_exists", "right_lane_exists",  # Adjacent lanes
            "is_junction"  # Junction detection
        ]
        self.tracking_data = np.zeros(len(self.state_dimension), dtype=np.float32)  # Placeholder for state vector
        self.obstacle_distance = float(999.0)  # Initialize obstacle distance

        # Define reward range
        self.reward_range = (-float('inf'), float('inf'))
        # Goal coordinates for the vehicle
        self.goal_x = None
        self.goal_y = None

        self.frame_skip = 2  # Number of frames to skip

    def reset(self, mount_camera = True):
        """Reset the environment for a new episode."""
        self.actor_list = []
        self.collision_hist = []
        self.lane_hist = []

        # Get available spawn points
        self.spawn_points = self.world.get_map().get_spawn_points()

        # Attempt to spawn the vehicle, handle failures
        vehicle_spawned = False
        max_attempts = 10
        attempts = 0

        while not vehicle_spawned and attempts < max_attempts:
            try:
                self.transform = random.choice(self.spawn_points)  # Random spawn point
                self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
                self.actor_list.append(self.vehicle)
                vehicle_spawned = True
            except Exception as e:
                print(f"Error spawning vehicle: {e}. Retrying with a new spawn point...")
                attempts += 1
                time.sleep(0.1)  # Pause briefly before retrying

        if not vehicle_spawned:
            # If vehicle could not be spawned after multiple attempts, skip the episode
            self.spawn_fail_count += 1  # Increment the spawn failure counter
            print(f"Failed to spawn vehicle. Skipping episode. Total spawn failures: {self.spawn_fail_count}")
            return None  # Return None to signal skipping the episode

        time.sleep(1.5)  # Give UE time to spawn the vehicle

        # Set the goal coordinates to be different from the spawn point
        while True:
            goal_point = random.choice(self.spawn_points)
            if goal_point != self.transform:
                break

        # Set goal coordinates (goal_x, goal_y)
        self.goal_x = goal_point.location.x
        self.goal_y = goal_point.location.y
        print("Goal: ", self.goal_x, self.goal_y)
        print("Start Location: ", self.vehicle.get_location())
        self.initial_distance_to_goal = self.get_distance_to_goal(self.vehicle.get_location().x, self.vehicle.get_location().y)

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # Set up sensors (collision, lane invasion, obstacle)
        self.setup_sensors()

        # Placeholder: Initialize tracking data with [pos_x, pos_y, v_x, v_y, goal_x, goal_y, obj_dist, nearest_vehicle_dist]
        self.tracking_data = self.get_tracking_data()

        return self.tracking_data  # Return the state vector

    def step(self, action):
        """Apply action and step the simulation."""
        steering, throttle = action
        if throttle >= 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), brake=0.0, steer=float(steering)))
        else:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=float(abs(throttle)), steer=float(steering)))
        time.sleep(0.1)

        # Get velocity and position
        v = self.vehicle.get_velocity()
        pos = self.vehicle.get_location()

        # Get distance to goal
        prev_dist_to_goal = self.dist_to_goal if hasattr(self, 'dist_to_goal') else self.initial_distance_to_goal
        self.dist_to_goal = self.get_distance_to_goal(pos.x, pos.y)

        # Reward function
        reward = 0
        done = False

        # Penalize collisions
        if len(self.collision_hist) != 0:
            done = True
            reward -= 2000  # Collision penalty

        # Penalize lane invasion (specifically solid lines)
        for invasion in self.lane_hist:
            if 'Solid' in invasion['lane_types']:
                reward -= 1000  # Solid line penalty

        # Reward for moving towards the goal
        if self.dist_to_goal < prev_dist_to_goal:
            reward += 0.95  # Encourage movement toward the goal


        # Episode termination based on time or goal achievement
        if self.dist_to_goal < 5.0:
            done = True

        # Update tracking data (state vector)
        self.tracking_data = self.get_tracking_data()

        return self.tracking_data, reward, done, {}

    def setup_sensors(self):
        """Set up sensors for collision, lane invasion, and obstacle detection."""
        # Set up collision sensor
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, self.transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        # Set up lane invasion sensor
        lane_sensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(lane_sensor, self.transform, attach_to=self.vehicle)
        self.actor_list.append(self.lane_sensor)
        self.lane_sensor.listen(lambda event: self.lane_data(event))

        # Set up obstacle sensor
        obstacle_sensor_bp = self.blueprint_library.find("sensor.other.obstacle")
        self.obstacle_sensor = self.world.spawn_actor(obstacle_sensor_bp, self.transform, attach_to=self.vehicle)
        self.actor_list.append(self.obstacle_sensor)
        self.obstacle_sensor.listen(lambda event: self.obstacle_data(event))
        
        # Set up Camera sensor
        if self.mount_camera:
            cam_bp = self.blueprint_library.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
            cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
            cam_bp.set_attribute("fov", "110")

            spawn_point = carla.Transform(carla.Location(x = 2.5, z = 0.7))
            self.sensor = self.world.spawn_actor(cam_bp, spawn_point, attach_to = self.vehicle)
            self.actor_list.append(self.sensor)
            self.sensor.listen(lambda data: self.process_img(data))



    def get_distance_to_goal(self, pos_x, pos_y):
        """Calculate the distance to the goal."""
        return math.sqrt((self.goal_x - pos_x) ** 2 + (self.goal_y - pos_y) ** 2)

    def collision_data(self, event):
        """Handle collision events."""
        self.collision_hist.append(event)

    def lane_data(self, event):
        """Handle lane invasion events."""
        lane_types = [str(x.type) for x in event.crossed_lane_markings]
        invasion_info = {'frame': event.frame, 'lane_types': lane_types}
        self.lane_hist.append(invasion_info)

    def obstacle_data(self, event):
        """Handle obstacle sensor data."""
        self.obstacle_distance = event.distance
    
    def process_img(self, data):
        if hasattr(self, 'frame_count'):
            self.frame_count += 1
        else:
            self.frame_count = 0

        if self.frame_count % self.frame_skip != 0:
            return  # Skip this frame
        
        i = np.array(data.raw_data)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        cv2.imshow("", i3)
        cv2.waitKey(1)

    def get_lane_information(self):
        """Get information about the current lane and adjacent lanes."""
        vehicle_location = self.vehicle.get_location()
        map = self.world.get_map()

        # Get the nearest waypoint
        waypoint = map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        
        # Detect if we are in a junction
        is_junction = waypoint.is_junction

        # Get lane information
        lane_width = waypoint.lane_width
        lane_center = waypoint.transform.location
        lane_id = waypoint.lane_id
        dist_to_center = vehicle_location.distance(lane_center)

        # Check for adjacent lanes
        left_lane = waypoint.get_left_lane() if waypoint.lane_change & carla.LaneChange.Left else None
        right_lane = waypoint.get_right_lane() if waypoint.lane_change & carla.LaneChange.Right else None

        left_lane_exists = left_lane is not None and left_lane.lane_type == carla.LaneType.Driving
        right_lane_exists = right_lane is not None and right_lane.lane_type == carla.LaneType.Driving

        return dist_to_center, lane_width, lane_id, left_lane_exists, right_lane_exists, is_junction

    def get_tracking_data(self):
        """Get tracking sensor data, including position, velocity, goal location, obstacle distance, nearby vehicles, and lane information."""
        v = self.vehicle.get_velocity()
        pos = self.vehicle.get_location()

        # Get lane information
        dist_to_center, lane_width, lane_id, left_lane_exists, right_lane_exists, is_junction = self.get_lane_information()

        # Find nearest vehicles
        vehicles = self.world.get_actors().filter('vehicle.*')
        vehicle_distances = [vehicle.get_location().distance(pos) for vehicle in vehicles if vehicle.id != self.vehicle.id]
        nearest_distances = sorted(vehicle_distances)[:5]
        while len(nearest_distances) < 5:
            nearest_distances.append(999.0)

        state_vector = np.array([
            pos.x, pos.y, v.x, v.y,  # Position and velocity
            self.goal_x, self.goal_y,  # Goal
            self.obstacle_distance,  # Obstacle distance
            nearest_distances[0], nearest_distances[1], nearest_distances[2], nearest_distances[3], nearest_distances[4],  # Nearest vehicle distances
            dist_to_center,  # Lane info
            left_lane_exists, right_lane_exists,  # Adjacent lanes
            is_junction  # Junction detection
        ], dtype=np.float32)

        return state_vector


    def destroy(self):
        """Destroy all actors."""
        for actor in self.actor_list:
            if actor.is_alive:  # Check if the actor still exists
                try:
                    actor.destroy()
                except Exception as e:
                    print(f"Error destroying actor {actor.id}: {e}")
            else:
                print(f"Actor {actor.id} was already destroyed.")
