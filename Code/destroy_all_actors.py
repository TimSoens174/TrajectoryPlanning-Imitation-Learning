import carla
import os
import sys
import glob


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

client = carla.Client("localhost", 2000)
client.set_timeout(5.0)
world = client.get_world()
vehicles = world.get_actors().filter('vehicle.*')

for actor in vehicles:
            if actor.is_alive:  # Check if the actor still exists
                try:
                    actor.destroy()
                except Exception as e:
                    print(f"Error destroying actor {actor.id}: {e}")
            else:
                print(f"Actor {actor.id} was already destroyed.")