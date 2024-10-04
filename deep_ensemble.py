import pathlib
import statistics
import time
import tqdm
from udacity_gym import UdacitySimulator, UdacityGym, UdacityAction
from udacity_gym.agent import PIDUdacityAgent, DaveUdacityAgent
from udacity_gym.agent_callback import LogObservationCallback

if __name__ == '__main__':

    # Configuration settings
    host = "127.0.0.1"
    port = 4567
    simulator_exe_path = "/home/luigia/linux_build/Builds/udacity_linux.x86_64"

    # Track settings
    track = "lake"
    daytime = "day"
    weather = "sunny"

    # Creating the simulator wrapper
    simulator = UdacitySimulator(
        sim_exe_path=simulator_exe_path,
        host=host,
        port=port,
    )

    # Creating the gym environment
    env = UdacityGym(
        simulator=simulator,
    )
    simulator.start()
    observation, _ = env.reset(track=f"{track}", weather=f"{weather}", daytime=f"{daytime}")

    # Wait for environment to set up
    while not observation or not observation.is_ready():
        observation = env.observe()
        print("Waiting for environment to set up...")
        time.sleep(1)

    log_observation_callback = LogObservationCallback(pathlib.Path("dataset2"))
    agents = []
    check_point_paths=[]
    check_point_paths.append("/home/luigia/checkpoints_42/dave2.ckpt")
    check_point_paths.append("/home/luigia/checkpoints_43/dave2.ckpt")
    check_point_paths.append("/home/luigia/checkpoints_44/dave2.ckpt")
    check_point_paths.append("/home/luigia/checkpoints_45/dave2.ckpt")
    check_point_paths.append("/home/luigia/checkpoints_46/dave2.ckpt")
    
    for path in check_point_paths:
        agents.append(
            DaveUdacityAgent(checkpoint_path= path,
                            before_action_callbacks=[],
                            after_action_callbacks=[log_observation_callback])
        )
    '''
    agents.append(DaveUdacityAgent(checkpoint_path= "/home/luigia/checkpoints_42/dave2.ckpt",
                            before_action_callbacks=[],
                            after_action_callbacks=[log_observation_callback]))
    
    agents.append(DaveUdacityAgent(checkpoint_path= "/home/luigia/checkpoints_43/dave2.ckpt",
                            before_action_callbacks=[],
                            after_action_callbacks=[log_observation_callback]))
    
    agents.append(DaveUdacityAgent(checkpoint_path= "/home/luigia/checkpoints_44/dave2.ckpt",
                            before_action_callbacks=[],
                            after_action_callbacks=[log_observation_callback]))
    
    agents.append(DaveUdacityAgent(checkpoint_path= "/home/luigia/checkpoints_45/dave2.ckpt",
                            before_action_callbacks=[],
                            after_action_callbacks=[log_observation_callback]))
    
    agents.append(DaveUdacityAgent(checkpoint_path= "/home/luigia/checkpoints_46/dave2.ckpt",
                            before_action_callbacks=[],
                            after_action_callbacks=[log_observation_callback]))

    '''
    
    # Interacting with the gym environment
    for _ in tqdm.tqdm(range(20)):
        actions=[]

        for agent in agents:
            actions.append(agent(observation))
        
        steering_angles= []
        throttles=[]

        for action in actions:
            steering_angles.append(action.steering_angle)
            throttles.append(action.throttle)
        
        print(steering_angles)

        final_steering_angle= statistics.mean(steering_angles)
        final_throttle= statistics.mean(throttles)

        #print(final_steering_angle)
        #print(str(statistics.variance(steering_angles)))
        final_action = UdacityAction(steering_angle=final_steering_angle, throttle=final_throttle)
        last_observation = observation
        observation, reward, terminated, truncated, info = env.step(final_action)

        while observation.time == last_observation.time:
            observation = env.observe()
            time.sleep(0.005)

    log_observation_callback.save()
    simulator.close()
    env.close()
    print("Experiment concluded.")
