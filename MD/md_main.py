import math
from datetime import datetime
import os
import time
import gym
import pygame
import metadrive
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.obs.top_down_renderer import draw_top_down_map
from metadrive.policy.idm_policy import IDMPolicy, WaymoIDMPolicy
from panda3d.core import PNMImage, Texture
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import sys
from albumentations import RandomBrightnessContrast
import argparse

sys.path.insert(1, 'C:\\Users\\bbenja\\Desktop\\Chen - PilotNet\\Autopilot-TensorFlow-master\\SWIN_TF')
from models import *
from random import randint

textures_path = "C:\\Users\\bbenja\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\" \
                "site-packages\\metadrive\\assets\\textures\\"

road_new = textures_path + "sci\\road_texture_"
road_original = textures_path + "sci\\color.jpg"

image_list = []
image_counter = 0
control_list = []
recording = False
network = False
model = None
SEED = 200
MODEL_PATH = "models_100_ep_pretrain/2_256"
IMAGE_DIMS = (84, 84)
LANE_NUM = 1
LANE_WIDTH = 3.5
folder = f"{SEED}_{LANE_NUM}_{LANE_WIDTH}"
infractions = dict(out_of_road=0,
                   crash_vehicle=0,
                   crash_object=0,
                   line_crossed=0)
last_step = dict(out_of_road=0,
                 crash_vehicle=0,
                 crash_object=0,
                 on_line=0,
                 position=(0, 0))
autonomy = dict(started=False,
                start_time=time.perf_counter(),
                n_interventions=int(0),
                elapsed_time=float(0),
                distance=float(0)
                )


def randomize_road_texture():
    aug = RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, always_apply=True)
    img = cv2.imread(road_new + str(randint(0, 6)) + ".jpg")
    img = aug(image=img)["image"]
    img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
    cv2.imwrite(road_original, img)


def draw_map(map):
    if not os.path.exists(folder):
        os.makedirs(folder)
    m = draw_top_down_map(map, return_surface=True)
    pygame.image.save(m, f"map_{SEED}_{LANE_NUM}_{LANE_WIDTH}.jpg")


def save_data():
    global image_list, image_counter, control_list
    print(f"Saving... {folder} ")
    draw_map(env.current_map)
    with open(f"{folder}/data.txt", "a") as file:
        for img, c in zip(image_list, control_list):
            img.write(f"{folder}/{image_counter}.jpg")
            file.write(f"{image_counter},{c[0]},{c[1]},{c[2]}\n")
            image_counter += 1

    image_list = []
    control_list = []

    time.sleep(0.1)


config = metadrive.MetaDriveEnv.default_config()

config.update(dict(environment_num=1,
                   use_render=True,
                   #agent_policy=IDMPolicy,
                   manual_control=False,
                   traffic_density=0.0,
                   show_logo=False,
                   start_seed=SEED,
                   offscreen_render=True,
                   # rgb_clip=True,
                   window_size=(640, 480),
                   # random_spawn_lane_index=False,
                   vehicle_config={#"rgb_camera": (1280, 720),
                       "image_source": "rgb_camera",
                   },
                   map_config={BaseMap.LANE_WIDTH: LANE_WIDTH,
                               BaseMap.LANE_NUM: LANE_NUM,
                               BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                               BaseMap.GENERATE_CONFIG: "CCCCCCCCCC",
                               },
                   need_inverse_traffic=True,
                   random_traffic=True,
                   success_reward=0,
                   out_of_road_penalty=0.1,
                   crash_vehicle_penalty=0.1,
                   crash_object_penalty=0.1,
                   driving_reward=0,
                   speed_reward=0,
                   ))



texture = Texture()

def start_neural_network():
    global network, model, autonomy, IMAGE_DIMS
    if model is None:
        print("Loading model at " + MODEL_PATH)
        model = load_model(MODEL_PATH,
                           custom_objects={"PatchExtract": PatchExtract,
                                           "PatchEmbedding": PatchEmbedding,
                                           "SwinBlock": SwinBlock,
                                           "PatchMerging": PatchMerging,
                                           "weighted_loss": weighted_loss})
        IMAGE_DIMS = model.layers[0].input_shape[-3:-1][::-1]
        print(IMAGE_DIMS)

    if not autonomy['started']:
        autonomy['started'] = True
        autonomy['start_time'] = time.perf_counter()
        autonomy['distance'] = 0.0
        last_step['position'] = env.vehicle.position

    network = not network
    if not network:
        autonomy['n_interventions'] += 1
        env.config.update(dict(agent_policy=IDMPolicy))

    print(f"Network status: {network}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='MetaDrive simulator script')
    argparser.add_argument(
        '--modelpath',
        help='Path to neural network model')
    argparser.add_argument(
        '--seed',
        help='Seed for procedural generation')
    args = argparser.parse_args()
    MODEL_PATH = args.modelpath
    SEED = int(args.seed)
    st_wheel = cv2.imread("steering_wheel_image.jpg")
    rows, cols, _ = st_wheel.shape

    config.update(dict(start_seed=SEED))
    env = metadrive.MetaDriveEnv(config=dict(config))
    obs = env.reset()

    cam = env.vehicle.image_sensors["rgb_camera"]

    # env.engine.force_fps.toggle()
    print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in obs.items()})



    #draw_map(env.current_map)
    angle = 0.0
    throttle = 0.

    for i in range(10000):
        env.engine.accept("e", start_neural_network)
        if recording and i > 20:
            network = False
            image_list.append(cam.get_image(env.vehicle))
            control_list.append((info["raw_action"][0], info["raw_action"][1], info["velocity"]))
        if network and i > 20:
            prediction_time = 0
            texture.load(cam.get_image(env.vehicle))
            img = texture.getRamImageAs("RGB").getData()
            img = np.fromstring(img, np.uint8).reshape((84, 84, 3))
            img = cv2.flip(img, 0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img[-35:]
            img = cv2.resize(img, IMAGE_DIMS, interpolation=cv2.INTER_AREA) / 255.0

            if model.name == "LSTM_multiple_input":
                if len(image_list) == 0:
                    image_list.extend([img, img, img, img, img])
                else:
                    del image_list[0]
                    image_list.append(img)

                for i, im in enumerate(image_list):
                    cv2.imshow(f"{i}", im)
                    cv2.waitKey(1)
                images = np.expand_dims(np.asarray(image_list), axis=0)
                angle = model.predict(images)[0][0] * (-1)
            else:
                img = np.expand_dims(img, axis=0)
                start_time = time.perf_counter()
                angle = model.predict(img)[0][0] * (-1)
                prediction_time = time.perf_counter() - start_time

            #throttle = -1/30 * info["velocity"] + 1

            throttle = 0.0
            if info["velocity"] < 50:
                speed = min(1 / (0.3 * abs(angle)), 50)
                if speed > info["velocity"]:
                    throttle = 0.5
                else:
                    throttle = -0.03

            ### VIDEO
            print(f"Predicted Angle: {round(angle, 5)}, \t"
                  f"Speed: {round(info['velocity'], 2)} km/h, \t"
                  f"Prediction Time: {round(prediction_time, 3)} s")
            rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle * 450, 1)
            dst = cv2.warpAffine(st_wheel, rot, (cols, rows))
            cv2.imshow("Predicted Angle", dst)
            cv2.waitKey(1)
            ###

            autonomy['elapsed_time'] = round(time.perf_counter() - autonomy['start_time'], 2)
            autonomy['distance'] += math.sqrt((env.vehicle.position[0] - last_step['position'][0]) ** 2 +
                                              (env.vehicle.position[1] - last_step['position'][1]) ** 2)
        else:
            angle = 0
            throttle = 0
        env.render(
            text={
                # "Auto-Drive": "on" if env.current_track_vehicle.expert_takeover else "off",
                "NN": model.name if network else "off",
                # "FPS": f"{env.engine.force_fps.fps}",
                "Crash vehicle": infractions['crash_vehicle'],
                "Crash object": infractions['crash_object'],
                # "Out_of_road": infractions['out_of_road'],
                "Lines crossed": infractions['line_crossed'],
                "Distance": f"{np.round(autonomy['distance'], 2)}m",
                "Elapsed time": f"{np.round(autonomy['elapsed_time'], 2)}s",
                "N interventions": int(np.floor(autonomy['n_interventions'] / 2)),
            }
        )

        last_step['position'] = env.vehicle.position

        obs, reward, done, info = env.step([angle, throttle])

        if last_step['crash_vehicle'] < info['crash_vehicle']:
            infractions['crash_vehicle'] += 1
        if last_step['crash_object'] < info['crash_object']:
            infractions['crash_object'] += 1
        if last_step['out_of_road'] < info['out_of_road']:
            infractions['out_of_road'] += 1
        on_line = env.vehicle.on_broken_line or env.vehicle.on_white_continuous_line or env.vehicle.on_yellow_continuous_line
        if last_step['on_line'] < on_line:
            infractions['line_crossed'] += 1

        last_step['crash_vehicle'] = info['crash_vehicle']
        last_step['crash_object'] = info['crash_object']
        last_step['out_of_road'] = info['out_of_road']
        last_step['on_line'] = on_line



        if done and info['arrive_dest']:
            autonomy['elapsed_time'] = round(time.perf_counter() - autonomy['start_time'], 2)
            autonomy['distance'] = round(autonomy['distance'], 2)
            break

    # print(f"The observation shape: {obs.shape}.")
    # print(f"The returned reward: {reward}.")
    # print(f"The returned information: {info}.\n")
    print(f"End info:\n\tCrash: {info['crash']}, \n\tOut of Road: {info['out_of_road']}, "
          f"\n\tArrive Dest.: {info['arrive_dest']}, \n\tMax N of Steps: {info['max_step']}. \n"
          f"Autonomy:\n\tLines Crossed: {infractions['line_crossed']}, "
          f"\n\tCrashes: {infractions['crash_object']+infractions['crash_vehicle']},"
          f"\n\tInterventions: {autonomy['n_interventions']}, "
          f"\n\tElapsed Time: {autonomy['elapsed_time']} s, "
          f"\n\tDistance: {autonomy['distance']} m")
    if recording:
        if info['arrive_dest'] or info['max_step']:
            del image_list[-50:]
            del control_list[-50:]
            save_data()
        else:
            print("Not a good dataset")
    env.close()
