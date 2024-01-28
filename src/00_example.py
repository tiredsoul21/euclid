#!/usr/bin/env python3
import pathlib
import argparse
import gym.wrappers

import torch
import torch.optim as optim

from lib import data
from lib import models
from lib import agents
from lib import actions
from lib import experiences
from lib import environments

SAVES_DIR = pathlib.Path("output")

# How many bars to feed into the model
BAR_COUNT = 50

# EPSILON GREEDY - for exploration
EPS_START = 1.0
EPS_FINAL = 0.1
EPS_STEPS = 1000000

LEARNING_RATE = 0.0001
GAMMA = 0.99

REWARD_STEPS = 2

REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(      "--cuda",                help="Enable cuda", default=False, action="store_true")
    parser.add_argument("-p", "--path", required=True, help="Directory or file of price data")
    parser.add_argument("-v", "--val",                 help="Validation data, default=path/val/")
    parser.add_argument("-t", "--test",                help="Test data, default=path/test/")
    parser.add_argument("-r", "--run",  required=True, help="Run name")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    # Create output directory
    savesPath = SAVES_DIR / f"00-{args.run}"
    savesPath.mkdir(parents=True, exist_ok=True)

    # Set data paths
    dataPath = pathlib.Path(args.path)
    dataFolder = dataPath
    
    # If dataPath is a file, use fetch containing directory
    if dataPath.is_file():
        dataFolder = dataPath.parent

    # Set validation path
    if args.val is None:
        valPath = dataFolder / "val"
    else:
        valPath = pathlib.Path(args.val)

    # Set test path
    if args.test is None:
        testPath = dataFolder / "test"
    else:
        testPath = pathlib.Path(args.test)

    # Create Environment
    if dataPath.is_file():
        # Import data from file to dictionary
        index = dataPath.stem
        priceData = {index: data.readCSV(str(dataPath)) }
        env = environments.StocksEnv(priceData, barCount=BAR_COUNT)
        env._state.barCount = BAR_COUNT

        env_tst = environments.StocksEnv(priceData, barCount=BAR_COUNT)
    elif dataPath.is_dir():
        env = environments.StocksEnv.fromDirectory(dataPath, barCount=BAR_COUNT)
        env_tst = environments.StocksEnv.fromDirectory(dataPath, barCount=BAR_COUNT)
    else:
        raise RuntimeError("No data to train on")

    # Create validation environment
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    envVal = environments.StocksEnv.fromDirectory(valPath, barCount=BAR_COUNT)

    # Create the networks
    net = models.DQNConv1D(env.observation_space.shape, env.action_space.n).to(device)
    targetNet = models.TargetNet(net)

    # Create the action selector
    selector = actions.EpsilonGreedyActionSelector(EPS_START)
    epsilonTracker = actions.EpsilonTracker(selector, EPS_START, EPS_FINAL, EPS_STEPS)

    # Create the agent
    agent = agents.DQNAgent(net, selector, device=device)

    # Create the experience source
    exp_source = experiences.ExperienceSourceFirstLast(env, agent, GAMMA, stepsCount=REWARD_STEPS)
    buffer = experiences.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)

    # Create the optimizer
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # def process_batch(engine, batch):
    #     optimizer.zero_grad()
    #     loss_v = common.calc_loss(
    #         batch, net, tgt_net.target_model,
    #         gamma=GAMMA ** REWARD_STEPS, device=device)
    #     loss_v.backward()
    #     optimizer.step()
    #     epsilonTracker.frame(engine.state.iteration)

    #     if getattr(engine.state, "eval_states", None) is None:
    #         eval_states = buffer.sample(STATES_TO_EVALUATE)
    #         eval_states = [np.array(transition.state, copy=False)
    #                        for transition in eval_states]
    #         engine.state.eval_states = np.array(eval_states, copy=False)

    #     return {
    #         "loss": loss_v.item(),
    #         "epsilon": selector.epsilon,
    #     }

    # engine = Engine(process_batch)
    # tb = common.setup_ignite(engine, exp_source, f"conv-{args.run}",
    #                          extra_metrics=('values_mean',))

    # @engine.on(ptan.ignite.PeriodEvents.ITERS_1000_COMPLETED)
    # def sync_eval(engine: Engine):
    #     tgt_net.sync()

    #     mean_val = common.calc_values_of_states(
    #         engine.state.eval_states, net, device=device)
    #     engine.state.metrics["values_mean"] = mean_val
    #     if getattr(engine.state, "best_mean_val", None) is None:
    #         engine.state.best_mean_val = mean_val
    #     if engine.state.best_mean_val < mean_val:
    #         print("%d: Best mean value updated %.3f -> %.3f" % (
    #             engine.state.iteration, engine.state.best_mean_val,
    #             mean_val))
    #         path = saves_path / ("mean_value-%.3f.data" % mean_val)
    #         torch.save(net.state_dict(), path)
    #         engine.state.best_mean_val = mean_val

    # @engine.on(ptan.ignite.PeriodEvents.ITERS_10000_COMPLETED)
    # def validate(engine: Engine):
    #     res = validation.validation_run(env_tst, net, device=device)
    #     print("%d: tst: %s" % (engine.state.iteration, res))
    #     for key, val in res.items():
    #         engine.state.metrics[key + "_tst"] = val
    #     res = validation.validation_run(env_val, net, device=device)
    #     print("%d: val: %s" % (engine.state.iteration, res))
    #     for key, val in res.items():
    #         engine.state.metrics[key + "_val"] = val
    #     val_reward = res['episode_reward']
    #     if getattr(engine.state, "best_val_reward", None) is None:
    #         engine.state.best_val_reward = val_reward
    #     if engine.state.best_val_reward < val_reward:
    #         print("Best validation reward updated: %.3f -> %.3f, model saved" % (
    #             engine.state.best_val_reward, val_reward
    #         ))
    #         engine.state.best_val_reward = val_reward
    #         path = saves_path / ("val_reward-%.3f.data" % val_reward)
    #         torch.save(net.state_dict(), path)


    # event = ptan.ignite.PeriodEvents.ITERS_10000_COMPLETED
    # tst_metrics = [m + "_tst" for m in validation.METRICS]
    # tst_handler = tb_logger.OutputHandler(
    #     tag="test", metric_names=tst_metrics)
    # tb.attach(engine, log_handler=tst_handler, event_name=event)

    # val_metrics = [m + "_val" for m in validation.METRICS]
    # val_handler = tb_logger.OutputHandler(
    #     tag="validation", metric_names=val_metrics)
    # tb.attach(engine, log_handler=val_handler, event_name=event)

    # engine.run(common.batch_generator(buffer, REPLAY_INITIAL, BATCH_SIZE))
