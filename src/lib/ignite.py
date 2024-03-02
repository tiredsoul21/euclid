"""Set of custom handlers for Ignite engine"""
import time
from typing import Optional
from ignite.engine import Engine, State
from ignite.engine import Events, EventEnum
from ignite.handlers.timing import Timer

from . import experiences

class EpisodeEvents(EventEnum):
    """Set of episode events"""
    EPISODE_COMPLETED = "episodeCompleted"
    BOUND_REWARD_REACHED = "bound_reward_reached"
    BEST_REWARD_REACHED = "best_reward_reached"

class PeriodEvents(EventEnum):
    """Set of period events"""
    ITERS_10_COMPLETED = "iterations_10_completed"
    ITERS_100_COMPLETED = "iterations_100_completed"
    ITERS_1000_COMPLETED = "iterations_1000_completed"
    ITERS_10000_COMPLETED = "iterations_10000_completed"
    ITERS_100000_COMPLETED = "iterations_100000_completed"

class EndOfEpisodeHandler:
    """ Handler to fire event at the end of the episode """
    def __init__(self, exp_source: experiences.ExperienceSource,
                 alpha: float = 0.98,
                 bound_avg_reward: Optional[float] = None,
                 sub_sample_end_of_episode: Optional[int] = None):
        """
        Construct end-of-episode event handler
        :param exp_source: experience source to use
        :param alpha: smoothing alpha param
        :param bound_avg_reward: optional boundary for average reward
        :param sub_sample_end_of_episode: if given, end of episode 
            event will be subsampled by this amount
        """
        # Check input parameters
        assert isinstance(exp_source, experiences.ExperienceSource)
        assert isinstance(alpha, float)
        assert isinstance(bound_avg_reward, (float, type(None)))
        assert isinstance(sub_sample_end_of_episode, (int, type(None)))
        assert alpha >= 0.0 and alpha <= 1.0

        # Set initial parameters
        self._exp_source = exp_source
        self._alpha = alpha
        self._bound_avg_reward = bound_avg_reward
        self._best_avg_reward = None
        self._sub_sample_end_of_episode = sub_sample_end_of_episode

    def attach(self, engine: Engine):
        """ Attach the engine and register the events """
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)
        engine.register_events(*EpisodeEvents)

        # Set the event to attribute mapping
        State.event_to_attr[EpisodeEvents.EPISODE_COMPLETED] = "episode"
        State.event_to_attr[EpisodeEvents.BOUND_REWARD_REACHED] = "episode"
        State.event_to_attr[EpisodeEvents.BEST_REWARD_REACHED] = "episode"

    def __call__(self, engine: Engine):
        for reward, steps in self._exp_source.pop_rewards_steps():
            # Update the engine state
            engine.state.episode = getattr(engine.state, "episode", 0) + 1
            engine.state.episode_reward = reward
            engine.state.episode_steps = steps
            engine.state.metrics['reward'] = reward
            engine.state.metrics['steps'] = steps

            # Update the smoothed metrics
            self._update_smoothed_metrics(engine, reward, steps)

            # Fire the events
            if self._sub_sample_end_of_episode is None or \
                    engine.state.episode % self._sub_sample_end_of_episode == 0:
                # Fire the episode completed event
                engine.fire_event(EpisodeEvents.EPISODE_COMPLETED)
            if self._bound_avg_reward is not None and \
                    engine.state.metrics['avgReward'] >= self._bound_avg_reward:
                # Fire the bound reward reached event
                engine.fire_event(EpisodeEvents.BOUND_REWARD_REACHED)

            # Update the best average reward
            if self._best_avg_reward is None:
                self._best_avg_reward = engine.state.metrics['avgReward']
            elif self._best_avg_reward < engine.state.metrics['avgReward']:
                engine.fire_event(EpisodeEvents.BEST_REWARD_REACHED)
                self._best_avg_reward = engine.state.metrics['avgReward']

    def _update_smoothed_metrics(self,
                               engine: Engine,
                               reward: float,
                               steps: int):
        """
        Update smoothed metrics
        :param engine: engine to update
        :param reward: reward to use
        :param steps: steps in the episode
        """
        for metric, val in zip(('avgReward', 'avgSteps'), (reward, steps)):
            if metric not in engine.state.metrics:
                # Initialize the metric
                engine.state.metrics[metric] = val
            else:
                # Update the metric -- rolling average
                engine.state.metrics[metric] *= self._alpha
                engine.state.metrics[metric] += (1-self._alpha) * val

class EpisodeFPSHandler:
    """ Handler to update FPS metrics """
    FPS_METRIC = 'fps'
    AVG_FPS_METRIC = 'avg_fps'
    TIME_PASSED_METRIC = 'timePassed'

    def __init__(self,
                 fps_multiplier: float = 1.0,
                 fps_smoothing_alpha: float = 0.98):
        """
        Construct FPS handler
        :param fps_multiplier: multiplier for FPS
        :param fps_smoothing_alpha: smoothing alpha param
        """
        # Check input parameters
        assert isinstance(fps_multiplier, float)
        assert isinstance(fps_smoothing_alpha, float)
        assert fps_multiplier > 0.0
        assert fps_smoothing_alpha >= 0.0 and fps_smoothing_alpha <= 1.0

        # Initialize parameters
        self._timer = Timer(average=True)
        self._fps_multiplier = fps_multiplier
        self._started_time_stamp = time.time()
        self._fps_smoothing_alpha = fps_smoothing_alpha

    def attach(self,
               engine: Engine,
               manual_step: bool = False):
        """
        Attach the FPS handler to the engine
        :param engine: engine to attach to
        :param manual_step: if True, step() method should be called manually
        """
        self._timer.attach(engine, step=None if manual_step else Events.ITERATION_COMPLETED)
        engine.add_event_handler(EpisodeEvents.EPISODE_COMPLETED, self)

    def step(self):
        """
        If manual_step=True on attach(), this method should be used every 
        time we've communicated with environment to get proper FPS
        """
        self._timer.step()

    def __call__(self,
                 engine: Engine):
        """
        Update FPS metrics
        :param engine: engine to update
        """
        time_hack = self._timer.value()
        if engine.state.iteration > 1:
            # Update the FPS / Avg FPS metrics
            fps = self._fps_multiplier / time_hack
            avg_fps = engine.state.metrics.get(self.AVG_FPS_METRIC)
            if avg_fps is None:
                avg_fps = fps
            else:
                avg_fps *= self._fps_smoothing_alpha
                avg_fps += (1-self._fps_smoothing_alpha) * fps

            # Update the metrics
            engine.state.metrics[self.AVG_FPS_METRIC] = avg_fps
            engine.state.metrics[self.FPS_METRIC] = fps
        engine.state.metrics[self.TIME_PASSED_METRIC] = time.time() - self._started_time_stamp
        self._timer.reset()

class PeriodicEvents:
    """
    The same as CustomPeriodicEvent from ignite.contrib, but use true amount of iterations,
    which is good for TensorBoard
    """

    INTERVAL_TO_EVENT = {
        10: PeriodEvents.ITERS_10_COMPLETED,
        100: PeriodEvents.ITERS_100_COMPLETED,
        1000: PeriodEvents.ITERS_1000_COMPLETED,
        10000: PeriodEvents.ITERS_10000_COMPLETED,
        100000: PeriodEvents.ITERS_100000_COMPLETED,
    }

    def attach(self,
               engine: Engine):
        """
        Attach to engine and register events"
        :param engine: engine to attach to
        """
        # Attach the engine and register the events
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)
        engine.register_events(*PeriodEvents)

        # Add the event to attribute mapping
        for e in PeriodEvents:
            State.event_to_attr[e] = "iteration"

    def __call__(self, engine: Engine):
        """
        For every event in INTERVAL_TO_EVENT, fire the event
        if the iteration is divisible by the interval
        :param engine: engine to use
        """
        for period, event in self.INTERVAL_TO_EVENT.items():
            if engine.state.iteration % period == 0:
                engine.fire_event(event)
