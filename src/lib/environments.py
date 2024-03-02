"""Defines environments for the reinforcement learning models"""
import enum

import gym
import gym.spaces
from gym.utils import seeding
import numpy as np

from lib import data
from lib import stockTools

# Default values
DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION = 0.0
DEFAULT_SEED = 42

# Actions available to the agent
class StockActions(enum.Enum):
    """Enumeration of Stock Actions"""
    HOLD = 0
    BUY = 1
    SELL = 2

class StocksEnv(gym.Env):
    """ Environment for stock trading """
    metadata = {'render.modes': ['human']}

    def __init__(self, prices: data.Prices,
                 bar_count: int = DEFAULT_BARS_COUNT,
                 commission: float = DEFAULT_COMMISSION,
                 reset_on_close: bool = True,
                 random_offset: bool = True,
                 reward_on_close: bool = False,
                 volumes: bool = False):
        """
        :param prices: (dict) instrument -> price data
        :param bar_count: (int) window size
        :param commission: (float) value of the commission
        :param reset_on_close: (bool) reset position on each close
        :param random_offset: (bool) randomize position on reset
        :param reward_on_close: (bool) give reward only at the end of the position
        :param volumes: (bool) use volumes
        """

        # Check input parameters
        assert isinstance(prices, dict)
        assert isinstance(bar_count, int)
        assert isinstance(commission, float)
        assert isinstance(reset_on_close, bool)
        assert isinstance(random_offset, bool)
        assert isinstance(reward_on_close, bool)
        assert isinstance(volumes, bool)
        assert bar_count > 0
        assert commission >= 0.0

        # Set random seed
        self.seed(DEFAULT_SEED)

        # Set prices (dict of instrument -> price data)
        self.prices = prices

        # Build action space
        self.action_space = gym.spaces.Discrete(n=len(StockActions))
        self._state = StockState(bar_count=bar_count,
                                 commission=commission,
                                 reset_on_close=reset_on_close,
                                 reward_on_close=reward_on_close,
                                 volumes=volumes)
        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=(3, bar_count),
                                                dtype=np.float32)
        self.random_offset = random_offset

        self.moving_avg_flag = False
        self.moving_avg_window = 10
        self.moving_avg = {}

        self._instrument = None

    def reset(self):
        # make selection of the instrument and price data
        self._instrument = np.random.choice(list(self.prices.keys()))
        prices = self.prices[self._instrument]
        technicals = {}

        # set offset if random_offset is True
        bars = self._state.bar_count
        offset = np.random.choice(prices.high.shape[0] - 2 * bars) \
            + bars if self.random_offset else bars

        if self.moving_avg_flag:
            technicals['moving_avg'] = self.moving_avg[self._instrument]

        # reset state and return observation
        self._state.reset(prices, technicals, offset)
        return self._state.encode()

    def step(self, action: int):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param actionIdx: index of the action to perform
        :return: observation, reward, done, info
        """
        # Check input parameters
        # assert isinstance(actionIdx, int)
        # assert self.action_space.contains(actionIdx)

        # Perform our action and get reward
        action = StockActions(action)
        reward, done = self._state.step(action)

        # Get observation and info and return
        obs = self._state.encode()
        info = {
            "instrument": self._instrument,
            "offset": self._state.offset
        }
        truncated = None
        return obs, reward, done, truncated, info

    def render(self):
        """
        Not implemented -- no visualization yet implemented
        """

    def close(self):
        """
        Not implemented -- no resources to release
        Required by gym interface
        """

    def seed(self, seed=None):
        """
        Create random or static random seeds, return list of seeds
        If seed is given, the random number generator is initialized
        with static seed
        :param seed: seed to use (default: None -- random seed)
        :return: list of seeds
        """
        # Create random seed
        _, seed1 = seeding.np_random(seed)
        # Create second seed in range [0, 2**31)
        seed2 = hash(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def state_shape(self):
        """
        Return shape of the state
        """
        return self._state.shape

    @classmethod
    def from_directory(cls, directory, **kwargs):
        """
        Create environment from directory with price data
        :param directory: directory with price data
        :param kwargs: arguments for the environment
        :return: environment
        """
        load_rel_kwargs = {key: kwargs.pop(key) \
                           for key in ['sep', 'fix_open_price'] if key in kwargs}
        prices = {
            file: data.load_relative(file, **load_rel_kwargs)
            for file in data.find_files(directory)
        }
        return StocksEnv(prices, **kwargs)

    def use_moving_avg(self, toggle: bool = True, window: int = 10):
        """
        Moving average of the close price
        :param toggle: (bool) if True, return moving average
        :param window: (int) window size
        """
        # Check input parameters
        assert isinstance(toggle, bool)
        assert isinstance(window, int) and window > 0

        self.moving_avg_flag = toggle
        self.moving_avg_window = window

        if toggle:
            # Create a moving average dictionary
            self.moving_avg = {}
            for instrument, prices in self.prices.items():
                print(f'Calculating moving average for {instrument}')
                self.moving_avg[instrument] = stockTools.moving_avg(prices, window)
        else:
            self.moving_avg = None
        self.reset()

class StockState:
    """ State of the stock environment """
    def __init__(self,
                 bar_count: int,
                 commission: float,
                 reset_on_close: bool,
                 reward_on_close: bool = True,
                 volumes: bool = True):

        # Check input parameters
        assert isinstance(bar_count, int)
        assert isinstance(commission, float)
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        assert isinstance(volumes, bool)
        assert bar_count > 0
        assert commission >= 0.0

        # Set initial parameters
        self.bar_count = bar_count
        self.commission = commission
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes
        self.technicals = {}

        # Set initial state
        self.has_position = False
        self.open_price = 0.0
        self.prices = None
        self.offset = 0


    def reset(self,
              prices: data.Prices,
              technicals: dict,
              offset: int):
        """
        Reset state to the beginning of the price data
        :param prices: price data
        :param offset: offset to start from
        """

        # Check input parameters
        assert isinstance(prices, data.Prices)
        assert isinstance(offset, int)
        assert offset >= self.bar_count-1

        # Set initial parameters
        self.has_position = False
        self.open_price = 0.0
        self.prices = prices
        self.offset = offset
        self.technicals = technicals

    @property
    def shape(self):
        """
        Return shape of the state
        """
        shapes = {}
        shapes['priceData'] = (3, self.bar_count)
        shapes['volumeData'] = (self.bar_count,) if self.volumes else None
        shapes['moving_avg'] = (self.bar_count,) if 'moving_avg' in self.technicals else None
        return shapes

    def encode(self):
        """
        Encode current state a dictionary of numpy arrays
        """
        data_range = range(self.offset - (self.bar_count - 1), self.offset + 1)

        # Create dictionary
        encoded_data = {
            'priceData': np.zeros(shape=(3, self.bar_count), dtype=np.float32),
            'volumeData': self.prices.volume[data_range] \
                if self.volumes else np.array(0, dtype=np.float32),
            'moving_avg': self.technicals['moving_avg'][data_range] \
                if 'moving_avg' in self.technicals else np.array(0, dtype=np.float32),
            'hasPosition': np.array([0.0], dtype=np.float32),
            'position': np.array([0.0], dtype=np.float32)
        }

        # Set values
        encoded_data['priceData'][0] = self.prices.high[data_range]
        encoded_data['priceData'][1] = self.prices.low[data_range]
        encoded_data['priceData'][2] = self.prices.close[data_range]

        # Set position if needed
        if self.has_position:
            encoded_data['hasPosition'][0] = 1.0
            encoded_data['position'][0] = self._current_close() / self.open_price - 1.0

        return encoded_data

    def _current_close(self):
        """
        Calculate real close price for the current bar
        """
        open_price = self.prices.open[self.offset]
        relative_close = self.prices.close[self.offset]
        return open_price * (1.0 + relative_close)

    def step(self, action: StockActions):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action: action to perform
        :return: reward, done
        """

        # Check input parameters
        assert isinstance(action, StockActions)

        # Initialize
        reward = 0.0
        done = False
        close = self._current_close()

        # Handle position change
        if action == StockActions.BUY and not self.has_position:
            # get a position, price, and charge a commission
            self.has_position = True
            self.open_price = close
            reward -= self.commission
        elif action == StockActions.SELL and self.has_position:
            # close a position, price, and charge a commission
            reward -= self.commission
            done |= self.reset_on_close

            # calculate reward
            if self.reward_on_close:
                reward += 100.0 * (close / self.open_price - 1.0)
            self.has_position = False
            self.open_price = 0.0

        # Check if the NEXT bar is out of bounds...if so this is the last step
        done |= self.offset + 1 >= self.prices.close.shape[0] - 1

        # Move forward
        self.offset += 1
        previous_close = close
        close = self._current_close()

        if self.has_position and not self.reward_on_close:
            reward += 100.0 * (close / previous_close - 1.0)

        # Close the position if the last bar is reached
        if done and self.has_position:
            reward -= self.commission
            reward += 100.0 * (close / self.open_price - 1.0)
            self.has_position = False
            self.open_price = 0.0

        return reward, done
