import enum

import gym
import gym.spaces
from gym.utils import seeding
from gym.envs.registration import EnvSpec
import numpy as np

from lib import data

# Default values
DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION = 0.0
DEFAULT_SEED = 42

# Actions available to the agent
class Actions(enum.Enum):
    Hold = 0
    Buy = 1
    Close = 2

class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices: data.Prices,
                 barCount: int = DEFAULT_BARS_COUNT,
                 commission: float = DEFAULT_COMMISSION,
                 resetOnClose: bool = True,
                 randomOffset: bool = True,
                 rewardOnClose: bool = False,
                 volumes: bool = False):
        """
        :param prices: (dict) instrument -> price data
        :param barCount: (int) window size
        :param commission: (float) value of the commission
        :param resetOnClose: (bool) reset position on each close
        :param randomOffset: (bool) randomize position on reset
        :param rewardOnClose: (bool) give reward only at the end of the position
        :param volumes: (bool) use volumes
        """

        # Check input parameters
        assert isinstance(prices, dict)
        assert isinstance(barCount, int)
        assert isinstance(commission, float)
        assert isinstance(resetOnClose, bool)
        assert isinstance(randomOffset, bool)
        assert isinstance(rewardOnClose, bool)
        assert isinstance(volumes, bool)
        assert barCount > 0
        assert commission >= 0.0

        # Set random seed
        self.seed(DEFAULT_SEED)

        # Set prices (dict of instrument -> price data)
        self.prices = prices

        # Build action space
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self._state = StockState(barCount=barCount,
                                 commission=commission,
                                 resetOnClose=resetOnClose,
                                 rewardOnClose=rewardOnClose,
                                 volumes=volumes)
        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=self._state.shape,
                                                dtype=np.float32)
        self.randomOffset = randomOffset

    def reset(self):
        # make selection of the instrument and price data
        self._instrument = np.random.choice(list(self.prices.keys()))
        prices = self.prices[self._instrument]

        # set offset if randomOffset is True
        bars = self._state.barCount
        if self.randomOffset:
            # offset keeps at least bars distance from the beginning and from the end
            offset = np.random.choice(prices.high.shape[0] - 2 * bars) + bars
        else:
            offset = bars

        # reset state and return observation
        self._state.reset(prices, offset)
        return self._state.encode()

    def step(self, actionIdx: int):
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
        action = Actions(actionIdx)
        reward, done = self._state.step(action)

        # Get observation and info and return
        obs = self._state.encode()
        info = {
            "instrument": self._instrument,
            "offset": self._state.offset
        }
        truncated = None
        return obs, reward, done, truncated, info

    def render(self, mode='human', close=False):
        """
        Not implemented -- no visualization yet implemented
        """
        pass

    def close(self):
        """
        Not implemented -- no resources to release
        Required by gym interface
        """
        pass

    def seed(self, seed=None):
        """
        Create random or static random seeds, return list of seeds
        If seed is given, the random number generator is initialized
        with static seed
        :param seed: seed to use (default: None -- random seed)
        :return: list of seeds
        """
        # Create random seed
        np_random, seed1 = seeding.np_random(seed)
        # Create second seed in range [0, 2**31)
        seed2 = hash(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    @classmethod
    def fromDirectory(cls, directory, **kwargs):
        """
        Create environment from directory with price data
        :param directory: directory with price data
        :param kwargs: arguments for the environment
        :return: environment
        """
        prices = {
            file: data.loadRelative(file)
            for file in data.findFiles(directory)
        }
        return StocksEnv(prices, **kwargs)

class StockState:
    def __init__(self,
                 barCount: int,
                 commission: float,
                 resetOnClose: bool,
                 rewardOnClose: bool = True,
                 volumes: bool = True):

        # Check input parameters
        assert isinstance(barCount, int)
        assert isinstance(commission, float)
        assert isinstance(resetOnClose, bool)
        assert isinstance(rewardOnClose, bool)
        assert isinstance(volumes, bool)
        assert barCount > 0
        assert commission >= 0.0

        # Set initial parameters
        self.barCount = barCount
        self.commission = commission
        self.resetOnClose = resetOnClose
        self.rewardOnClose = rewardOnClose
        self.volumes = volumes

    def reset(self, prices: data.Prices, offset: int):
        """
        Reset state to the beginning of the price data
        :param prices: price data
        :param offset: offset to start from
        """

        # Check input parameters
        assert isinstance(prices, data.Prices)
        assert isinstance(offset, int)
        assert offset >= self.barCount-1

        # Set initial parameters
        self.havePosition = False
        self.openPrice = 0.0
        self.prices = prices
        self.offset = offset

    @property
    def shape(self):
        """
        Return shape of the state
        """
        size = 5
        if self.volumes:  # add volumes
            size += 1
        return (size, self.barCount)

    def encode(self):
        """
        Encode current state a dictionary of numpy arrays
        """

        # Create dictionary
        encodedData = {
            'priceData': np.zeros(shape=(3, self.barCount), dtype=np.float32),
            'volumeData': np.zeros(shape=(1, self.barCount), dtype=np.float32) if self.volumes else np.array(0, dtype=np.float32),
            'hasPosition': np.array([0.0], dtype=np.float32),
            'position': np.array([0.0], dtype=np.float32)
        }

        # Set values
        start = self.offset - (self.barCount - 1)
        stop = self.offset + 1
        encodedData['priceData'][0] = self.prices.high[start:stop]
        encodedData['priceData'][1] = self.prices.low[start:stop]
        encodedData['priceData'][2] = self.prices.close[start:stop]

        # Set volumes if needed
        if self.volumes:
            encodedData['volumeData'][0] = self.prices.volume[start:stop]

        # Set position if needed
        if self.havePosition:
            encodedData['hasPosition'][0] = 1.0
            encodedData['position'][0] = self._currentClose() / self.openPrice - 1.0

        return encodedData

    def _currentClose(self):
        """
        Calculate real close price for the current bar
        """
        open = self.prices.open[self.offset]
        relativeClose = self.prices.close[self.offset]
        return open * (1.0 + relativeClose)

    def step(self, action: Actions):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action: action to perform
        :return: reward, done
        """

        # Check input parameters
        assert isinstance(action, Actions)

        # Initialize
        reward = 0.0
        done = False
        close = self._currentClose()

        # Handle position change
        if action == Actions.Buy and not self.havePosition:
            # get a position, price, and charge a commission
            self.havePosition = True
            self.openPrice = close
            reward -= self.commission
        elif action == Actions.Close and self.havePosition:
            # close a position, price, and charge a commission
            reward -= self.commission
            done |= self.resetOnClose

            # calculate reward
            if self.rewardOnClose:
                reward += 100.0 * (close / self.openPrice - 1.0)
            self.havePosition = False
            self.openPrice = 0.0

        # Check if the NEXT bar is out of bounds...if so this is the last step
        done |= self.offset + 1 >= self.prices.close.shape[0] - 1

        # Move forward
        self.offset += 1
        previousClose = close
        close = self._currentClose()

        if self.havePosition and not self.rewardOnClose:
            reward += 100.0 * (close / previousClose - 1.0)

        return reward, done
