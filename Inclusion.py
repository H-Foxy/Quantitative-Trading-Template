import random, numpy as np, torch
import os, time, pandas as pd, yfinance as yf, talib
from talib import abstract
from talib.abstract import *
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
from pandas.tseries.offsets import CustomBusinessDay
import gymnasium as gym, gym_anytrading
from gym_anytrading.envs import TradingEnv, StocksEnv, Actions, Positions 
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import trange
from backtesting import Backtest, Strategy
import math