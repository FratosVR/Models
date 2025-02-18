import tensorflow as tf
import numpy as np
import time
from itertools import product
import matplotlib.pyplot as plt


class RNNTrainer:
    def __init__(self):
        self.HP_ACTIVATION = hp.HParam('activation', hp.Discrete(
            ["tanh", "linear", "relu", "sigmoid"]))
        self.HP_USE_BIAS = hp.HParam('use_bias', hp.Discrete([True, False]))
        self.__use_bias_list: list[bool] = [True, False]
        self.__kernel_initializer_list: list[str] = [
            "glorot_uniform", "he_normal", "he_uniform"]
        self.__recurrent_initializer_list: list[str] = [
            "glorot_uniform", "he_normal", "he_uniform"]
        self.__bias_initializer_list: list[str] = [
            "glorot_uniform", "he_normal", "he_uniform"]
        self.__kernel_regularizer_list: list[str] = ["l1", "l2", "l1_l2", None]
        self.__recurrent_regularizer_list: list[str] = [
            "l1", "l2", "l1_l2", None]
        self.__bias_regularizer_list: list[str] = ["l1", "l2", "l1_l2", None]
        self.__activity_regularizer_list: list[str] = [
            "l1", "l2", "l1_l2", None]
        self.__kernel_constraint_list: list[str] = [
            "max_norm", "non_neg", None]
        self.__recurrent_constraint_list: list[str] = [
            "max_norm", "non_neg", None]
        self.__bias_constraint_list: list[str] = ["max_norm", "non_neg", None]
        self.__dropout_list: np.ndarray = np.arange(0, 1, 0.1)
        self.__recurrent_dropout_list: np.ndarray = np.arange(0, 1, 0.1)
        self.__unroll_list: list[bool] = [True, False]

    def train(self, X, y):
        # TODO
        pass
