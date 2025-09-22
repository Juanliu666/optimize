# -*- coding:utf-8 -*- #
# @time 2025/9/17 21:48
# author:pengda
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Multiply, Permute, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import joblib
import json

current_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(current_dir, "attention_lstm_model.h5")  # 创建Flask应用
scaler_x_path = os.path.join(current_dir, "scaler_X.pkl")  # 创建Flask应用
scaler_y_path = os.path.join(current_dir, "scaler_y.pkl")  # 创建Flask应用


# 自定义注意力机制层（与训练时相同）
def attention_block(inputs, time_steps):
    a = Permute((2, 1))(inputs)  # (batch_size, features, time_steps)
    a = Dense(time_steps, activation='softmax')(a)  # 为每个特征学习权重
    a = Permute((2, 1), name='attention_weights')(a)  # 恢复原始维度
    output = Multiply()([inputs, a])  # 应用注意力权重
    return output


# 加载模型和标准化器
def load_models():
    # 加载模型（注意自定义层）
    model = load_model(model_path,
                       custom_objects={'attention_block': attention_block})

    # 加载标准化器
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    return model, scaler_X, scaler_y


# 定义输入输出列（与训练时相同）
input_features = ['污泥添加比例（%）', 'C含量（%）', '热解温度（℃）']
output_features = [
    '液体产率（%）', '气体产率（%）', '气体中CO2含量（%）', '气体中CH4含量（%）',
    '热解油中酸含量（%）', '热解油中酚含量（%）', '热解油中含氮化合物含量（%）'
]

# 加载模型和标准化器
print("正在加载模型和标准化器...")
model, scaler_X, scaler_y = load_models()
print("模型和标准化器加载完成！")


def predict(input_values: list):
    try:
        print("-------")
        print(input_values)
        new_data = np.array([input_values])

        # 使用标准化器进行标准化
        new_data_scaled = scaler_X.transform(new_data)

        # 重塑数据为LSTM输入格式
        new_data_reshaped = new_data_scaled.reshape(1, 1, new_data_scaled.shape[1])

        # 使用模型进行预测
        prediction_scaled = model.predict(new_data_reshaped)

        # 将预测结果反标准化到原始尺度
        prediction = scaler_y.inverse_transform(prediction_scaled)

        # 准备结果
        results = {}
        for i, feature in enumerate(output_features):
            results[feature] = f"{prediction[0][i]:.4f}"

        # 返回JSON响应
        return {
            'success': True,
            'input_values': dict(zip(input_features, input_values)),
            'predictions': results
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
