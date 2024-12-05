'''
本模块包含了通用的数据前处理工具：
read_data 数据读取与输出

low_pass 构造一个低通滤波器以去除数据中的高频成分

max_min 输出数据中的极大极小值数组

fft_frequency_spectrum 对输入的数据进行傅里叶变换

:author: SUGAR-ADDICT
:date: 2024-11-23
'''
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq


def read_data(file_path: str, list_number: int, start_time=0, end_time=100, skiprows_number=4, comments='#'):
    '''
    数据文本文件，并返回时间和所需数据

    :param file_path: 文件路径
    :param list_number: 要读取的列数
    :param start_time: 文件的开始时间
    :param end_time: 文件的结束时间
    :param skiprows_number: 要跳过的行数
    :param commets: 要跳过的符号

    :return time: 时间数据
    :return data_you_wanted: 所需的数据数组
    '''
    # 加载数据
    data = np.loadtxt(file_path, skiprows=skiprows_number, comments='#')

    # 筛选出大于等于起始时间的数据
    data = data[data[:, 0] >= start_time]
    data = data[data[:, 0] <= end_time]

    # 提取时间列和所需数据列
    time = data[:, 0]  # 一般的文件结构中时间都位于第一列
    data_you_wanted = data[:, list_number]

    return time, data_you_wanted


def low_pass(data: np.ndarray,
             period: float,
             time_step: float,
             cut_off_order=3
             ) -> tuple[np.ndarray, np.ndarray]:
    '''
    过滤输入数据中的高频误差，输出平滑后的数据

    :param data: 输入数据数组
    :param period: 数据的周期
    :param cut_off_order: 截至频率的阶数，默认为3

    :return output_data: 平滑后的数据数组
    '''
    try:
        # 计算采样率
        sampling_rate = 1/time_step

        # 计算截止频率，默认为信号频率的三倍
        cut_off_frequency = cut_off_order / period

        # 计算归一化截止频率
        Wn = 2 * cut_off_frequency / sampling_rate

        # 构造 Butterworth 低通滤波器（3阶）
        b, a = signal.butter(3, Wn, 'lowpass')

        # 应用滤波器
        output_data = signal.filtfilt(b, a, data)

        return output_data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def max_min(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    输出数据中的极大极小值的平均值

    :param data: 输入的数据数组

    :return max_values: 极大值数组
    :return min_values: 极小值数组
    '''
    try:
        max_index = signal.argrelextrema(data, np.greater)[0]  # 找到所有最大值的索引
        min_index = signal.argrelextrema(data, np.less)[0]  # 找到所有最小值的索引

        max_values = data[max_index]  # 找到所有最大值
        min_values = data[min_index]  # 找到所有最小值

        return max_values, min_values
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def fft_frequency_spectrum(
    data: np.ndarray,
    time_step: float,
    apmli_precision=4,
    freq_precision=2
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    对数据进行傅里叶变换

    :param data: 输入的数据数组
    :param time_step: 采样间隔
    :param apmli_precision: 幅值的小数位数, 默认为4
    :param freq_precision: 频率的小数位数, 默认为2

    :return frequency: 频率
    :return normalized_amplitude: 归一化幅值
    :return f1_norm_amplitude:  归一化一阶谐波幅值
    :return f2_norm_amplitude:  归一化二阶谐波幅值
    """
    # 信号长度和采样率
    signal_length = len(data)
    sampling_rate = 1 / time_step

    # 计算傅里叶变换
    fourier_transform = fft(data)
    frequency = fftfreq(signal_length, 1 / sampling_rate)[:signal_length // 2]
    normalized_amplitude = 2.0 / signal_length * \
        np.abs(fourier_transform[0:signal_length // 2])  # 归一化并取半处理

    # 找到一阶谐波频率与幅值
    f1_index = np.argmax(np.abs(fourier_transform[:signal_length // 2]))
    f1 = frequency[f1_index]
    f1_amplitude = np.abs(fourier_transform[f1_index])
    f1_norm_amplitude = 2.0 / signal_length*f1_amplitude

    # 二阶谐波频率与幅值
    f2 = 2*f1
    f2_index = np.where(frequency == f2)[0][0]
    f2_amplitude = np.abs(fourier_transform[f2_index])
    f2_norm_amplitude = 2.0 / signal_length*f2_amplitude

    # 输出幅值最大频率和幅值
    print(
        rf"1st harmonic frequency: {f1:.{freq_precision}f} Hz , amplitude: {f1_norm_amplitude:.{apmli_precision}f}")
    print(
        rf"2nd harmonic frequency: {f2:.{freq_precision}f} Hz , amplitude: {f2_norm_amplitude:.{apmli_precision}f}")

    return frequency, normalized_amplitude, f1_norm_amplitude, f2_norm_amplitude
