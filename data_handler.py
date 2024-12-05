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


def read_data(file_path: str, column_number: int, start_time=0, end_time=0, skiprows_number=4, comments='#'):
    '''
    数据文本文件，并返回时间和所需数据

    Args:
        file_path (str): 数据文件路径。
        column_number (int): 要读取的列号（从 0 开始）。
        start_time (float, optional): 筛选的开始时间，默认为 0。
        end_time (float, optional): 筛选的结束时间，默认为 0，表示不限制结束时间。
        skiprows_number (int, optional): 跳过的行数，默认为 4。
        comments (str, optional): 要跳过的注释行符号，默认为 '#'

    Returns:
        np.ndarray: 时间列数据。
        np.ndarray: 指定列的数据。


    Raises:
        ValueError: 当开始时间和结束时间相同时抛出异常。
        FileNotFoundError: 如果文件路径无效。
        IndexError: 如果列号超出范围。
    '''
    # 输入检查
    if start_time == end_time and end_time != 0:
        raise ValueError("开始时间与结束时间相同，请检查 start_time 和 end_time 的输入。")

    try:

        # 加载数据
        data = np.loadtxt(
            file_path, skiprows=skiprows_number, comments=comments)
    except OSError:
        raise FileExistsError(f"无法找到文件: {file_path}")

    # 筛选出大于等于起始时间的数据
    data = data[data[:, 0] >= start_time]

    # 只有在end_time不为0的时候设置结束时间
    if end_time != 0:
        data = data[data[:, 0] <= end_time]

    try:

        # 提取时间列和所需数据列
        time = data[:, 0]  # 一般的文件结构中时间都位于第一列
        data_you_wanted = data[:, column_number]
    except IndexError:
        raise IndexError(f"列号 {column_number} 超出范围，文件中仅有 {data.shape[1]} 列。")

    return time, data_you_wanted


def low_pass(data: np.ndarray,
             period: float,
             time_step: float,
             cut_off_order=3
             ) -> tuple[np.ndarray, np.ndarray]:
    """
    过滤输入数据中的高频误差，输出平滑后的数据。

    使用Butterworth低通滤波器平滑输入数据，去除高频噪声。

    Args:
        data (np.ndarray): 输入数据数组。
        period (float): 数据的周期。
        time_step (float): 采样时间步长。
        cut_off_order (int, optional): 截止频率的阶数，默认为3。

    Returns:
        tuple: 一个包含平滑后的数据的元组 (output_data,)，
               如果发生错误，返回None。
    """
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
    """
    输出数据中的极大值和极小值。

    通过检测局部最大值和最小值，返回数据的极大值和极小值数组。

    Args:
        data (np.ndarray): 输入的数据数组。

    Returns:
        tuple: 包含极大值和极小值数组的元组 (max_values, min_values)，
               如果发生错误，返回 (None, None)。
    """
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
    对数据进行傅里叶变换并计算频率谱，返回归一化幅值及一阶、二阶谐波幅值。

    该函数对输入的数据进行快速傅里叶变换（FFT），并计算数据的频率谱。然后，返回归一化的幅值频谱以及一阶和二阶谐波的幅值。

    Args:
        data (np.ndarray): 输入数据数组，表示时间域信号。
        time_step (float): 采样间隔，单位为秒。
        apmli_precision (int, optional): 幅值的小数位数，默认为4。
        freq_precision (int, optional): 频率的小数位数，默认为2。

    Returns:
        tuple: 包含四个元素的元组：
            - frequency (np.ndarray): 频率数组。
            - normalized_amplitude (np.ndarray): 归一化幅值数组。
            - f1_norm_amplitude (float): 归一化一阶谐波幅值。
            - f2_norm_amplitude (float): 归一化二阶谐波幅值。
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
