import csv
import datetime
from collections import defaultdict

import torch
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter
import wandb

COMMON_TRAIN_FORMAT = [('frame', 'F', 'int'), ('step', 'S', 'int'),
                       ('episode', 'E', 'int'), ('episode_length', 'L', 'int'),
                       ('episode_reward', 'R', 'float'),
                       ('buffer_size', 'BS', 'int'), ('fps', 'FPS', 'float'),
                       ('total_time', 'T', 'time')]

COMMON_EVAL_FORMAT = [('frame', 'F', 'int'), ('step', 'S', 'int'),
                      ('episode', 'E', 'int'), ('episode_length', 'L', 'int'),
                      ('episode_reward', 'R', 'float'),
                      ('total_time', 'T', 'time'),
                      ('success_rate', 'SR', 'float')]


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    """
    实验指标组合管理类。
    
    主要功能：
    1. 收集并记录实验过程中的各类指标（如 Reward, Loss, FPS 等）。
    2. 自动计算各指标在统计周期内的平均值。
    3. 支持多渠道输出：CSV 文件持久化、控制台格式化打印、WandB 云端同步。
    4. 具备断点续训兼容性：自动处理 CSV 文件，删除重复步数的数据。
    """

    def __init__(self, csv_file_name, formating, use_wandb):
        """
        初始化 MetersGroup 实例。

        参数说明:
            csv_file_name (pathlib.Path): 存储日志的 CSV 文件路径对象。
            formating (list): 一个三元组列表 [(key, display_name, type), ...]，定义控制台打印的格式。
            use_wandb (bool): 是否启用 Weights & Biases 远程实验追踪。
        """
        self._csv_file_name = csv_file_name
        self._formating = formating
        # 使用 defaultdict 自动创建 AverageMeter，用于存储累加值并计算平均
        self._meters = defaultdict(AverageMeter)
        self._csv_file = None
        self._csv_writer = None
        self.use_wandb = use_wandb

    def log(self, key, value, n=1):
        """
        记录一条数据。

        参数说明:
            key (str): 指标名称。
            value (float/int): 指标数值。
            n (int): 计数值（通常用于 Batch 数据，表示该值包含了多少个样本的平均）。
        """
        self._meters[key].update(value, n)

    def _prime_meters(self):
        """
        内部逻辑说明: 准备并清洗数据。
        
        逻辑步骤:
        1. 遍历所有 Meter，提取其平均值。
        2. 将 key 中的前缀（train_/eval_）剥离，并将路径符号 '/' 替换为下划线 '_'。
        
        返回:
            dict: 处理好格式的键值对数据。
        """
        data = dict()
        for key, meter in self._meters.items():
            # 移除键名中的前缀，以便于统一 CSV 的列名格式
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()  # 获取当前周期的平均值
        return data

    def _remove_old_entries(self, data):
        """
        内部逻辑说明: 处理断点续训时的 CSV 数据冲突。
        
        WARNING: 该函数会重写 CSV 文件。
        如果当前保存的数据 episode 与旧文件冲突，它会删除旧文件中所有 >= 当前 episode 的行。
        
        参数说明:
            data (dict): 当前准备写入的数据。
        """
        rows = []
        # 以只读模式读取旧文件内容
        with self._csv_file_name.open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 如果旧文件中的 episode 已经达到了当前值，停止读取后续内容
                if float(row['episode']) >= data['episode']:
                    break
                rows.append(row)
        
        # 以写模式重开文件，抹除旧内容，重新写入保留的行
        with self._csv_file_name.open('w') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=sorted(data.keys()),
                                    restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _dump_to_csv(self, data):
        """
        将数据持久化到本地 CSV 文件。
        
        INFO: 首次调用时会检查文件是否存在。如果存在，会触发 _remove_old_entries。
        
        参数说明:
            data (dict): 整理后的键值对数据。
        """
        if self._csv_writer is None:
            should_write_header = True
            # 如果文件已存在，说明是断点续训，需要处理冲突
            if self._csv_file_name.exists():
                self._remove_old_entries(data)
                should_write_header = False

            # 以追加模式打开文件
            self._csv_file = self._csv_file_name.open('a')
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.0)
            if should_write_header:
                self._csv_writer.writeheader()

        self._csv_writer.writerow(data)
        self._csv_file.flush()  # 强制刷新缓冲区，确保程序崩溃时数据也已写入磁盘

    def _format(self, key, value, ty):
        """
        根据指定的类型对数值进行字符串格式化。
        
        参数说明:
            key (str): 显示的名称。
            value (any): 原始数值。
            ty (str): 目标类型，支持 'int', 'float', 'time'。
        """
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        elif ty == 'time':
            # 将秒数转换为 HH:MM:SS 格式
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{key}: {value}'
        else:
            raise f'invalid format type: {ty}'

    def _dump_to_console(self, data, prefix):
        """
        将数据格式化为易读的表格行并打印到控制台。
        
        逻辑说明:
        根据 self._formating 的定义，依次提取 data 中的值并按格式拼接。
        """
        # 训练日志显示黄色，评估日志显示绿色
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = [f'| {prefix: <14}']
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(' | '.join(pieces))

    def dump(self, step, prefix):
        """
        核心外部调用接口：将当前周期的所有指标清算并输出。
        
        输入参数:
            step (int): 当前的总步数（frame）。
            prefix (str): 数据前缀，通常为 'train' 或 'eval'。
            
        内部流程:
        1. 检查是否有数据，无则跳过。
        2. 调用 _prime_meters 计算平均值。
        3. 如果开启了 wandb，同步远程。
        4. 写入 CSV。
        5. 打印到屏幕。
        6. 清空当前计数器，等待下一个周期。
        """
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['frame'] = step
        
        # 处理 WandB 逻辑：加上前缀以在网页端区分 train/eval 曲线图
        if self.use_wandb:
            wandb_data = {prefix + '/' + key: val for key, val in data.items()}
            self._dump_to_wandb(data=wandb_data)
        
        self._dump_to_csv(data)
        self._dump_to_console(data, prefix)
        # 清空状态：保证下一次计算的是新周期的数据，而不是历史累加
        self._meters.clear()

    def _dump_to_wandb(self, data):
        """
        将字典数据上传到 WandB 服务器。
        """
        wandb.log(data)


class Logger(object):
    """
    实验日志总管理类。
    
    主要功能:
        1. 统一管理训练（train）和评估（eval）两套指标系统。
        2. 协调多种后端：CSV 文件、TensorBoard (SummaryWriter) 和 WandB。
        3. 提供上下文管理器接口，简化实验主循环的代码量。
    """
    def __init__(self, log_dir, use_tb=False, use_wandb=False):
        """
        初始化 Logger 实例。

        参数说明:
            log_dir (Path): 日志存储的总目录。
            use_tb (bool): 是否启用 TensorBoard 记录。
            use_wandb (bool): 是否启用 WandB 记录。
        """
        self._log_dir = log_dir
        
        # 初始化训练指标组：负责记录所有以 'train' 开头的指标
        self._train_mg = MetersGroup(log_dir / 'train.csv',
                                     formating=COMMON_TRAIN_FORMAT,
                                     use_wandb=use_wandb)
        
        # 初始化评估指标组：负责记录所有以 'eval' 开头的指标
        self._eval_mg = MetersGroup(log_dir / 'eval.csv',
                                    formating=COMMON_EVAL_FORMAT,
                                    use_wandb=use_wandb)
        
        # 如果启用 TensorBoard，初始化 SummaryWriter
        if use_tb:
            self._sw = SummaryWriter(str(log_dir / 'tb'))
        else:
            self._sw = None
            
        self.use_wandb = use_wandb

    def _try_sw_log(self, key, value, step):
        """
        内部辅助函数：尝试向 TensorBoard 写入标量数据。
        """
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def log(self, key, value, step):
        """
        核心记录函数：将单个指标存入系统。

        参数说明:
            key (str): 指标键名，必须以 'train' 或 'eval' 开头（例如 'train/loss'）。
            value (any): 指标数值，支持 torch.Tensor 或普通数值。
            step (int): 当前全局步数。
        """
        # 强制性规范：确保日志类别明确
        assert key.startswith('train') or key.startswith('eval')
        
        # 如果传入的是 PyTorch 张量，将其转为 Python 标量以便记录
        if type(value) == torch.Tensor:
            value = value.item()
            
        # 1. 记录到 TensorBoard（如果可用）
        self._try_sw_log(key, value, step)
        
        # 2. 根据前缀，选择对应的数据管理器（MetersGroup）进行累加
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value)

    def log_metrics(self, metrics, step, ty):
        """
        批量记录指标。

        参数说明:
            metrics (dict): 包含多个键值对的字典（例如 {'loss': 0.1, 'acc': 0.9}）。
            step (int): 当前步数。
            ty (str): 类别前缀，'train' 或 'eval'。
        """
        for key, value in metrics.items():
            # 自动拼接前缀并转发给 log 函数
            self.log(f'{ty}/{key}', value, step)

    def dump(self, step, ty=None):
        """
        清算函数：将缓存中的平均指标清空并输出到控制台、CSV 和 WandB。

        参数说明:
            step (int): 当前步数。
            ty (str): 指定清算哪一类指标。None 表示全部清算，'train' 或 'eval' 表示指定清算。
        """
        if ty is None or ty == 'eval':
            self._eval_mg.dump(step, 'eval')
        if ty is None or ty == 'train':
            self._train_mg.dump(step, 'train')

    def log_and_dump_ctx(self, step, ty):
        """
        上下文管理器工厂函数。
        
        逻辑说明:
            返回一个 LogAndDumpCtx 实例，用于支持 'with' 语法，
            实现自动化的 log 和随后的 dump 操作。
        """
        return LogAndDumpCtx(self, step, ty)


class LogAndDumpCtx:
    """
    日志记录与自动清空（Dump）上下文管理器。
    
    主要功能:
        通过 Python 的 contextmanager 协议（__enter__ 和 __exit__），
        确保在一个逻辑块（如一个训练回合）结束时，日志能够自动被写入磁盘/显示。
        它通过 __call__ 方法将自身变成一个可调用对象，简化了记录数据的语法。
    """

    def __init__(self, logger, step, ty):
        """
        初始化上下文管理器。

        参数说明:
            logger (Logger): 全局 Logger 对象的引用。
            step (int): 当前的全局训练帧数或步数（global_step）。
            ty (str): 日志类型前缀，通常为 'train' 或 'eval'。
        """
        self._logger = logger  # 持有 Logger 的引用，以便调用其方法
        self._step = step      # 记录当前的时间戳/步数
        self._ty = ty          # 记录数据的业务分类（训练还是评估）

    def __enter__(self):
        """
        进入 'with' 语句块时的逻辑。
        
        WARNING: 必须返回 self 才能在 'as' 后面的变量中使用该实例。
        
        返回:
            LogAndDumpCtx: 实例自身。
        """
        return self

    def __call__(self, key, value):
        """
        让实例可以像函数一样被直接调用。
        例如：log('reward', 10)
        
        内部逻辑:
            1. 自动拼接前缀：将 key 转化为 'train/reward' 这种完整路径。
            2. 转发调用：将处理好的 key、value 和 step 传给真正的 logger.log 函数。
            
        参数说明:
            key (str): 指标名称。
            value (any): 指标数值。
        """
        self._logger.log(f'{self._ty}/{key}', value, self._step)

    def __exit__(self, *args):
        """
        退出 'with' 语句块时的逻辑（无论是否发生异常都会执行）。
        
        INFO: 这是该类的核心价值，它自动触发了 dump 操作，
        保证了本周期内记录的所有指标被统一输出并清空缓存。
        
        参数说明:
            *args: 包含异常类型、异常值、追溯信息（如果有）。
        """
        self._logger.dump(self._step, self._ty)
