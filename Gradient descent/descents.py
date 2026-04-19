import numpy as np
from abc import ABC, abstractmethod
from interfaces import LearningRateSchedule, AbstractOptimizer, LinearRegressionInterface

# ===== Learning Rate Schedules =====
class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        """
        returns: float, learning rate для iteration шага обучения
        """
        return self.lambda_*(self.s0/(self.s0+iteration)**self.p)


# ===== Base Optimizer =====
class BaseDescent(AbstractOptimizer, ABC):
    """
    Оптимизатор, имплементирующий градиентный спуск.
    Ответственен только за имплементацию общего алгоритма спуска.
    Все его составные части (learning rate, loss function+regularization) находятся вне зоны ответственности этого класса (см. Single Responsibility Principle).
    """

    def __init__(self,
                 lr_schedule: LearningRateSchedule = TimeDecayLR(),
                 tolerance: float = 1e-6,
                 max_iter: int = 1000
                 ):
        self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.iteration = 0
        self.model: LinearRegressionInterface = None

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """
        Вычисляет обновление согласно конкретному алгоритму и обновляет веса модели, перезаписывая её атрибут.
        Не имеет прямого доступа к вычислению градиента в точке, для подсчета вызывает model.compute_gradients.

        returns: np.ndarray, w_{k+1} - w_k
        """
        pass

    def _step(self) -> np.ndarray:
        """
        Проводит один полный шаг интеративного алгоритма градиентного спуска

        returns: np.ndarray, w_{k+1} - w_k
        """
        delta = self._update_weights()
        self.iteration += 1
        return delta

    def optimize(self) -> None:
        """
        Оркестрирует весь алгоритм градиентного спуска.
        """
        self.model.loss_history.append(self.model.compute_loss())
        while (self.iteration < self.max_iter):
            delta = self._step()
            self.model.loss_history.append(self.model.compute_loss())
            
            if (np.linalg.norm(delta)**2<self.tolerance):
                break

            if np.any(np.isnan(self.model.w)):
                break


        # в конце также приcваивает атрибуту модели полученный loss_history


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        gradient = self.model.compute_gradients()
        n = self.lr_schedule.get_lr(self.iteration)
        self.model.w -= n*gradient
        return - n*gradient


class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать стохастический градиентный спуск
        # 1) выбрать случайный батч
        random_array = np.random.choice(self.model.X_train.shape[0], size=self.batch_size, replace=False)
        random_batch = self.model.X_train[random_array]
        random_target = self.model.y_train[random_array]
        # 2) вычислить градиенты на батче
        gradient = self.model.compute_gradients(random_batch,random_target)
        # 3) обновить веса модели
        n = self.lr_schedule.get_lr(self.iteration)
        self.model.w -= n* gradient
        return - n * gradient


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_memory = None
        self.grad_sum = None
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape

        if self.grad_memory is None:
            self.grad_memory = np.zeros((num_objects, num_features))
            self.grad_sum = np.zeros_like(self.model.w)
            
        random_array = np.random.choice(num_objects, size=self.batch_size, replace=False)   

        for j in random_array:
            gradient = self.model.compute_gradients(X_train[j:j+1],y_train[j:j+1])
            self.grad_sum += (gradient - self.grad_memory[j])/num_objects
            self.grad_memory[j] = gradient
        n = self.lr_schedule.get_lr(self.iteration)
        self.model.w = self.model.w - n * self.grad_sum
        return -n * self.grad_sum



class MomentumDescent(BaseDescent):
    def __init__(self, *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(self.model.w)
        
        gradient = self.model.compute_gradients()
        n = self.lr_schedule.get_lr(self.iteration)
        self.velocity *= self.beta
        self.velocity += n * gradient
        self.model.w -= self.velocity
        return -self.velocity



class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        gradient = self.model.compute_gradients()
        if self.m is None or self.v is None:
            self.m,self.v = np.zeros_like(self.model.w),np.zeros_like(self.model.w)


        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        M = self.m / (1 - self.beta1 ** (self.iteration+1))
        V = self.v / (1 - self.beta2 ** (self.iteration+1))
        t =self.lr_schedule.get_lr(self.iteration) * M / (np.sqrt(V) + self.eps)
        self.model.w -= t
        return -t

# ===== Non-iterative Algorithms ====
class AnalyticSolutionOptimizer(AbstractOptimizer):
    """
    Универсальный дамми-класс для вызова аналитических решений 
    """

    def __init__(self):
        self.model = None

    def optimize(self) -> None:
        X = self.model.X_train
        y = self.model.y_train
        self.model.w = self.model.loss_function.analytic_solution(X,y)


        