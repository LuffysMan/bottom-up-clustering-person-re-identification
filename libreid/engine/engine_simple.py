import inspect
from enum import Enum
from collections import defaultdict


class Events(Enum):
    r"""Events that are fired by the :class:`ignite.engine.Engine` during execution"""
    EPOCH_STARTED = "epoch_started"
    EPOCH_COMPLETED = "epoch_completed"
    STARTED = "started"
    COMPLETED = "completed"
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"
    EXCEPTION_RAISED = "exception_raised"


class State(object):
    r"""An object that is used to pass internal and user-defined state between event handlers"""
    def __init__(self, **kwargs):
        self.iteration = 0
        self.output = None
        self.batch = None
        for k, v in kwargs.items():
            setattr(self, k, v)


class Engine(object):
    r"""一个通用的引擎. 主要的目的是, 在模型的每个iteration, 每个epoch的前后, 执行一些操作, 比如记录日志, 保存模型等.
    """
    def __init__(self, process_function):
        self.__process_function = process_function
        self.state = None
        self.__event_handlers = defaultdict(list)
        self.__allowed_events = set()
        self.register_events(*Events)

    def __fire_event(self, event_name):
        for func, args, kwargs in self.__event_handlers[event_name]:
            func(self, *args, **kwargs)

    def __run_once_on_dataset(self, data):
        self.state.iteration = 0
        for batch in data:
            self.state.iteration += 1
            self.__fire_event(Events.ITERATION_STARTED)
            self.state.output = self.__process_function(self, batch)
            self.__fire_event(Events.ITERATION_COMPLETED)


    def register_events(self, *event_names):
        for name in event_names:
            self.__allowed_events.add(name)

    def has_event_handler(self, handler, event_name=None):
        """Check if the specified event has the specified handler.

        Args:
            handler (Callable): the callable event handler.
            event_name: The event the handler attached to. Set this
                to ``None`` to search all events.
        """
        if event_name is not None:
            if event_name not in self.__event_handlers:
                return False
            events = [event_name]
        else:
            events = self.__event_handlers
        for e in events:
            for h, _, _ in self.__event_handlers[e]:
                if h == handler:
                    return True
        return False

    def add_event_handler(self, event_name, handler, *args, **kwargs):
        if event_name not in self.__allowed_events:
            raise ValueError("Event {} is not a valid event name for this engine".format(event_name))
        
        signature = inspect.signature(handler)
        signature.bind(self, *args, **kwargs)
        self.__event_handlers[event_name].append((handler, args, kwargs))

    def on(self, event_name, *args, **kwargs):
        def decorator(func):
            self.add_event_handler(event_name, func, *args, **kwargs)
            return func
        return decorator

    def run(self, data, start_epoch = 0, max_epochs = 1):
        self.state = State(epoch=start_epoch, max_epochs=max_epochs, metrics={})

        self.__fire_event(Events.STARTED)

        while self.state.epoch < max_epochs:
            self.state.epoch += 1
            self.__fire_event(Events.EPOCH_STARTED)
            self.__run_once_on_dataset(data)
            self.__fire_event(Events.EPOCH_COMPLETED)

        self.__fire_event(Events.COMPLETED)

        return self.state


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    loss_fn = nn.CrossEntropyLoss()

    N = 100
    C = 10
    BATCH_SIZE = 32
    data = torch.randn(N, C)
    labels = torch.empty((N,1), dtype=torch.long).random_(C)
    trainset = torch.cat([data, labels], dim=1)

    model = nn.Linear(10, C)
    data_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    
    def train_and_store_loss(engine, batch):
        inputs = batch[:,:-1]
        targets = batch[:, -1:].squeeze_().long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        return loss.item()

    trainer = Engine(train_and_store_loss)

    @trainer.on(Events.ITERATION_STARTED)
    def do_at_iteration_started(engine):
        print("iteration {} start".format(engine.state.iteration))

    @trainer.on(Events.ITERATION_COMPLETED)
    def do_at_iteration_completed(engine):
        print("iteration {} complete".format(engine.state.iteration))

    @trainer.on(Events.EPOCH_STARTED)
    def do_at_epoch_started(engine):
        print("epoch {} start".format(engine.state.epoch))

    @trainer.on(Events.EPOCH_COMPLETED)
    def do_at_epoch_completed(engine):
        print("epoch {} complete".format(engine.state.epoch))

    trainer.run(data_loader)