import os

import torch


class ModelCheckpoint(object):
    """ ModelCheckpoint handler can be used to periodically save objects to disk.

    This handler accepts two arguments:

        - an `ignite.engine.Engine` object
        - a `dict` mapping names (`str`) to objects that should be saved to disk.

    See Notes and Examples for further details.

    Args:
        dirname (str):
            Directory path where objects will be saved
        filename_prefix (str):
            Prefix for the filenames to which objects will be saved. See Notes
            for more details.
        save_interval (int, optional):
            if not None, objects will be saved to disk every `save_interval` calls to the handler.
            Exactly one of (`save_interval`, `score_function`) arguments must be provided.
        n_saved (int, optional):
            Number of objects that should be kept on disk. Older files will be removed.
        require_empty (bool, optional):
            If True, will raise exception if there are any files starting with `filename_prefix`
            in the directory 'dirname'
        create_dir (bool, optional):
            If True, will create directory 'dirname' if it doesnt exist.
        save_as_state_dict (bool, optional):
            If True, will save only the `state_dict` of the objects specified, otherwise the whole object will be saved.

    Notes:
          This handler expects two arguments: an `Engine` object and a `dict`
          mapping names to objects that should be saved.

          These names are used to specify filenames for saved objects.
          Each filename has the following structure:
          `{filename_prefix}_{name}_{step_number}.pth`.
          Here, `filename_prefix` is the argument passed to the constructor,
          `name` is the key in the aforementioned `dict`, and `step_number`
          is incremented by `1` with every call to the handler.

          If `score_function` is provided, user can store its absolute value using `score_name` in the filename.
          Each filename can have the following structure:
          `{filename_prefix}_{name}_{step_number}_{score_name}={abs(score_function_result)}.pth`.
          For example, `score_name="val_loss"` and `score_function` that returns `-loss` (as objects with highest scores
          will be retained), then saved models filenames will be `model_resnet_10_val_loss=0.1234.pth`.

    Examples:
        >>> import os
        >>> from libreid.engine_simple import Engine, Events
        >>> from libreid.checkpoint_simple import ModelCheckpoint
        >>> from torch import nn
        >>> trainer = Engine(lambda batch: None)
        >>> handler = ModelCheckpoint('/tmp/models', 'myprefix', save_interval=2, n_saved=2, create_dir=True)
        >>> model = nn.Linear(3, 3)
        >>> trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {'mymodel': model})
        >>> trainer.run([0], max_epochs=6)
        >>> os.listdir('/tmp/models')
        ['myprefix_mymodel_4.pth', 'myprefix_mymodel_6.pth']
    """

    def __init__(self, dirname, filename_prefix,
                 save_interval=None, 
                 n_saved=1,
                 require_empty=True,
                 create_dir=True,
                 save_as_state_dict=True):

        self._dirname = os.path.expanduser(dirname)
        self._fname_prefix = filename_prefix
        self._n_saved = n_saved
        self._save_interval = save_interval
        self._saved = []  # list of tuples (priority, saved_objects)
        self._iteration = 0
        self._save_as_state_dict = save_as_state_dict

        if create_dir:
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        # Ensure that dirname exists
        if not os.path.exists(dirname):
            raise ValueError("Directory path '{}' is not found".format(dirname))

        if require_empty:
            matched = [fname
                       for fname in os.listdir(dirname)
                       if fname.startswith(self._fname_prefix)]

            if len(matched) > 0:
                raise ValueError("Files prefixed with {} are already present "
                                 "in the directory {}. If you want to use this "
                                 "directory anyway, pass `require_empty=False`. "
                                 "".format(filename_prefix, dirname))

    def _save(self, obj, path):
        self._internal_save(obj, path)

    def _internal_save(self, obj, path):
        if not self._save_as_state_dict:
            torch.save(obj, path)
        else:
            if not hasattr(obj, "state_dict") or not callable(obj.state_dict):
                raise ValueError("Object should have `state_dict` method")
            torch.save(obj.state_dict(), path)

    def __call__(self, to_save):
        if len(to_save) == 0:
            raise RuntimeError("No objects to checkpoint found.")

        self._iteration += 1

        priority = self._iteration
        if (self._iteration % self._save_interval) != 0:
            return

        if (len(self._saved) < self._n_saved) or (self._saved[0][0] < priority):
            saved_objs = []

            suffix = ""
            for name, obj in to_save.items():
                fname = '{}_{}_{}{}.pth'.format(self._fname_prefix, name, self._iteration, suffix)
                path = os.path.join(self._dirname, fname)

                self._save(obj=obj, path=path)
                saved_objs.append(path)

            self._saved.append((priority, saved_objs))
            self._saved.sort(key=lambda item: item[0])

        if len(self._saved) > self._n_saved:
            _, paths = self._saved.pop(0)
            for p in paths:
                os.remove(p)
